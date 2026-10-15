---
aiGenerated: true
---

NVIDIA 张量核心上的 FP8 是一套用法：从 Hopper（H100）开始，张量核心把 E4M3 和 E5M2 两种 8 位格式混着用，配合逐张量的动态缩放（amax → scale），矩阵乘吞吐约为 FP16 的 2 倍，同时保住训练精度。

> 这套工程配方的做法是：前向用更精的 E4M3，反向用范围更大的 E5M2，每个张量再按自己的 amax 算一个缩放因子，把值塞进 8 位。格式本身的推导见 [[learning/precision/05-fp8|FP8]]；这里只看张量核心怎么用它。

### 位布局（简要）

FP8 在张量核心上有两种编码，前向/反向各挑一种：

| 格式 | sign | exponent | mantissa | 总计 | 指数偏置 | 最大有限值 | 特殊值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E4M3 | 1 | 4 | 3 | 8 | 7 | 448 | 仅 NaN，无 ±inf |
| E5M2 | 1 | 5 | 2 | 8 | 15 | 57344 | ±inf + NaN |

普通数的表示和 FP16 同构，只是字段更窄：

$$
x = (-1)^s \cdot 2^{E-\text{bias}} \cdot (1.f)_2
$$

- E4M3（bias = 7）：值域窄但尾数宽，适合前向的激活和权重。
- E5M2（bias = 15）：指数多一位、范围大，但只留 2 位尾数，适合反向的梯度；梯度需要更大的动态范围，对精度不那么敏感。

完整的格式推导、次正规数，以及 E4M3 为什么没有无穷的编码细节，见 [[learning/precision/05-fp8|FP8]]。

### 关键数值

| 格式 | 有效精度（significand bits） | 单位舍入 $u$ | 十进制有效位 | 最小正规数 | 最大有限值 |
| --- | --- | --- | --- | --- | --- |
| E4M3 | 4（3 尾数 + 隐式 1） | $2^{-4} = 0.0625$ | ≈1.2 | $2^{-6} \approx 1.56\times10^{-2}$ | 448 |
| E5M2 | 3（2 尾数 + 隐式 1） | $2^{-3} = 0.125$ | ≈0.9 | $2^{-14} \approx 6.10\times10^{-5}$ | 57344 |

两者是交换关系：E4M3 精度高，但对“值有多大”敏感；E5M2 范围大，但精度更低。没有任何一种 8 位格式能同时守住精度和范围，所以训练时要把两种混着用。

### 和相邻格式对比

| 格式 | 指数位 | 尾数位（含隐式） | 单位舍入 | 十进制有效位 | 最大有限值 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 5 | 11 | $2^{-11}$ | ≈3.3 | 65504 | 训练基线 |
| bf16 | 8 | 8 | $2^{-8}$ | ≈2.4 | $3.4\times10^{38}$ | 范围同 fp32 |
| tf32 | 8 | 11 | $2^{-11}$ | ≈3.3 | $3.4\times10^{38}$ | Ampere+，存 19 位 |
| E4M3 | 4 | 4 | $2^{-4}$ | ≈1.2 | 448 | 前向 |
| E5M2 | 5 | 3 | $2^{-3}$ | ≈0.9 | 57344 | 反向 |

单看精度，FP8 的单位舍入（E4M3 $2^{-4}$、E5M2 $2^{-3}$）比 FP16 的 $2^{-11}$ 差约两个数量级，所以它只适合做矩阵乘这块最重、量最大的活。

### 动态缩放：amax 与“防裁剪”

FP8 的问题在于范围太窄。一个层里如果某次激活振幅特别大，量化到 8 位就会裁剪（clip）到 448，信息全丢；反过来整体都很小，量化又会被噪声淹没。所以每个 8 位张量都要配一个缩放因子：

$$
x_{fp8} = \left\lfloor \frac{x_{fp32}}{scale} \right\rceil,\qquad
scale = \frac{amax}{X_{max}}
$$

其中 $X_{max}$ 是当前格式的最大值（E4M3 用 448，E5M2 用 57344），$amax$ 是该张量中绝对值最大的元素。scale 是 FP32 标量，每个张量一个。

“先知道 amax 再量化”要扫两遍数据，代价不小。NVIDIA 的解法是 delayed scaling（延迟缩放）：用最近若干次迭代记录的 amax 历史估计下一次的 scale，训练时拿上一次的估计直接量化，同时把本次真实 amax 追加进历史。这就是 Transformer Engine（TE）的配方里有个 `amax_history_len` 的原因。

> 对照：MXFP8 把“一个张量一个 scale”改成“每 32 个连续元素一个 scale”，缩放因子用 8 位的 E8M0 存。scale 细化后动态范围需求骤降，所有值都能用更精的 E4M3，不用再靠 E5M2 兜底。这就是 [[learning/precision/07-mxfp8|MXFP8]] 的本质，也是 Blackwell 的新方向。

### 什么时候需要 / 什么时候不用

需要：

1. 大模型训练和推理里的矩阵乘。FFN、Attention 的 QKV/Proj 这些地方计算密度最高、最吃吞吐，把输入和权重压到 FP8，张量核心直接 2 倍吞吐。
2. 能承受校准成本。用 `amax`/延迟缩放要有历史迭代预热，提前跑几步或复用历史；推理侧可以用静态校准（离线跑一批数据定 scale），不需要在线调。
3. 8 位就能装下的张量。分布稳定、振幅不怎么突跳的层，量化后几乎无感。

不用：

1. 要精度的操作。LayerNorm、Softmax、残差累加、`exp`/`log` 这类操作要么是小矩阵，要么对相对误差极敏感，留在 FP32/FP16。
2. 范围随层剧烈变化。有的层 amax 小到 $10^{-3}$，下一层就大到 $10^{2}$，单一 scale 很难同时照顾；这时要么用逐通道 scale，要么直接上 [[learning/precision/07-mxfp8|MXFP8]]。
3. 科学计算和 HPC。要的是 FP64/F32 的精确度，不是吞吐，FP8 在这里帮倒忙。
4. 小矩阵或内存瓶颈。FP8 的 2 倍收益来自张量核心的算力，对 memory-bound 的小运算、逐元素 op 没有意义。

### 硬件与软件现状

- Hopper H100（首个 FP8 张量核心）：Transformer Engine 加持，张量核心支持任意混合的 FP8 输入格式，所以同一套训练循环里前向 E4M3、反向 E5M2 可以并存（recipe 里就叫 `Format.HYBRID`）。官方 H100 SXM 规格：FP16 张量核心 1,979 TFLOPS，FP8 张量核心 3,958 TFLOPS（均含稀疏），正好 2 倍。
- Blackwell（第五代）：张量核心原生支持 FP8（E4M3/E5M2）与社区定义的 MX 系列（MXFP8/FP6/FP4），并加入 NVFP4。NVIDIA 没有另立一个叫“NVFP8”的独立格式。FP8 的用法就是 E4M3/E5M2 混合配方加 amax 缩放。
- Rubin（Vera Rubin，Blackwell 的下一代）：沿用 Blackwell 的全部精度（NVFP4、FP64、TF32、BF16、FP16、FP8/FP6、INT8，与 Blackwell 那张表完全一致），并加上 Tensor Core 模拟（emulated）FP32/FP64。做法是把输入拆成低位分量，用低位宽张量核算，从而在低位宽核上逼近甚至超过原生 IEEE 精度。Rubin 与 Blackwell 向上兼容，已优化的代码可直接迁移。NVIDIA 对 Rubin 的定位是“增强的第五代 Tensor Core”（enhanced fifth-generation），而不是另起一代；增强的 Transformer Engine 提供 50 PFLOPS 的 NVFP4 推理吞吐。
- 存储格式 vs 纯计算：FP8 作为存储格式能省显存和带宽（激活、权重、KV cache 都能按 FP8 存），缩放因子本身是 FP32（每个张量一个），与数据一起维护。MXFP8 则把 scale 换成 E8M0，随每 32 元素一块存，于是存和算用同一套块结构。
- 框架支持：Transformer Engine（PyTorch / JAX）把 FP8 封装在 `fp8_autocast` 与 `DelayedScaling`/`MXFP8BlockScaling` 配方里；Megatron-LM、NeMo 训练栈、TensorRT-LLM / vLLM 推理栈、CUDA 均支持。CUDA 层面对纯手写 kernel 会暴露 `__nv_fp8` 类型与相关转换/缩放 intrinsic。

### 代码示例

用 Transformer Engine 在 H100 上跑一个 FP8（HYBRID：前向 E4M3、反向 E5M2）的 Linear 层。**这段需要 CUDA GPU + PyTorch + transformer_engine**，站点里的浏览器 Python runner 只带 NumPy，跑不了，所以下面不贴运行输出：

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# 配方：前向 E4M3，反向 E5M2；用最近 16 次迭代的 amax 估计 scale
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
)

x = torch.rand(1024, 768).cuda()
lin = te.Linear(768, 768, bias=True)   # 维度通常需能被 16 整除

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = lin(x)          # 输入 & 权重在进入张量核心前被量化为 FP8

y.mean().backward()     # 反向用 E5M2；由前向是否在 autocast 内决定
```

`fp8_autocast` 替你做了三件事：把 FP8 安全的操作转成 FP8、更新 amax 历史、算好下一次的 scale。

注意反向调用要写在 `fp8_autocast` 块外面：TE 会把块内的张量聚合起来做一次通信（在多卡训练里同步 scale 和 amax 历史），聚合要求 `backward()` 不能还留在这个上下文里。这不影响精度：反向用不用 FP8，由前向是不是在 `fp8_autocast` 里算决定（TE 文档：“the precision of the backward pass is determined by the precision of the forward pass”）。

下面是一段纯 Python 的 amax→scale 心智模型（不依赖 GPU），演示防裁剪的原理：

```python
import numpy as np

def to_fp8(x, amax, fmt="e4m3"):
    xmax = 448.0 if fmt == "e4m3" else 57344.0
    scale = (amax / xmax) if amax > 0 else 1.0
    q = np.clip(np.round(x / scale), -xmax, xmax)
    return q, scale

x = np.array([0.1, -0.6, 3.5, 0.02])
y1, s1 = to_fp8(x, amax=3.5)          # amax 准确 → 全部落在 448 内
y2, s2 = to_fp8(x, amax=0.6)          # amax 估小 → 3.5 撑爆量程
print("scale(3.5):", s1, "->", y1)
print("scale(0.6):", s2, "->", y2)
```

```text
scale(3.5): 0.0078125 -> [ 13. -77. 448.   3.]
scale(0.6): 0.0013392857142857143 -> [  75. -448.  448.   15.]
```

第一行里 `amax = 3.5` 正好取到真实最大值，`3.5 / 0.0078125 = 448`，刚好顶到量程但没溢出；第二行把 `amax` 估成 `0.6`（真实最大值的 $0.6/3.5 \approx 1/5.8$），scale 同比例变小，`3.5` 和 `-0.6` 一起裁到 $\pm 448$：这两个原本相差约 5.8 倍的数绝对值都顶到 448，相对大小彻底丢了，所以工程上宁肯 scale 偏大、不能偏小。

### 一些 tips

1. scale 不能想当然地算。`scale = amax / xmax` 用的是真实 amax；延迟缩放用的是历史估计，估计一旦偏小就会裁剪。宁可 scale 略大（损失一点量化精度），也不要 scale 偏小（直接裁剪）。
2. `amax_history_len` 是个权衡。它决定取历史窗口里哪个 amax：窗口太小，scale 对分布突变反应快，但容易被单次尖刺带偏；窗口太大，历史里长期混着早已不存在的极大值，scale 恒定偏大，有效位白浪费。TE 的默认值是 1024（配 `amax_compute_algo="max"`，即取窗口内最大的那个 amax），实际训练里很多配方会显式调小到几十到几百，让 scale 跟得更快；上面例子里的 16 属于偏激进的一端。
3. 2 倍吞吐只对张量核心矩阵乘成立。对逐元素、归一化、memory-bound 的运算没有收益，反而多一次量化成本。所以用 `fp8_autocast` 这种逐 op 开关的机制，别整图无脑 FP8。
4. E4M3 没有 ±inf，只有一个 NaN 模式；E5M2 有 ±inf 和 NaN。所以 E5M2 更能容忍算爆，E4M3 一旦上到 448 就直接饱和。
5. 注意转置和重排。到 Blackwell 的 MXFP8，块必须在归约维度上连续，转置要重量化，TE 会同时保留原始与转置两份拷贝。这是 FP8 简单、MXFP8 麻烦的典型差异。
6. FP8 打不了精度战。它只留 1～2 位十进制有效数字给矩阵乘；需要更多有效位的地方（残差、归一化、softmax）留在 FP16/FP32。

## Reference

- FP8 Formats for Deep Learning（arXiv:2209.05433）：<https://arxiv.org/abs/2209.05433>
- NVIDIA Transformer Engine（FP8 / delayed scaling）：<https://github.com/NVIDIA/TransformerEngine>
- Microscaling Data Formats for Deep Learning（arXiv:2310.10537）：<https://arxiv.org/abs/2310.10537>
