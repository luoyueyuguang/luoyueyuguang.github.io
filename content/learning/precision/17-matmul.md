前面的笔记都在讲"单个数字长什么样"——fp8、bf16、int8、fp4 每种格式的位布局和精度。但真正决定一个模型快不快、准不准的，不是单个数字，而是**这些数字在矩阵乘法（GEMM）里怎么被乘起来、加起来的**。神经网络里绝大部分计算量都是矩阵乘法，所以"低精度矩阵乘"才是这些格式真正发挥价值的地方。想先看清整个精度家族，见 [[learning/precision/01-overview|精度总览]]。

> 一句话给直觉：**低精度矩阵乘的关键不是"输入有多准"，而是"乘法和累加在哪个精度进行"。** 张量核心用低精度输入、高精度累加，所有的精度故事都集中在三个点上：输入怎么量化、累加放哪个精度、结果怎么缩放回去。

## 为什么矩阵乘法是重点

推理和训练的计算量几乎全在 GEMM 上：attention 的 QK^T、softmax × V、FFN 的 Wx、以及各种投影。所以加速计算 = 加速 GEMM，而加速 GEMM 靠的是**张量核心**（tensor core）——一种专门做"一个小的矩阵块相乘并累加"的硬件单元。

张量核心天生为低精度设计：低精度输入能塞进更宽的加法树，单位周期吞吐更高。同一块 H100 上，fp16 的密集算力约 1,979 TFLOPS，FP8 约 3,958 TFLOPS（都算上稀疏），正好翻倍——这就是"用精度换吞吐"的硬件版。

## 张量核心怎么做低精度乘

一次低精度 GEMM 在硬件上被拆成很多次小型的 **MMA（matrix-multiply-accumulate）**。以 fp16 为例，一次 HMMA 处理一个很小的 tile（如 16×8×16），把 A、B 的乘法送进一个**加法树**里就地累加，累加器用更高的精度（通常是 fp32）。

- **输入精度**：A、B 用低精度（fp8 / fp16 / int8 / fp4）。
- **累加精度**：乘积和累加在 fp32（或 int32）里进行——这一步几乎是"免费的"，因为加法树比乘法宽得多。
- **为什么要高精度累加**：如果累加也用低精度（比如 fp16 累加 fp16 乘积），每加一次就又要舍入一次，$K$ 次累加的舍入误差会和输入的量化误差叠在一起，把精度彻底毁掉。fp32 累加把"累加舍入"这一项直接消灭，只剩下输入的量化误差。

所以低精度矩阵乘的精度损失，**几乎全部来自输入量化**，而不是乘法或累加。

## 输入的量化：一个缩放因子的故事

把 fp32 输入变成低精度，核心就是"除以一个缩放因子再舍入"：

$$
q = \operatorname{round}\!\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot q
$$

$s$ 称为缩放因子（scale）。$s$ 怎么取，直接决定误差有多大：

| 缩放方式 | $s$ 怎么定 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 每张量（per-tensor） | 整个矩阵取 $\max \lvert x \rvert$ | 简单，开销极小 | 一个 outlier 会拉低全矩阵精度 |
| 每通道（per-channel） | 每个输出通道一个 $s$ | 精度明显更好 | 需要多存一份 scale，内核逻辑复杂 |
| 每块（per-block / MX） | 每 32 个元素共享一个 $s$ | 动态范围好、精度高 | 需要块布局，硬件支持较新 |

- **每张量**是最简单的，也是 fp8 推理的默认做法：$s = \operatorname{amax}/448$（E4M3 最大值）。缺点是一个离群值（outlier）就把整个 $s$ 撑大，其余数值都被压到很小的量化步长里。
- **每通道**对权重特别有效——权重分布通常按通道差异很大。
- **每块**（OCP MX，见 [[learning/precision/07-mxfp8|MXFP8]]）用每 32 个元素一个共享指数，兼顾动态范围和开销，是下一代的主流。

动态缩放在推理里按 batch 现算（用 `amax`），训练里常用"延迟缩放"——用上一步的 amax 作为这一部的 scale，避免同步等待。

## 三条低精度矩阵乘路线

同样是"做矩阵乘"，低精度下有三条工程路线，精度和复杂度递增：

### 1. 直接低精度：一个 scale 走天下

把输入整个量化到 fp8 / fp4 / int8，各用一个（或每通道一个）scale，喂给张量核心，结果乘回 scale 得到 fp32 输出。这是最常见、最快的路线。

- 优点：实现简单，几乎全是现成内核。
- 缺点：误差主要由**输入量化**决定（E4M3 的 $u=0.0625$，也就是约 6% 的相对误差）。一个离群值就可能毁了整块的精度。
- 适用：推理、对精度不敏感的大矩阵、已经用 calibration 摸清分布的模型。

### 2. 块缩放：MX 系列

用 [[learning/precision/07-mxfp8|MXFP8]] 或 [[learning/precision/03-mxfp4|MXFP4]] 的块共享指数，把动态范围从"每张量一个"细化到"每 32 元素一个"。输入量化误差从"被全局 outlier 拖累"变成"只在块内被影响"，精度提升明显，开销仅约 1/32。

- 适用：训练（MXFP8 已被用于训练轨迹）、离群值多、但对吞吐有要求的大 GEMM。

### 3. Ozaki / 误差自由分解：用 int8 拿 fp32 精度

[[learning/precision/16-ozaki-scheme|Ozaki Scheme]] 不把一个数量化成一个 int8，而是**拆成一串 int8 残差**，让每个残差都是精确的（误差自由变换），再用 int8 张量核把这些残差矩阵乘起来，最终合成接近 fp32 的结果。它用 int8 的吞吐拿到 fp32 的精度，代价是计算量增大（残差个数倍）。

- 适用：需要 fp32 精度但又想在 INT8 张量核上跑的 GEMM（如某些科学计算、参考实现）。

## 低精度矩阵乘的误差有多大

直觉上可以这样估算：假设输入量化相对误差 $\sim u$，那么一个内积（长度 $K$）的误差大约为

$$
\frac{\|\hat{A}\hat{B} - AB\|}{\|AB\|} \sim u \cdot c \cdot n
$$

其中 $n$ 是 $K$ 的一半（每次乘积两个输入各量化一次），$c$ 是一个与数值分布有关的常数。随机趋向下约 $\sqrt{K}\,u$，最坏约 $K\,u$。所以：

- **$K$ 越大，误差越大**——attention 里的 $QK^\top$ 在上下文很长时，$K = \text{seqlen}$ 会到几千，fp8 的误差会被放大。
- **累加用 fp32** 能把"累加舍入"这一项压到可忽略，剩下的就是输入量化项。
- **分层量化/块缩放**能把 $u$ 对动态范围敏感的部分压小，但不能消掉 $u$ 本身。

## 怎么选

| 场景 | 推荐 | 理由 |
| --- | --- | --- |
| 推理、吞吐优先 | fp8（E4M3）+ 每通道 scale | 简单、快、误差可接受 |
| 训练 | bf16 / fp16 混合精度（fp32 累加） | 范围够、精度够、误差可控 |
| 训练且要压显存 | [[learning/precision/07-mxfp8|MXFP8]] | 块缩放让训练轨迹更稳 |
| 权重大、离群值多 | 每通道 / 每块缩放 | 离群值不再拖累全局 |
| 要 fp32 精度但只有 int8 核 | [[learning/precision/16-ozaki-scheme|Ozaki]] | 精度和吞吐兼得，代价是计算量 |

## 代码示例

一个最简单的 fp8（E4M3）矩阵乘，用 per-tensor scale 量化 A、B，int/fp32 累加，再乘回 scale：

```python
import numpy as np

def quantize_fp8(x, max_abs=448.0):
    """per-tensor E4M3 量化：s = amax / max, q = clamp(round(x/s), -max, max)."""
    amax = np.abs(x).max()
    if amax == 0:
        return x.astype(np.float32), 1.0
    s = amax / max_abs
    q = np.clip(np.round(x / s), -max_abs, max_abs)
    return q.astype(np.float32), s   # q 实际上可以存成 int8 位模式

def fp8_matmul(A, B):
    Aq, sa = quantize_fp8(A)
    Bq, sb = quantize_fp8(B)
    # 张量核里是低精度输入、fp32/int32 累加
    C = (Aq.astype(np.float32) @ Bq.astype(np.float32)) * (sa * sb)
    return C

rng = np.random.default_rng(0)
A = rng.normal(0, 1, (256, 4096)).astype(np.float32)
B = rng.normal(0, 1, (4096, 256)).astype(np.float32)
C8 = fp8_matmul(A, B)
C32 = A @ B
rel_err = np.linalg.norm(C8 - C32) / np.linalg.norm(C32)
print(f"fp8 matmul relative error: {rel_err:.4f}")   # 通常为 1e-2 量级，取决于 K 和分布
```

`K = 4096` 时，fp8 的相对误差通常在 $10^{-2}$ 量级——比 fp32 的 $10^{-6}$ 差约 4 个数量级，但吞吐翻倍或更多。真实系统还会用每通道缩放、clipping calibration 或块缩放把误差往下压。

## 一些 tips

1. **先看 $K$ 和分布，再定格式**。内积长度越长、离群值越多，越低精度越危险。
2. **累加永远用高精度**（fp32/int32）。任何"低精度累加"都在同时毁掉输入和累加两层精度。
3. **per-tensor 要小心 outlier**。一个异常大的数就能撑大 $s$，让整块精度崩塌；用 per-channel / per-block 或 clipping 校准。
4. **训练用延迟缩放，推理用动态缩放**。避免为了等 amax 而同步，吞吐损失往往比精度损失更痛。
5. **量化不是免费的误差**。fp8 的 $u \approx 6\%$ 在 $K$ 很大会被放大成可见的偏差，别在长上下文的 attention 上无脑用 fp8。
