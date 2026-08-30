前面的笔记都在讲"单个数字长什么样"：fp8、bf16、int8、fp4 每种格式的位布局和精度。但决定一个模型快不快、准不准的，是**这些数字在矩阵乘法（GEMM）里怎么被乘起来、加起来的**。神经网络里绝大部分计算量都是矩阵乘法，所以"低精度矩阵乘"才是这些格式发挥价值的地方。想先看清整个精度家族，见 [[learning/precision/01-overview|精度总览]]。

> **低精度矩阵乘的关键是乘法和累加在哪个精度进行，输入精度反而是次要的。** 张量核心用低精度输入、高精度累加，精度问题集中在三处：输入怎么量化、累加放哪个精度、结果怎么缩放回去。

## 为什么矩阵乘法是重点

推理和训练的计算量几乎全在 GEMM 上：attention 的 QK^T、softmax × V、FFN 的 Wx、以及各种投影。所以加速计算 = 加速 GEMM，而加速 GEMM 靠的是**张量核心**（tensor core），一种专门做"一个小的矩阵块相乘并累加"的硬件单元。

张量核心天生为低精度设计：低精度输入能塞进更宽的加法树，单位周期吞吐更高。同一块 H100 上，fp16 的密集算力约 1,979 TFLOPS，FP8 约 3,958 TFLOPS（都算上稀疏），正好翻倍，这就是"用精度换吞吐"的硬件版。

## 张量核心怎么做低精度乘

一次低精度 GEMM 在硬件上被拆成很多次小型的 **MMA（matrix-multiply-accumulate）**。以 fp16 为例，一次 HMMA 处理一个很小的 tile（如 16×8×16），把 A、B 的乘法送进一个**加法树**里就地累加，累加器用更高的精度（通常是 fp32）。

- **输入精度**：A、B 用低精度（fp8 / fp16 / int8 / fp4）。
- **累加精度**：乘积和累加在 fp32（或 int32）里进行，这一步几乎"免费"，因为加法树比乘法宽得多。
- **为什么要高精度累加**：如果累加也用低精度（比如 fp16 累加 fp16 乘积），每加一次就又要舍入一次，$K$ 次累加的舍入误差会和输入的量化误差叠在一起，把精度彻底毁掉。fp32 累加把"累加舍入"这一项直接消灭，只剩下输入的量化误差。

## 混合精度矩阵乘：三个精度位各自独立

前面讲的"低精度矩阵乘"里，A、B 都走同一种低精度，只有累加留在 fp32。**混合精度**（mixed precision）把这个思路推得更广：**输入 A、输入 B、累加器这三个精度点，可以各自独立选择**。所谓"混合"，就是这三个位选了不同的精度。

| 精度点 | 通常是 | 为什么 |
| --- | --- | --- |
| A 的精度 | fp16 / fp8 / int8 / fp4 | 吃吞吐、省带宽的关键 |
| B 的精度 | 可与 A 不同 | 权重与激活的可压性不一样 |
| 累加精度 | **fp32 / int32** | 杀灭累加舍入，只留下输入量化项 |

![GEMM 的三个精度点：A、B、累加](/learning/assets/matmul-mixed-precision.svg)

"混合精度"分两类，逻辑不同：

**第一类：训练混合精度（fp16 / bf16 计算 + fp32 主权重）。** 这是 ML 里"混合精度"最经典的含义："混合"发生在**权重与计算**之间，而不是矩阵乘的两个输入之间。主权重留在 **fp32**（每步用 fp32 更新，误差不累积），计算量最大的**矩阵乘**用 **fp16 / bf16** 跑，累加用 **fp32**，fp16 时还要配合 **loss scaling** 防下溢。详见 [[learning/precision/09-fp16|FP16]] 与 [[learning/precision/10-bf16|BF16]]。

**第二类：非对称操作数精度（asymmetric operand precision）。** 这才是**矩阵乘内部**的混合。两个输入用不同精度，因为它俩的"可压性"不一样：

- **权重**：静态、可校准、通常更"听话"，能压得更低。
- **激活**：动态、随输入变、块内动态范围大，通常要留更高精度。

| 组合 | A（权重） | B（激活） | 典型场景 |
| --- | --- | --- | --- |
| W8A16 | fp8 / int8 | fp16 | 权重大幅压，激活保精度 |
| FP4 + FP8 | [[learning/precision/03-mxfp4\|MXFP4]] | [[learning/precision/07-mxfp8\|MXFP8]] | 权重压到 4 bit，激活用 8 bit 块尺度兜动态范围 |
| fp16 × bf16 | fp16 | bf16 | 少见，需硬件支持 |

例如 OCP MX 论文里的常见搭配就是 **MXFP4 权重 + MXFP8 激活**：权重压到极限的 4 bit，激活用 8 bit 块尺度兜住更宽的动态范围，这比"两个都用 4 bit"稳得多。

**为什么混合精度有效。** 误差的主要来源是**输入量化**（见下一节"输入的量化"），而累加用的是高精度。所以把**最容易出错、最需要动态范围的地方**（激活、累加）留在高精度，把**最重、最可压缩的地方**（权重）压到低精度，就能在精度损失可控的前提下拿到吞吐。这也解释了为什么"两个输入都压到极限"（如 FP4 × FP4 直接推理）往往崩溃，而"不对称混合"能撑住。

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
- **每通道**对权重特别有效：权重分布通常按通道差异很大。
- **每块**（OCP MX，见 [[learning/precision/07-mxfp8|MXFP8]]）用每 32 个元素一个共享指数，兼顾动态范围和开销，是下一代的主流。

动态缩放在推理里按 batch 现算（用 `amax`），训练里常用"延迟缩放"：用上一步的 amax 作为这一部的 scale，避免同步等待。

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

[[learning/precision/16-ozaki-scheme|Ozaki Scheme]] 把一个数**拆成一串 int8 残差**，每个残差都是精确的（误差自由变换），再用 int8 张量核把这些残差矩阵乘起来，最终合成接近 fp32 的结果。它用 int8 的吞吐拿到 fp32 的精度，代价是计算量增大（残差个数倍）。**这个思路已被硬件吸收**：NVIDIA Rubin 的 Tensor Core 原生提供"模拟（emulated）FP32/FP64"，就是靠低位宽核拆分量去逼近原生精度；软件版就是 Ozaki 式的误差自由分解。

- 适用：需要 fp32 / fp64 精度但又想在低精度张量核上跑的 GEMM（科学计算、参考实现）。在 Blackwell/Rubin 上，simulated 高精度由硬件/库承担；在只有 INT8 核的旧硬件上则靠手写 Ozaki。

## 低精度矩阵乘的误差有多大

假设输入量化到相对精度 $u$：每个元素被舍入成 $\pm u$ 档的误差，$\hat A\hat B$ 就是这些带误差项的随机和。在 fp32 累加、误差不是同向对齐时，相对误差近似为

$$
\frac{\|\hat A\hat B - AB\|}{\|AB\|} \approx u \cdot c
$$

$c$ 是一个**分布常数**，取决于元素的动态范围、有没有离群值、用整体缩放还是逐通道/逐块缩放。关键是它**不随 $K$ 增长**：误差是 $K$ 个独立小项的随机和，分子 $\sim\sqrt{\sum_k(A_{jk}B_{kl})^2}$ 和分母 $\|AB\|$ 都以 $\sqrt K$ 的节奏增长，相对误差就被消掉了。只有误差**同向对齐**（系统性舍入偏差）时，才会涨到 $\sim\sqrt K\,u$。

实测最直观。下面用 per-tensor E4M3 量化一个随机矩阵乘，先看相对误差随 $K$ 怎么变，再看一个离群行把 per-tensor 和 per-row（per-token 缩放）拉开多大差距：

```python
import numpy as np

def q_tensor(x, max_abs=448.0):
    amax = np.abs(x).max()
    s = amax / max_abs if amax else 1.0
    q = np.clip(np.round(x / s), -max_abs, max_abs)
    return q.astype(np.float32), s

def q_rows(x, max_abs=448.0):          # per-row（per-token / per-channel）缩放
    amax = np.abs(x).max(axis=1, keepdims=True)
    s = np.where(amax == 0, 1.0, amax / max_abs)
    q = np.clip(np.round(x / s), -max_abs, max_abs)
    return q.astype(np.float32), s

rng = np.random.default_rng(0)

print("--- rel_err vs K (no outlier, per-tensor) ---")
for K in (256, 1024, 4096, 8192):
    A = rng.normal(0, 1, (128, K)).astype(np.float32)
    B = rng.normal(0, 1, (K, 128)).astype(np.float32)
    C = A @ B
    Aq, sa = q_tensor(A); Bq, sb = q_tensor(B)
    Cq = (Aq @ Bq) * (sa * sb)
    print(f"K={K:5d}  rel_err={np.linalg.norm(Cq-C)/np.linalg.norm(C):.4f}")

print("--- one outlier row: per-tensor vs per-row ---")
A = rng.normal(0, 1, (256, 1024)).astype(np.float32)
A[0] *= 100
B = rng.normal(0, 1, (1024, 256)).astype(np.float32)
C = A @ B
Aq, sa = q_tensor(A); Bq, sb = q_tensor(B)
print(f"per-tensor rel_err={np.linalg.norm((Aq@Bq)*(sa*sb)-C)/np.linalg.norm(C):.4f}")
Ar, sr = q_rows(A)
print(f"per-row    rel_err={np.linalg.norm((Ar@Bq)*(sr*sb)-C)/np.linalg.norm(C):.4f}")
```

```text
K=  256  rel_err=0.0042
K= 1024  rel_err=0.0041
K= 4096  rel_err=0.0044
K= 8192  rel_err=0.0046
per-tensor rel_err=0.0362
per-row    rel_err=0.0039
```

看两个结论：

- **$K$ 从 256 到 8192，相对误差基本钉在约 0.4%**，并没有随 $K$ 变大。所以 attention 里 $QK^\top$ 的 $K=\text{seqlen}$ 很大，并不是"直接放大相对误差"的原因——真正让误差暴涨的是**动态范围和离群值**。
- **一个离群行（放大 100 倍）**：per-tensor 一下从 0.4% 涨到 3.6%（$s$ 被 outlier 撑大，其余行被压进很少的档位），per-row 缩放基本不变（0.4%）。这就是"离群值拖累全局缩放"的实测。

顺带，代码最后那步 `(sa * sb)` 就是**输出缩放**：把量化结果乘回两个 scale，得到近似 fp32 的输出。这个乘在真实内核里通常和 bias、激活一起融进 **epilogue**；但如果 scale 本身也用低精度保存（比如 fp8/bf16 的 scale），这一步会再引入一次舍入。

判断误差可以归纳成几条：

- **相对误差 $\approx u\cdot c$，与 $K$ 基本无关**（随机输入）。要压误差，优先压低 $u$（更深位宽、更细缩放）或压 $c$（消离群值、窄动态范围）。
- **累加用 fp32** 把"累加舍入"消掉后，剩下的才是输入量化项；**低精度累加**才会让误差随 $K$ 累积（见下一节）。
- **分层量化 / 块缩放**压的是 $c$ 里"被全局 outlier 拖累"的部分，并不能消掉 $u$ 本身。

## K 维怎么累加：split-K 与归约顺序

前面"累加精度"说的是**位宽**，但**累加的顺序**同样影响结果。真实 GEMM 不会把 $K$ 一次全乘完，而是沿 $K$ 切成块：

- **K-tiling**：把 $A$、$B$ 沿 $K$ 切成若干 chunk，每个 chunk 对应一组 MMA；一个 CTA 处理一个 $(M_{\text{tile}}, N_{\text{tile}})$ 输出块，沿 $K$ 逐步累加进本地的 fp32 accumulator。
- **split-K**：当 $K$ 非常大而 $M\times N$ 偏小（或为了让 tile 更小、塞进更多 CTA 并行）时，把 $K$ 拆给多个 CTA，每个算一份 partial sum，最后再合并。

合并的方式决定结果**是不是确定**：

- **固定顺序（顺序累加 / 固定归约树）**：partial 按固定顺序相加，结果可复现。张量核心内部的加法树本来就是固定树序，舍入误差也最低。
- **atomic 累加**：多个 CTA 算完 partial，直接 atomically 加进同一个 fp32 输出 buffer——**谁先到谁先加**，求和顺序不稳定，每次运行结果会差在最后几位。fp32 累加下这点差异通常可接受；但换成低精度累加，顺序带来的差异会被放大，跑出来不再可复现。

所以生产内核里的 split-K 合并，几乎都在 fp32 / int32 accumulator 上做固定归约，避免"低精度累加 + 乱序"把误差和不确定性同时放大。这也是把**累加精度和归约顺序**一起放进"怎么选"的原因——它们共同决定误差会不会随 $K$ 累积，以及每次跑出来是不是同一个数。

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

`K = 4096` 时，fp8 的相对误差通常在 $10^{-2}$ 量级，比 fp32 的 $10^{-6}$ 差约 4 个数量级，但吞吐翻倍或更多。真实系统还会用每通道缩放、clipping calibration 或块缩放把误差往下压。

## 一些 tips

1. **先看分布和离群值，再定格式**。离群值越多、动态范围越宽，越低精度越危险；$K$ 只影响绝对误差和跨层传播，随机输入下相对误差约 $u\cdot c$。
2. **累加永远用高精度**（fp32/int32）。任何"低精度累加"都在同时毁掉输入和累加两层精度。
3. **per-tensor 要小心 outlier**。一个异常大的数就能撑大 $s$，让整块精度崩塌；用 per-channel / per-block 或 clipping 校准。
4. **训练用延迟缩放，推理用动态缩放**。避免为了等 amax 而同步，吞吐损失往往比精度损失更痛。
5. **量化不是免费的误差**。fp8 的 $u \approx 6\%$ 在长上下文 attention 里，容易被离群值和跨层传播放大成可见的偏差，别无脑用 fp8。

## Reference

- FP8 Formats for Deep Learning（arXiv:2209.05433）：<https://arxiv.org/abs/2209.05433>
- Microscaling Data Formats for Deep Learning（arXiv:2310.10537）：<https://arxiv.org/abs/2310.10537>
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale（arXiv:2208.07339）：<https://arxiv.org/abs/2208.07339>
- DGEMM on Integer Matrix Multiplication Unit（Ozaki scheme，arXiv:2306.11975）：<https://arxiv.org/abs/2306.11975>
- NVIDIA Transformer Engine（张量核心低精度）：<https://github.com/NVIDIA/TransformerEngine>
- CUTLASS：高性能 GEMM 内核模板（K-tiling / split-K / 累加与归约顺序）：<https://github.com/NVIDIA/cutlass>
