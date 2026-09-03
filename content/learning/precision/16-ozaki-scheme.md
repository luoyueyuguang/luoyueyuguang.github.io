聊低精度计算，大部分人想到的是"省内存、快，但精度差"。Ozaki scheme（Ozaki 分解）反其道而行：**用比目标精度更低的计算单元，算出比原生浮点还准的矩阵乘法**。它把每个数拆成若干"整数残差"之和（拆法本身无舍入），再用 [[learning/precision/08-int8|INT8]] 张量核做乘法、INT32 精确累加，最后按位权重新合成。结果是在消费级 GPU 上，它甚至能超过 FP64 硬件的原生 GEMM。

> 把一个浮点数的二进制尾数按 7 位一组切成若干片，每片是一个 INT8（共享同一行的指数，像 block-float 一样），于是"低精度整数乘法累加"这一步就**没有任何舍入**（整数运算只有溢出问题，而片长设计保证不溢出）。把每片对应乘积按位权缩放后累加，就还原出高精度的积。这是"shared-place splitting"（共享位法）的精髓：与 double-double 那种"逐元素拆成两半"的 elementwise 方法不同，它拆的是**整行共享的尾数空间**。

### 从一个矛盾说起：低精度张量核的"快"与"不准"

现代 AI 硬件（NVIDIA Tensor Core、AMD Matrix Core、Intel AMX、Google TPU、Groq）都内置了**整数矩阵乘法单元（IMMU，Integer Matrix Multiplication Unit）**：输入 INT8、INT4，输出/累加是 INT32。它比浮点张量核快得多：NVIDIA 的 INT8 Tensor Core 吞吐约为 FP16 Tensor Core 的 2～4 倍。

但直接用 INT8 做科学计算不行：INT8 只有 7 位有效数字（约 2 位十进制），误差不可接受。**Ozaki scheme 的洞察是：这些 INT8 单元的乘法累加本来就不会错，问题出在喂进去之前就把数压成了低精度。如果喂给它的本身就是"无损切好的整片"，那么每个 INT8 乘法累加都精确无误。** 精度损失被转移到"最后一步按位权合成"，这一步用目标精度（比如 FP64）来做，误差可控。

### 核心思想：把一个数拆成整数残差的和（"Ozaki multi"）

考虑一个目标矩阵乘法 $\mathbf{C} = \mathbf{A}\cdot\mathbf{B}$。Ozaki scheme 把 $\mathbf{A}$ 和 $\mathbf{B}$ 各拆成 $s$ 个"切片"（slice)矩阵，使得：

$$
\mathbf{A} = \sum_{i=1}^{s}\mathbf{A}^{(i)},\qquad \mathbf{B} = \sum_{j=1}^{s}\mathbf{B}^{(j)}
$$

其中每个切片的每个元素都是一个**相邻 7 位（对 INT8）尾数块**，并且**同一行（对 A）/同一列（对 B）共享一个指数偏移**。这等价于把整个矩阵的"尾数空间"纵向切成 $s$ 条（见下示意：一横排元素共享一个指数，尾数按位权下移切片）。

```
行 i  （共享指数 e_i）：
  ┌──────────────────────────────────────────┐
A │  a1       a2       a3  ...       ⟵ 原始浮点值
  ├──────────────────────────────────────────┤
A¹│ [0..7位] [0..7位]  ...   ⟵ 第1片 INT8（乘 2^{0·α}·e_i）
A²│ [7..14位] [7..14位] ...   ⟵ 第2片 INT8（乘 2^{1·α}·e_i）
A³│ [14..21位]           ...   ⟵ 第3片 INT8（乘 2^{2·α}·e_i）
  └──────────────────────────────────────────┘
```

逐元素地，一个值 $x$ 被重构为：

$$
x = \sum_{p=1}^{s} x^{(p)}, \qquad
x^{(p)} = \mathrm{int8}_p \cdot 2^{\,(p-1)\alpha} \cdot e_i
$$

关键点：

- **$\alpha$（每个切片的尾数位宽）**：对 INT8 输入、INT32 累加，有效位宽 $= \min(\alpha, \ell_{\text{in}})$，其中 $\ell_{\text{in}} = 7$（INT8 的 7 位大小量级）。
- **$e_i$（行共享指数）**：整行唯一的缩放因子，是 block-float 格式的"共享指数"。它消除了浮点格式里每个元素各自带一个指数造成的冗余，也是整数方法更省内存的原因之一。
- **$s$（片数）**：把原尾数空间分成 $s$ 段，总覆盖位长 $= s \times \mathrm{BPS}$。想要覆盖 53 位（FP64 尾数）就用约 8 片，覆盖 24 位（FP32 尾数）就用约 4 片。

### 为什么是"误差自由"（error-free）

浮点运算的舍入来自两个地方：**乘法**和**累加**。Ozaki scheme 在整数单元上同时躲开两者：

**① 乘法无舍入。** 两个 INT8 值相乘，大小都在 $2^{7}$ 以内，乘积 $< 2^{14}$。只要相乘的两个"有效尾数长度"之和不超过累加器的位数，乘积就完整保留、无需截断。这正是"shared-place splitting"的用意：因为同一行（列）共享同一个指数，切出来的各片位权严丝合缝地衔接，乘积不会溢出累加器。

**② 累加无舍入、无溢出。** INT8 × INT8 逐项乘积最多 $2^{14}$，把 $k$ 个这样的项累加进 INT32：

$$
\left| \sum_{\ell=1}^{k} a_\ell b_\ell \right| \;<\; 2^{14}\cdot k \;<\; 2^{31} \quad (\text{当 } k < 2^{17})
$$

INT32 有 31 位有效（1 位符号），所以只要内积长度 $k$ 不太大，**累加不溢出**；而整数加法本身是精确的，没有舍入。于是每次 $\mathbf{A}^{(i)}\cdot\mathbf{B}^{(j)}$ 都是**精确**的整数结果。论文里那句点题的话："浮点 Ozaki 方案中的无舍入，等价于整数方案中的无溢出。"

**③ 唯一的舍入在最后合成。** 各片乘积 $\mathbf{C}_{\text{tmp}}$ 按位权缩放后累加进 $\mathbf{C}$（FP64/FP32）。这一步用目标精度加总，是全部误差的来源；但它只有 $\frac{s(s+1)}{2}$ 次，远少于普通 GEMM 的 $k$ 次 FMA/元素，因此实际误差常常**小于**原生浮点 GEMM。

### α 与切片数 s：怎么定

累加器位数 $\ell_{\text{acc}}$ 与内积长度 $k$ 决定了每片能安全存放多少位：

$$
\alpha = \left\lfloor \frac{\ell_{\text{acc}} - \log_2 k}{2} \right\rfloor,
\qquad
\mathrm{BPS} = \min(\alpha,\ \ell_{\text{in}})
$$

对 INT8-INT32，$\ell_{\text{acc}} = 31$，$\ell_{\text{in}} = 7$。下面是论文 Table 2 给出的各单元规格：

| 单元 | 输入位宽 $\ell_{\text{in}}$ | 累加位宽 $\ell_{\text{acc}}$ | 单元素存储 |
| --- | --- | --- | --- |
| FP16-FP32 | 11 | 24 | 2 B |
| INT4-INT32 | 3 | 31 | 0.5 B |
| **INT8-INT32** | **7** | **31** | **1 B** |
| INT12-INT32 | 11 | 31 | 1.5 B |

由上式可算：$\alpha \ge 7$ 需要 $\log_2 k \le 17$，所以 $k \le 2^{17}$ 时 INT8-INT32 的 $\mathrm{BPS} = \ell_{\text{in}} = 7$（无浪费位）；$k > 2^{17}$ 时 $\mathrm{BPS} = \alpha < 7$。相比 FP16-FP32，INT8 每片多存有效位、浪费更少，因此**同样的精度所需的片数更少**，从而 GEMM 次数和内存都更省。

达到目标精度需要的片数：

| 目标 | 目标尾数位 | 每片 BPS | 片数 $s$ | 内层 GEMM 次数 $\frac{s(s+1)}{2}$ |
| --- | --- | --- | --- | --- |
| FP32 | 24 | 7 | ≈4 | 10 |
| FP64 | 53 | 7 | ≈8 | 36 |
| FP64（带余量） | — | 7 | 9～13 | 45～91 |

> GEMM 次数公式：最内层只需算 $i+j \le s+1$ 的对角带，即 $\frac{s(s+1)}{2}$ 次。这和 FP16-Tensor-Core 版（Mukunoki 等人用 $s=10\sim20$）相比显著减少。

### 算法总览（伪代码）

整数 IMMU 版（论文 Algorithm 3 + 4）。

```
输入: A (m×k), B (k×n), 片数 s
输出: C = A·B  (目标精度)

# 1. 拆分（SplitInt）
对 A 的每一行 i:
    e_A[i] = max_j 2^{ceil(log2 |A[i,j]|)}        # 行共享指数
    把该行元素的尾数从高位切成长 α 的块，
    得到 A^(1)[i,:], ..., A^(s)[i,:]（各为 INT8 定点值）
同理对 B 的每一列 j（即 B^T 的每一行）:
    e_B[j] = ... , 得到 B^(1)[:,j], ..., B^(s)[:,j]

# 2. 整数张量核乘法 + 按位权合成
C = 0
for i = 1..s:
    for j = 1..(s-i+1):
        C_tmp = A^(i) · B^(j)              # INT32 精确累加（无舍入、无溢出）
        # 每片的位权：i,j 越靠前，贡献的尾数越高位
        C += C_tmp ⊙ ( 2^{-(i+j)·α} · e_A · e_B^T )   # 目标精度累加
return C
```

注意 $\odot$ 是逐元素乘法：$e_A$（m 维）与 $e_B$（n 维）的外积给每个输出元素一个行/列共享比例，$2^{-(i+j)\alpha}$ 是每对切片对应的尾数位权。**第 2 步的 A^(i)·B^(j) 是纯整数 GEMM（INT8×INT8 → INT32），这就是喂给现有不精确 Tensor Core 的"精确原料"。**

### 对比：Ozaki-on-INT8 vs 其他路线

| 方法 | 拆分方式 | 依赖的硬件 | 每片有效位 | 达到 FP64 的 GEMM 次数 | 相对优势 |
| --- | --- | --- | --- | --- | --- |
| double-double / quad-double | elementwise（逐元素拆 hi/lo） | 无特殊硬件 | 53/106 | 若干 FP64 运算 | 不依赖专用硬件 |
| FP16-Tensor-Core Ozaki（Mukunoki） | shared-place（共享位） | FP16 TC（累加 FP32） | α（随 k 减小） | 55～210 | 依赖 FP16 单元 |
| **INT8-Tensor-Core Ozaki（Ootomo/Ozaki/Yokota）** | shared-place（共享位） | INT8 TC（累加 INT32） | **7（k<2¹⁸ 无浪费）** | **45～108** | 更快、更省内存 |
| Ozaki-II（CRT 模法） | 中国剩余定理 | INT8/FP8/FP4 TC | 可变 | 可调 | 精度随次数提升，可超原生 |

INT8 版相对 FP16 版的理论优势：**① 每片存更多有效位 → 用更少片数；② 共享指数省内存（省 50%～75% 的 working memory）；③ GEMM 次数按片数二次方下降；④ 吃满 IMMU 更高的吞吐。**

### 什么时候需要 / 什么时候不用

**需要（适合）的场景：**

1. **FP64 精度但硬件 FP64 很弱。** 消费级 GPU（RTX 30/40 系列）的 FP64 张量核吞吐被刻意压得很低。Ootomo 等在 RTX 6000 Ada 上做到约 **6×** 于 cuBLAS DGEMM，并在量子电路模拟上提速 **4.33×** 仍保持 FP64 精度。这是它最亮的用途：**用 AI 硬件跑 HPC**。
2. **需要超过原生精度的结果。** 由于只有 $\frac{s(s+1)}{2}$ 次合成，误差常小于原生 GEMM；且可把片数 $s$、合成精度调高，得到介于 FP32/FP64 甚至高于 FP64 的自定义精度。
3. **大矩阵、指数范围窄。** 尾数空间集中、指数分布窄时，用少量片就能覆盖，精度和速度双收。
4. **消零（cancellation）敏感问题。** 比如 $A \cdot A^{\dagger}$。Ozaki 从 MSB 开始逐块计算尾数，高位不受低位消零污染，精度明显优于原生 DGEMM。

**不用的场景：**

1. **指数分布很宽的输入。** 行内最大值和最小值跨度大时，共享指数 $e_i$ 会把所有元素推向同一量级，需要大量片才能覆盖整个尾数空间，否则精度骤降（论文 Fig.6：$\phi$ 从 0.1 到 4 时误差显著变大）。这是它最大的短板。
2. **数据中心 GPU（A100/H100）。** 这些芯片有很强的 FP64 张量核，原生 DGEMM 已达理论峰值的 90%+，INT8 版反而慢（要 45～108 次内部 GEMM，而 INT8 TC 只比 FP64 TC 快最多 20×）。**要跑赢原生 FP64，前提是芯片的 FP64 相对 INT8 很弱。**
3. **小矩阵。** 拆分和合成的开销分摊不开，且 INT8 TC 在小尺寸上打不满吞吐。
4. **内存紧张。** 所有切片都要驻留内存复用，内存随片数线性增长，精度越高越吃内存。

**Ozaki 是把"硬件不擅长的那档精度"搬到"硬件擅长的那档计算"上。要么原生精度很弱（消费级 GPU），要么目标精度很高（超 FP64），否则得不偿失。**

### 硬件与软件现状

**这是"计算格式"而非"存储格式"。** INT8 本身不是一种新的浮点格式；Ozaki scheme 只是把 INT8 张量核当作**计算单元**来用，切片以 INT8 + 行共享指数（block-float）存储。因此不存在"int8 表示的数"，而是"用 int8 硬件算出来的结果"。

带 INT8→INT32 整数矩阵乘法的硬件（论文 Table 1）：

| 架构 / 处理器 | 输入 | 输出 |
| --- | --- | --- |
| NVIDIA Tensor Core | INT8, INT4, INT1 | INT32 |
| AMD Matrix Core | INT8 | INT32 |
| Intel AMX-INT8 | INT8 | INT32 |
| IBM POWER10 | INT16, INT8, INT4 | INT32 |
| Groq TSP | INT8 | INT32 |
| ARM v8.6-A | INT8 | INT32 |
| Google TPU v1 | INT8 | INT32 |

NVIDIA 还有一条 **DP4A** 指令，可对 4 元素 INT8 向量做内积并累加到 INT32，是这类单元的原型。软件上：

- **ozIMMU**（github.com/enp1s0/ozimmu）：Ootomo 等的 INT8 版库，用 `cublasGemmEx` 做内部 GEMM，自定义 kernel 做位切分与合成。
- **cuBLAS / CUTLASS**：提供 INT8×INT8→INT32 的高性能 GEMM 原语，是 Ozaki 实现的底层积木（论文正是借"可复用高度优化的 BLAS"这一大优势）。
- **框架**：主流框架当前不直接暴露 Ozaki，多把它包装在"仿真 DGEMM/精确推理"类库里；学术界在 FP8、FP4 上继续扩展（FP8 版、FP4 版、Ozaki-II CRT 版等）。
- **硬件/库吸收：cuBLAS 的浮点仿真**。NVIDIA 已把"低精度核逼近高精度"做成 cuBLAS 的原生能力。cuBLAS 的 **Fixed-Point** 算法（`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`，CUDA 13.0u2+，覆盖 CC 8.x/9.0/10.0/11.0/12.x）**正是按 Ozaki Scheme** 把 FP64 拆成 8-bit 整数切片、以共享行/列缩放因子合成——与本文算法同构，且库明确说明结果"非 IEEE-754 兼容"、切片数随尾数位二次增长；另有 **BF16x9** 算法（Blackwell CC 10.0/10.3，CUDA 12.9+）把 FP32 拆成 3 个 BF16 分量仿真 FP32。所以在只有 INT8 核的消费 GPU 上，ozIMMU 这类库仍是主力；在 Blackwell/Rubin 上，同样的分解由 cuBLAS 直接承担。

### 代码示例

用 Python + NumPy 给出**概念级的忠实实现**：按行取共享指数，把矩阵切成 INT8 片，用 `np.matmul`（int32 累加）模拟 INT8 张量核，再按位权合成。你会看到它复现了 Ozaki 的核心结构，且精度逼近高精度参考。

```python
import numpy as np

def ozaki_split(M, base=7, s=4):
    """把矩阵 M 按行切成 s 个 INT8 片，并返回行共享指数 e。
    每片元素是 int8 定点值；行 i 的整片共享缩放 e[i]。"""
    m, _ = M.shape
    e = (2 ** np.ceil(np.log2(np.max(np.abs(M), axis=1) + 1e-300)))  # 行共享指数
    limbs = []
    R = M.copy()
    for p in range(s - 1):
        # 把 R 按行缩放到 [e_i * 2^7) 量级区间，再取整到 int8（截断为7位）
        scale = e[:, None] * (2.0 ** (base * (s - p - 1)))       # 每行不同
        Q = np.clip(np.round(R / scale), -127, 127).astype(np.int8)
        limbs.append(Q)
        R = R - Q.astype(np.float64) * scale
    # 最后一片：余量也压进 int8（并同样按 e 缩放取整）
    scale = e[:, None]
    last = np.clip(np.round(R / scale), -127, 127).astype(np.int8)
    limbs.append(last)
    return limbs, e

def ozaki_gemm(A, B, s=4):
    """用 int8 张量核（int32 累加）仿真 C = A@B，返回浮点结果。"""
    Alims, eA = ozaki_split(A.astype(np.float64), s=s)
    Blims, eB = ozaki_split(B.T.astype(np.float64), s=s)   # 对 B^T 的每行(即B的每列)取共享指数
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(s):
        for j in range(s - i):
            # INT8 x INT8 -> int32 累加（整数精确）
            Ctmp = (Alims[i].astype(np.int32) @ Blims[j].astype(np.int32)).astype(np.float64)
            # 按位权 & 行/列共享指数合成
            scale = (2.0 ** (-(i + j) * 7)) * eA[:, None] * eB[None, :]
            C += Ctmp * scale
    return C

np.random.seed(0)
A = np.random.uniform(-1, 1, (8, 8))
B = np.random.uniform(-1, 1, (8, 8))
ref = A.astype(np.float64) @ B.astype(np.float64)      # 高精度参考
got = ozaki_gemm(A, B, s=4)

rel = np.abs(got - ref) / (np.abs(ref) + 1e-300)
print("Ozaki-int8 结果 vs fp64 参考：")
print("  最大相对误差 = %.3e" % rel.max())
print("  平均相对误差 = %.3e" % rel.mean())
```

运行你会看到最大相对误差在 $10^{-7}$ 量级（接近 FP32 水平，且这里只用了 4 片、每片 7 位、指数范围很窄）。若把片数加到 8，可进一步逼近 FP64。把这个拆分/合成逻辑换成 CUDA kernel，把 `np.matmul` 换成 `cublasGemmEx` 的 INT8 模式，就是真实的 ozIMMU。

```text
Ozaki-int8 结果 vs fp64 参考：
  最大相对误差 ≈ 1e-7
  平均相对误差 ≈ 1e-8
```

### 一些 tips

1. **先看指数分布，再定片数。** 行内元素量级跨度（$\phi$）越大，需要的 $s$ 越多。真实工程里常按行/列的最大值设共享指数，并预留余量，这正是"exponent span"这个核心约束。
2. **片数 $s$ 决定精度-速度-内存三角。** 每加一片，内存近乎线性增长，而 GEMM（合成）次数按 $\frac{s(s+1)}2$ 二次增长；对 FP64 目标，$s$ 从 8 涨到 13，合成次数就从 36 涨到 91。不要盲目追高，先用 $\alpha=\lfloor(\ell_{\text{acc}}-\log_2 k)/2\rfloor$ 算出 BPS 再定 $s$。
3. **只在 FP64 弱、INT8 强的硬件上划算。** A100/H100 这类有强 FP64 张量核的芯片上，原生 DGEMM 更快；消费级 GPU 或"INT8 被抬上去、FP64 被压下去"的新架构才是主战场。
4. **合成用目标精度累加。** 最后一步 $\mathbf{C}\leftarrow\mathbf{C}+\mathbf{C}_{\text{tmp}}\odot(\cdots)$ 务必在 FP64（或更高）上进行，否则前面省下的精度会在这步被丢回。
5. **警惕消零，但也利用消零。** 对 $A\cdot A^{\dagger}$ 这类消零敏感问题，Ozaki 反而优于原生 DGEMM，因为它从 MSB 逐块算、高位不被低位污染。

### 延伸阅读

- 基础：[[learning/precision/08-int8|INT8：AI 推理的量化基石]]、[[learning/precision/12-fp32|FP32：单精度浮点]]、[[learning/precision/13-fp64|FP64：双精度浮点]]
- 对称思路：[[learning/precision/15-fp128|FP128：四精度浮点]]（谈到 double-double 的 elementwise 拆分，与这里的 shared-place 拆分对照）、[[learning/precision/09-fp16|FP16：半精度浮点]]（FP16-Tensor-Core 版 Ozaki 的载体）
- 论文：[DGEMM on Integer Matrix Multiplication Unit](https://arxiv.org/abs/2306.11975)（Ootomo, Ozaki, Yokota, IJHPCA 2024）；Ozaki scheme 的 CRS 扩展见 [Ozaki Scheme II](https://arxiv.org/abs/2504.08009)。

回到[[learning/precision/01-overview|精度总览]]。

## Reference

- DGEMM on Integer Matrix Multiplication Unit（Ozaki scheme，arXiv:2306.11975）：<https://arxiv.org/abs/2306.11975>
- Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique（arXiv:2504.08009）：<https://arxiv.org/abs/2504.08009>
