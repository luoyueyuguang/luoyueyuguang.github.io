Roofline 模型回答一个问题：**在给定的硬件上，一个 kernel 的理论性能上限是多少，以及它到底被什么卡住。** 它把"算得快不快"拆成两个硬天花板——峰值算力和峰值带宽——然后用一条折线把可达到的性能框出来。名字取自这条折线长得像房子的屋顶。

> **一句话：可达到的性能 = min(峰值算力, 算术强度 × 峰值带宽)。** 关键不在算力本身，而在"算力 / 带宽"这个比值（ridge point）把你分成哪一区。

## 两个天花板

一块 GPU 有两条独立的资源线：

- **峰值算力 $\pi$**（compute ceiling，FLOPs/s）：单位时间内能算多少次乘加。
- **峰值带宽 $\beta$**（memory ceiling，bytes/s）：单位时间内能搬多少字节。

它们由不同部件决定，且互相独立：算力由 tensor core / ALU 数量与频率决定，带宽由 HBM 代际与位宽决定。所以一个 kernel 的理论性能，不可能超过两者中更紧的那一个。

## 算术强度：把两者联系起来

关键在于**算术强度**（arithmetic intensity，简称 AI），定义为一个 kernel **每搬 1 字节能做多少次浮点运算**：

$$
I = \frac{\text{FLOPs}}{\text{bytes}} \quad [\text{FLOPs / byte}]
$$

它是衡量"算得多还是搬得多"的唯一标尺。注意这里的 bytes 指**从 DRAM 搬**的字节，不是片上 SRAM/寄存器之间的搬移——同一个 kernel，分块做得好、复用高，DRAM 流量就低，有效 $I$ 就高。

## 折线与三区

用两个天花板写出**可达到性能上限**：

$$
\text{achievable} = \min\big(\pi,\; \beta \cdot I\big)
$$

这条线由两段组成，交点叫 **ridge point**：

$$
I_{\text{ridge}} = \frac{\pi}{\beta}
$$

| 区域 | 条件 | 性能由谁决定 | 优化方向 |
| --- | --- | --- | --- |
| **内存密集**（memory-bound） | $I < I_{\text{ridge}}$ | $\beta \cdot I$（被带宽卡住） | **减字节**：提高复用、降低精度、分块 |
| **计算密集**（compute-bound） | $I > I_{\text{ridge}}$ | $\pi$（被算力卡住） | **减 FLOPs** 或提高算力利用率 |
| **临界** | $I \approx I_{\text{ridge}}$ | 两者都紧 | 两头都抓 |

注意：**内存密集 ≠ "这个硬件带宽小"**，而是一个 kernel 的算术强度太低，喂不饱算力。

## 一个 A100 的例子

A100（80GB SXM）的官方规格是 $\beta = 2039\ \mathrm{GB/s} \approx 2.0\ \mathrm{TB/s}$（实测约 1.94–2.04），$\pi$：FP32 19.5 TFLOPS、BF16 tensor core 312 TFLOPS（稠密；624 那个数字是 2:4 稀疏）。代入得两个 ridge point：

$$
I_{\text{ridge}}^{\text{FP32}} = \frac{19.5\times10^{12}}{2.0\times10^{12}} \approx \mathbf{9.75},\qquad
I_{\text{ridge}}^{\text{BF16}} = \frac{312\times10^{12}}{2.0\times10^{12}} \approx \mathbf{156}
$$

用 NumPy 画一下这条折线，看每个 kernel 落在哪一区：

```python
import numpy as np

# A100 80GB SXM 规格
pi_fp32 = 19.5e12      # FP32 峰值（FLOPs/s）
pi_bf16 = 312e12       # BF16 tensor core 稠密峰值（624 是 2:4 稀疏）
beta    = 2.0e12       # HBM 带宽（官方 2039 GB/s，取 2 TB/s）

for name, pi in (("FP32", pi_fp32), ("BF16", pi_bf16)):
    ridge = pi / beta
    I = np.logspace(-1, 3, 300)          # 算术强度 0.1 .. 1000
    peak = np.minimum(pi, beta * I)      # 可达到性能（折线）
    print(f"{name}: ridge point = {ridge:8.1f} FLOPs/byte")

def classify(ai, ridge):
    """按算术强度判断一个 kernel 落在哪一区。"""
    return "memory-bound（被带宽卡）" if ai < ridge else "compute-bound（被算力卡）"

print("\n常见 kernel 的算术强度（粗略，对照 BF16 张量核 ridge=156）：")
print(f"  elementwise（bf16 读写各 2 字节、每元素约 2 FLOP）I≈0.5  → {classify(0.5, 156)}")
print(f"  注意力（FA1 实测 fwd+bwd：75.2 GFLOPs / 4.4 GB）  I≈17   → {classify(17, 156)}")
print(f"  大 N 的 GEMM（N=4096、bf16 操作数，I≈N/3≈1365）  I≈1365 → {classify(1365, 156)}")
```

实际运行输出：

```text
FP32: ridge point =      9.8 FLOPs/byte
BF16: ridge point =    156.0 FLOPs/byte

常见 kernel 的算术强度（粗略，对照 BF16 张量核 ridge=156）：
  elementwise（bf16 读写各 2 字节、每元素约 2 FLOP）I≈0.5  → memory-bound（被带宽卡）
  注意力（FA1 实测 fwd+bwd：75.2 GFLOPs / 4.4 GB）  I≈17   → memory-bound（被带宽卡）
  大 N 的 GEMM（N=4096、bf16 操作数，I≈N/3≈1365）  I≈1365 → compute-bound（被算力卡）
```

三个 example 的 $I$ 怎么来的：

- **elementwise**：一个元素读 2 字节、写 2 字节，做约 2 次浮点运算（如 $y = \mathrm{relu}(x)$ 的比较加乘），$I = 2/4 = 0.5$。
- **注意力**：$I \approx 17$ 不是估的——FA1 论文 Figure 2 实测 GPT-2 medium（$N=1024$、$d=64$、16 head、batch 64）forward+backward 是 75.2 GFLOPs 对 4.4 GB HBM 读写，相除得 17.1。理论值高得多（见下节），差在实现效率上。
- **大 N 的 GEMM**：$N \times N$ 的 $C = AB$，FLOPs $= 2N^3$，访存是 $A, B, C$ 各 $N^2$ 个元素、bf16 每个 2 字节共 $6N^2$，于是

$$
I_{\text{GEMM}} = \frac{2N^3}{6N^2} = \frac{N}{3}
$$

N=4096 时约 1365。这个式子对字节数敏感：同样 $N$，操作数换成 4 字节（FP32/TF32）就掉到 $N/6 \approx 683$。

**注意 ridge 跟着"用哪条计算路径"变**：attention 用的是 tensor core，所以必须对照 BF16 的 ridge（156），而不是 FP32 的（9.75）。同一份 $I$ 在不同 ridge 下可能属于不同区——这就是为什么先得确认"拿什么算"，再查表归类。

## 怎么用

Roofline 的用法不是"看个热闹"，而是**先定位瓶颈，再改对地方**：

1. **先算 $I$，再对照 ridge。** $I < I_{\text{ridge}}$ → 改字节；$I > I_{\text{ridge}}$ → 改 FLOPs 或利用率。改错方向等于白干。
2. **memory-bound 的直觉**：$\beta I$ 这条斜线上，性能被带宽死死压住。想提速只有两条路——提高复用降 bytes（分块、把数据留在片上），或降精度减字节（见 [[learning/precision/01-overview|精度系列]]）。
3. **compute-bound 的直觉**：性能贴着 $\pi$ 这条平线，已经跑满算力，再降字节也没用；要么减少需要算的量，要么把利用率从 30% 提到 70%。
4. **一个 kernel 可以跨区移动**。同一个 GEMM，块小、复用差时落在内存密集区；做 blocking 把 DRAM 流量降下来，$I$ 上升，就可能滑到计算密集区。**这就是分块优化为什么有效的数学借口。**

## 和 attention 的关系

attention 是典型的内存密集操作：softmax 是 reduction，大量 HBM 读写、算术很少（[[learning/flash-attention/01-flash-attention|FlashAttention 系列]] 开篇就在讲这个）。朴素实现会把 $N \times N$ 的 $S$、$P$ 写进 HBM 再读出来，每行 query 花 $4Nd$ FLOPs 却要搬约 $8N$ 字节，$I \approx d/2$（$d=64$ 时 32）——远低于 ridge。FlashAttention 把 $S$、$P$ 留在片上，但**流量也不是线性的**：每个 query 行块都要重读整条 $K, V$，总 HBM 读写是 $\Theta(N^2 d^2 M^{-1})$（$M$ 为片上 SRAM 元素数），比朴素实现的 $\Theta(Nd + N^2)$ 少了 $M/d^2$ 倍。按这个阶算理论 $I \approx 2M/d$（$M \approx 10^5$、$d=64$ 时三千多，已越过 ridge），但实测受实现效率所限只有十几（FA1 论文实测 $I \approx 17$），**仍落在 ridge 左边**。所以准确的说法是"把 $I$ 大幅抬向 ridge"，而不是"一步跨过 ridge"。而 FA4 在 Blackwell 上先做多资源 roofline 判断瓶颈**是否已经换人**——B200 的 BF16 tensor core 吞吐是 H100 的两倍（2.25 PFLOPS vs 1 PFLOPS），但共享内存读带宽仍是 128 B/clock/SM、指数单元仍是 16 op/clock/SM（和 Hopper 相同；B300 才把 exp 翻倍到 32），于是瓶颈从 MMA 转到 smem 流量和 exp——再决定改 kernel 还是改算法。

## Reference

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM 52(4), April 2009, pp. 65–76：<https://dl.acm.org/doi/10.1145/1498765.1498785>
- NVIDIA A100 产品规格页（80GB SXM：HBM2e 2039 GB/s、FP32 19.5 TFLOPS、BF16 tensor core 312 TFLOPS 稠密 / 624 稀疏）：<https://www.nvidia.com/en-us/data-center/a100/>
- NVIDIA GPU Performance Background User's Guide（arithmetic intensity / ops:byte、memory- vs math-limited 判定）：<https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>
