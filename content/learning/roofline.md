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

A100（80GB SXM）大致是 $\beta \approx 2.0\ \mathrm{TB/s}$（实测约 1.94–2.04），$\pi$：FP32 约 19.5 TFLOPS，BF16 tensor core 约 312 TFLOPS（稠密）。代入得两个 ridge point：

$$
I_{\text{ridge}}^{\text{FP32}} = \frac{19.5\times10^{12}}{2.0\times10^{12}} \approx \mathbf{9.75},\qquad
I_{\text{ridge}}^{\text{BF16}} = \frac{312\times10^{12}}{2.0\times10^{12}} \approx \mathbf{156}
$$

用 NumPy 画一下这条折线，看每个 kernel 落在哪一区：

```python
import numpy as np

# A100 大致参数
pi_fp32 = 19.5e12      # FLOPs/s
pi_bf16 = 312e12       # FLOPs/s
beta    = 2.0e12       # bytes/s（约 2 TB/s HBM）

for name, pi in (("FP32", pi_fp32), ("BF16", pi_bf16)):
    ridge = pi / beta
    I = np.logspace(-1, 3, 300)          # 算术强度 0.1 .. 1000
    peak = np.minimum(pi, beta * I)      # 可达到性能（折线）
    print(f"{name}: ridge point = {ridge:8.1f} FLOPs/byte")

def classify(ai, peak):
    """按算术强度判断一个 kernel 落在哪一区。"""
    if ai < peak:
        return "memory-bound（被带宽卡）"
    return "compute-bound（被算力卡）"

print("\n常见 kernel 的算术强度（粗略）：")
print(f"  elementwise（逐元素，如激活）        I≈0.5  → {classify(0.5, 9.75)}")
print(f"  大 N 的 GEMM（N=4096，AI≈2N/3≈2730） → {classify(2730, 9.75)}")
print(f"  注意力（分块把 HBM 压到 O(N) 后）    I≈几十 → {classify(60, 9.75)}")
```

实际运行会打印：`FP32: ridge point = 9.8`、`BF16: ridge point = 156.0`，并把三个 example 归好区。

## 怎么用

Roofline 的用法不是"看个热闹"，而是**先定位瓶颈，再改对地方**：

1. **先算 $I$，再对照 ridge。** $I < I_{\text{ridge}}$ → 改字节；$I > I_{\text{ridge}}$ → 改 FLOPs 或利用率。改错方向等于白干。
2. **memory-bound 的直觉**：$\beta I$ 这条斜线上，性能被带宽死死压住。想提速只有两条路——提高复用降 bytes（分块、把数据留在片上），或降精度减字节（见 [[learning/precision/01-overview|精度系列]]）。
3. **compute-bound 的直觉**：性能贴着 $\pi$ 这条平线，已经跑满算力，再降字节也没用；要么减少需要算的量，要么把利用率从 30% 提到 70%。
4. **一个 kernel 可以跨区移动**。同一个 GEMM，块小、复用差时落在内存密集区；做 blocking 把 DRAM 流量降下来，$I$ 上升，就可能滑到计算密集区。**这就是分块优化为什么有效的数学借口。**

## 和 attention 的关系

attention 是典型的内存密集操作：softmax 是 reduction，大量 HBM 读写、算术很少（[[learning/flash-attention/01-flash-attention|FlashAttention 系列]] 开篇就在讲这个）。FlashAttention 靠分块 + online softmax 把 HBM 读写从 $O(N^2)$ 压到 $O(N)$，本质就是**把 $I$ 一路抬到 ridge point 右边，把"带宽瓶颈"变成"算力瓶颈"**。而 FA4 在 Blackwell 上甚至先画 roofline 判断瓶颈是否已经换人（B200 的 tensor core 翻倍但共享内存带宽和指数单元没涨），再决定改 kernel 还是改算法。

## Reference

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM 52(4), 2009：<https://dl.acm.org/doi/10.1145/1498765.1498785>
- NVIDIA A100 规格表（HBM 带宽、FP32 / TF32 / BF16 张量核峰值）：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf>
- NVIDIA Performance Optimization 文档（memory-bound vs compute-bound 调优）：<https://docs.nvidia.com/deeplearning/performance/dl-performance-guidelines/>
