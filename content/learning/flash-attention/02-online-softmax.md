这一篇讲 FlashAttention 的算法本体。[[learning/flash-attention/01-flash-attention|系列总览]] 里说过，三件套是分块、在线 softmax、重计算。这三样里只有在线 softmax 是真正的数学，另外两个是工程。所以先把在线 softmax 的推导讲透。

> **FA1 的不变量很简单：softmax 每一行可以一边扫 block 一边算，只要维护两个统计量（最大值 `$ m $`、指数和 `$ \ell $`）。分块让矩阵乘能喂给 tensor core，在线 softmax 让 softmax 不必等整行算完，两者配合起来才能在一个 kernel 里跑完整段 attention。**

## 前驱：把除法挪到最后

先看 Rabe 和 Staats 那篇前驱（2021，`Self-attention Does Not Need $O(n^2)$ Memory`）。它的核心观察是个分配律：

$$
\mathrm{attention}(q, k, v) = \frac{\sum_i v_i e^{\langle q, k_i \rangle}}{\sum_j e^{\langle q, k_j \rangle}}
$$

也就是**分母的除法可以挪到所有 `$ i $` 累加完之后再做**。这样单条 query 只需要维护两个累加器：`$ v^* = \sum_i v_i e^{s_i} $`（一个 `$ d $` 维向量）和 `$ s^* = \sum_j e^{s_j} $`（一个标量）。扫完所有 key/value，除一下就是结果。

**代价是数值问题。** `$ e^{s_i} $` 对 `$ s_i \ge 89 $`（fp32 / bf16）直接溢出成 `$ \mathrm{inf} $`。标准实现靠"减去最大值"规避，但那要求先知道全部 `$ s_i $`。累积求和时最大值可能是最后一个 key 才出现的，等不了。所以必须引入第三个累加器：**当前行最大值 `$ m $`**，每见到新的分数就重缩一次。

这就是在线 softmax 的由来：`$ v^* $`、`$ s^* $` 都不是直接累加，而是按 `$ m $` 的变化"重缩放"后累加。

## 在线 softmax 的数学

对一个向量 `$ x \in \mathbb{R}^{B} $`，标准 softmax 分三层：

$$
m(x) = \max_i x_i, \qquad
f(x) = \begin{bmatrix} e^{x_1 - m(x)} & \cdots & e^{x_B - m(x)} \end{bmatrix}, \qquad
\ell(x) = \sum_i f(x)_i, \qquad
\mathrm{softmax}(x) = \frac{f(x)}{\ell(x)}
$$

关键是把两个子向量 `$ x^{(1)}, x^{(2)} $` 拼起来时，这三层怎么合并：

$$
m(x) = \max\big(m(x^{(1)}), m(x^{(2)})\big)
$$

$$
f(x) = \begin{bmatrix} e^{m(x^{(1)}) - m(x)} f(x^{(1)}) & e^{m(x^{(2)}) - m(x)} f(x^{(2)}) \end{bmatrix}
$$

$$
\ell(x) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)})
$$

合并只用 `$ m $` 和 `$ \ell $`，这就是"代数聚合"。

![在线 softmax：每个新块的分数重缩放到运行最大值，再合并进累计状态](/learning/assets/online-softmax.svg)

**用一段 NumPy 验证**（博客代码块可以直接跑）。单条 query、9 个 key、每块 3 个，在线版把 `$ O $` 和 `$ \ell $` 一路累积，最后除以 `$ \ell $`，得到和 full softmax 一致的结果：

```python
import numpy as np
rng = np.random.default_rng(0)

def attn_ref(Q, K, V):          # 参考：一次算全
    S = Q @ K.T
    M = S.max(axis=-1, keepdims=True)
    P = np.exp(S - M)
    P /= P.sum(axis=-1, keepdims=True)
    return P @ V

Q = rng.normal(size=(1, 4)); K = rng.normal(size=(9, 4)); V = rng.normal(size=(9, 4))
block = 3
O_tilde = np.zeros((1, 4)); m = -np.inf; ell = 0.0
for j in range(0, 9, block):
    Kj, Vj = K[j:j+block], V[j:j+block]
    S = Q @ Kj.T
    m_new = max(m, S.max())               # 新块行最大
    P = np.exp(S - m_new)                 # 用新 max 算指数
    ell = ell * np.exp(m - m_new) + P.sum()
    O_tilde = O_tilde * np.exp(m - m_new) + P @ Vj
    m = m_new
O = O_tilde / ell
print("online O 与参考一致:", np.allclose(O, attn_ref(Q, K, V)))
```

关键在 `O_tilde * np.exp(m - m_new)`：旧块的结果按新旧 max 的差重缩放，再和当前块的结果相加；`ell` 同样重缩放。最后除一次 `ell`，就和全量算出的 softmax 对齐。

## FA1 的 forward：分块 + 在线 softmax + 重计算

现在把在线 softmax 放到整个 attention 上，并让它和矩阵乘对齐。给定 `$ Q, K, V $`，SRAM 大小 `$ M $`，FA1 设置块大小：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad
B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

- `$ B_c $`：每次放进 SRAM 的 key/value 列块大小。
- `$ B_r $`：每块处理的 query 行数，一般不大于 `$ d $`。

为什么是 `4d`：SRAM 里要同时放 `$ Q_i $`、`$ K_j $`、`$ V_j $` 和一块输出 `$ O $`，四块 `$ B \times d $`，加起来约 `$ 4B d $` 个元素，得塞进 `$ M $`。

FA1 的 forward 伪代码（外层扫列块 `$ j $`，内层扫行块 `$ i $`）：

```text
1. 初始化 O=0, l=0, m=-inf（在 HBM）
2. 把 Q 切成 T_r 个行块，把 K、V 切成 T_c 个列块
3. for j in 1..T_c:                       # 外层：key/value 列块
4.     把 K_j, V_j 加载进 SRAM
5.     for i in 1..T_r:                   # 内层：query 行块
6.         把 Q_i, O_i, l_i, m_i 加载进 SRAM
7.         S_ij = Q_i K_j^T               # 在片上算
8.         m̃_ij = rowmax(S_ij)
9.         P̃_ij = exp(S_ij - m̃_ij)
10.        l̃_ij = rowsum(P̃_ij)
11.        m_i_new = max(m_i, m̃_ij)
12.        l_i_new = e^{m_i - m_i_new} l_i + e^{m̃_ij - m_i_new} l̃_ij
13.        O_i ← diag(l_i_new)^{-1} ( diag(l_i) e^{m_i - m_i_new} O_i
                                        + e^{m̃_ij - m_i_new} P̃_ij V_j )
14.        把 O_i, l_i, m_i 写回 HBM
15. 返回 O
```

第 13 行是重点。它把两个块的结果"对齐到新的 `$ m $` 再合并"。具体看：

- 旧的输出 `$ O_i $` 存的是**按旧 `$ m_i $` 归一化**的结果（每步都除以了当时的 `$ l_i $`），所以搬它的权重是 `$ e^{m_i - m_i^{new}} $`，还要乘回旧 `$ l_i $` 才和不归一化版本对齐。
- 新块 `$ \widetilde{P}_{ij} V_j $` 的权重是 `$ e^{\widetilde{m}_{ij} - m_i^{new}} $`，它把新块的指数也从新块自身 max 对齐到了全局 max。
- 最后一起除以 `$ l_i^{new} $` 得到归一化的 `$ O_i $`。

## 为什么这样能让 HBM 少一个数量级

FA1 证明了 IO 复杂度。设 SRAM 大小 `$ M $` 满足 `$ d \le M \le Nd $`：

| 实现 | HBM 访问 |
| --- | --- |
| 标准 attention | `$ \Theta(Nd + N^2) $` |
| FlashAttention | `$ \Theta(N^2 d^2 M^{-1}) $` |

标准实现要把 `$ N \times N $` 的 `$ S $`、`$ P $` 写 HBM，光是这两个就是 `$ \Theta(N^2) $`。FA1 的 `$ d^2 / M $` 一项在 `$ d \in [64, 128] $`、`$ M \approx 100 \text{KB} $` 时远小于 1，所以 HBM 访问从平方级掉到次二次。

而且对精确 attention 而言，这个复杂度是**下界**，不只是"比标准快"。FA1 的 Proposition 3 证明：对 `$ M \in [d, Nd] $` 的任意 SRAM 大小，都不存在一个精确 attention 算法能把 HBM 访问做到 `$ o(N^2 d^2 M^{-1}) $`。换句话说，在精确计算的前提下，这份 IO 复杂度是**渐近最优**的，能做的最多就是贡献一个常数因子。

论文里 GPT-2 medium（seq 1024、head dim 64、16 heads、batch 64）的实测直接印证：

| | GFLOPs | HBM R/W（GB） | Runtime（ms） |
| --- | ---: | ---: | ---: |
| 标准 attention | 66.6 | 40.3 | 41.7 |
| FlashAttention | 75.2 | **4.4** | **7.3** |

注意 FA1 的 GFLOPs 反而更高（75.2 > 66.6），因为反向要重算。**但 HBM 读写从 40.3 GB 掉到 4.4 GB，时间从 41.7 ms 掉到 7.3 ms。** 这就是"memory-bound 操作，瓶颈在带宽不在 FLOPs"最直接的证据。

## 重计算：反向不用存 S、P

标准实现反向需要 `$ S, P $` 来算梯度，于是 forward 时把它们写进 HBM，或者用梯度检查点换 `$ S $`。FA1 的做法：**forward 只存 `$ O $` 和统计量 `$ (m, \ell) $`，反向时按块重新算出 `$ S = Q K^\top $`、`$ P = \exp(S - m) $`。**

这多出来的 FLOPs 并不亏。反向 pass 的 GEMM 反而因为 HBM 访问更少而更快。完整推导在 [[learning/flash-attention/04-backward-kernel|反向内核逐行读]]。这里记住一件事：重计算不是"省存 S 的显存"，而是把 `$ O(N^2) $` 的显存需求压成 `$ O(N) $`（只存 `$ O $` 和 `$ L $`），同时反向还更快。

## 正确性、复杂度、显存一句话

一个定理概括：

> FlashAttention 返回 `$ O = \mathrm{softmax}(QK^\top) V $`，FLOPs 为 `$ O(N^2 d) $`，额外显存为 `$ O(N) $`（只多存 `$ O $` 和 `$ L $`）。**精确、不近似。**

后面三篇往里加的东西（FA2 的并行、FA3 的 TMA/FP8）都是这个骨架上的加速，算法本身没变。

## Reference

- FlashAttention（arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- Self-attention Does Not Need $O(n^2)$ Memory（arXiv:2112.05682）：<https://arxiv.org/abs/2112.05682>
- Online normalizer calculation for softmax（Milakov & Gimelshein）：<https://arxiv.org/abs/1805.02867>
- Reformer: The Efficient Transformer（分块 softmax 前驱）：<https://arxiv.org/abs/2001.04451>
