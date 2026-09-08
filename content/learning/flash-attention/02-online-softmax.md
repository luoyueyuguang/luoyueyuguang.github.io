这一篇讲 FlashAttention 的算法本体。[[learning/flash-attention/01-flash-attention|系列总览]] 里说过，fa1是由分块、online softmax、重计算组成。

> **FA1 的不变量很简单：softmax 每一行可以一边扫 block 一边算，只要维护两个统计量（最大值 $ m $、指数和 $ \ell $）。分块让矩阵乘能喂给 tensor core，online softmax 让 softmax 不必等整行算完，两者配合起来才能在一个 kernel 里跑完整段 attention。**

## 把除法挪到最后

先看 Rabe 和 Staats 的论文（2021，Self-attention Does Not Need $O(n^2)$ Memory）。它的核心观察是个分配律：

$$
\mathrm{attention}(q, k, v) = \frac{\sum_i v_i e^{\langle q, k_i \rangle}}{\sum_j e^{\langle q, k_j \rangle}}
$$

这个式子想说的是：**softmax 的归一化分母 $ \sum_j e^{s_j} $ 对每个 $ i $ 都是同一个标量**，所以能当一个公共常数、从"对 $ i $ 求和"里提出来——这就是所谓的"分配律"。拆开看——记一条 query 的分数 $ s_i = \langle q, k_i\rangle $，标准 attention 是"softmax 权重再对 $ v_i $ 加权"：

$$
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \qquad
\mathrm{attention} = \sum_i p_i\, v_i
$$

把分母（一个不依赖 $ i $ 的标量）提出来：

$$
\mathrm{attention} = \sum_i v_i\, \frac{e^{s_i}}{s^\star} = \frac{\sum_i v_i\, e^{s_i}}{s^\star} = \frac{v^\star}{s^\star},
\qquad v^\star = \sum_i v_i e^{s_i},\quad s^\star = \sum_j e^{s_j}
$$

关键在于：分子 $ v^\star $（一个 $ d $ 维向量）和分母 $ s^\star $（一个标量）是**两个互不依赖的累加**。常规做法得先攒齐一整行 $ p_i $——而那必须先知道分母、也就是先扫完所有 key；这里不必——一边扫 key 一边各自累加，最后除一次即可。

**省在内存**：常规 softmax 要把每条 query 的 $ e^{s_i} $（或 $ p_i $）都存下来，那是每行 $ O(n) $、整张 $ O(n^2) $；这里只存 $ v^\star $（$ d $ 维）和 $ s^\star $（标量），每行 $ O(d) $，而且可以逐 key 流式累加。这正是 Rabe 和 Staats 那篇标题里的观察。

**代价是数值问题。** $ e^{s_i} $ 对 $ s_i \ge 89 $（fp32 / bf16）直接溢出成 $ \mathrm{inf} $。标准实现靠"减去最大值"规避，但那要求先知道全部 $ s_i $。累积求和时最大值可能是最后一个 key 才出现的，等不了。所以必须引入第三个累加器：**当前行最大值 $ m $**，每见到新的分数就重缩一次。

这就是online softmax 的由来：$ v^\star $、$ s^\star $ 都不是直接累加，而是按 $ m $ 的变化"重缩放"后累加。

## online softmax 的数学

对一个向量 $ x \in \mathbb{R}^{B} $，标准 softmax 分三层：

$$
m(x) = \max_i x_i, \qquad
f(x) = \begin{bmatrix} e^{x_1 - m(x)} & \cdots & e^{x_B - m(x)} \end{bmatrix}, \qquad
\ell(x) = \sum_i f(x)_i, \qquad
\mathrm{softmax}(x) = \frac{f(x)}{\ell(x)}
$$

关键是把两个子向量 $ x^{(1)}, x^{(2)} $ 拼起来时，这三层怎么合并：

$$
m(x) = \max\big(m(x^{(1)}), m(x^{(2)})\big)
$$

$$
f(x) = \begin{bmatrix} e^{m(x^{(1)}) - m(x)} f(x^{(1)}) & e^{m(x^{(2)}) - m(x)} f(x^{(2)}) \end{bmatrix}
$$

$$
\ell(x) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)})
$$

所以在线 softmax 只需要带两个标量——当前行最大值 $ m $ 和指数和 $ \ell $，不用存完整的 $ f $ 向量（attention 里还要额外带一个输出累加器 $ O $，合并时把它按 $ e^{m_{\text{old}}-m_{\text{new}}} $ 重缩后再加上新块的贡献）。关键在于这里的合并**满足结合律**（构成一个 monoid / 幺半群）：哪两块先合并都不影响最终的 $ m,\ell $。正因为结合律，整行的 softmax 才能拆成任意大小的块、按任意顺序算，结果都和整行一起算完全一致。

> **monoid / 幺半群：一个"可以随便合并"的代数结构。** 它由三样东西组成——集合 $ S $、二元运算 $ \circ $、单位元 $ e $，满足三条：
> 1. **闭合**：$ a \circ b \in S $（合并结果仍在集合里）；
> 2. **结合律**：$ (a\circ b)\circ c = a\circ(b\circ c) $（先合哪两个，最终一样）；
> 3. **单位元**：$ e\circ a = a\circ e = a $（有个"什么都没做"的状态）。
>
> 对 online softmax：$ S $ = 所有状态 $ (m,\ell) $（attention 还要带 $ O $，即 $ (m,\ell,O) $），$ \circ $ = 上面的合并（取 max、重缩、加和），单位元 = $ (m=-\infty,\ \ell=0) $（什么都没扫的状态；合并它时 $ m=\max(-\infty,m_1)=m_1 $、$ \ell $ 不变）。真正起作用的是**结合律**——所以任意分块、任意顺序合并，最终 $ m,\ell $ 都一样。

![online softmax：每个新块的分数重缩放到运行最大值，再合并进累计状态](/learning/assets/online-softmax.svg)

**用一段 NumPy 验证**（blog代码块可以直接跑）。单条 query、9 个 key、每块 3 个，在线版把 $ O $ 和 $ \ell $ 一路累积，最后除以 $ \ell $，得到和 full softmax 一致的结果：

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
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
    print(f"block {j//block}: S = {S[0]}")
    ell = ell * np.exp(m - m_new) + P.sum()
    O_tilde = O_tilde * np.exp(m - m_new) + P @ Vj
    m = m_new
    print(f"   -> m={m:.4f} ell={ell:.4f} O_tilde={O_tilde[0]}")
O = O_tilde / ell
print("O (online) =", O[0])
print("O (full)   =", attn_ref(Q, K, V)[0])
print("完全一致:", np.allclose(O, attn_ref(Q, K, V)))
```

期望输出（`np.random.default_rng(0)` 下）：

```text
block 0: S = [ 0.8193 -0.3161 -1.1381]
   -> m=0.8193 ell=1.4625 O_tilde=[-0.9197  1.5988  1.8648  1.2405]
block 1: S = [ 0.3463 -0.5858 -0.4717]
   -> m=0.8193 ell=2.6059 O_tilde=[-0.937   1.8569  1.2498  1.188 ]
block 2: S = [-0.7552  0.0833  0.5937]
   -> m=0.8193 ell=4.0900 O_tilde=[ 0.2356  0.2548  3.0565  0.7352]
O (online) = [0.0576 0.0623 0.7473 0.1798]
O (full)   = [0.0576 0.0623 0.7473 0.1798]
完全一致: True
```

关键在 `O_tilde * np.exp(m - m_new)`：旧块的结果按新旧 max 的差重缩放，再和当前块的结果相加；`ell` 同样重缩放。最后除一次 `ell`，就和全量算出的 softmax 对齐。

## FA1 的 forward：分块 + online softmax + 重计算

现在把online softmax 放到整个 attention 上，并让它和矩阵乘对齐。给定 $ Q, K, V $，SRAM 大小 $ M $，FA1 设置块大小：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad
B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

- $ B_c $：每次放进 SRAM 的 key/value 列块大小。
- $ B_r $：每块处理的 query 行数，一般不大于 $ d $。

为什么是 `4d`：SRAM 里要同时放 $ Q_i $、$ K_j $、$ V_j $ 和一块输出 $ O $，四块 $ B \times d $，加起来约 $ 4B d $ 个元素，得塞进 $ M $。

FA1 的 forward 伪代码（外层扫列块 $ j $，内层扫行块 $ i $）：

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

第 13 行是重点。它把两个块的结果"对齐到新的 $ m $ 再合并"。具体看：

- 旧的输出 $ O_i $ 存的是**按旧 $ m_i $ 归一化**的结果（每步都除以了当时的 $ l_i $），所以搬它的权重是 $ e^{m_i - m_i^{new}} $，还要乘回旧 $ l_i $ 才和不归一化版本对齐。
- 新块 $ \widetilde{P}_{ij} V_j $ 的权重是 $ e^{\widetilde{m}_{ij} - m_i^{new}} $，它把新块的指数也从新块自身 max 对齐到了全局 max。
- 最后一起除以 $ l_i^{new} $ 得到归一化的 $ O_i $。

## 为什么这样能让 HBM 少一个数量级

FA1 证明了 IO 复杂度。设 SRAM 大小 $ M $ 满足 $ d \le M \le Nd $：

| 实现 | HBM 访问 |
| --- | --- |
| 标准 attention | $ \Theta(Nd + N^2) $ |
| FlashAttention | $ \Theta(N^2 d^2 M^{-1}) $ |

标准实现要把 $ N \times N $ 的 $ S $、$ P $ 写 HBM，光是这两个就是 $ \Theta(N^2) $。注意这里变的**只是常数**：$ d^2 / M $ 在 $ d \in [64, 128] $、$ M \approx 100 \text{KB} $ 时远小于 1，所以 HBM 访问少几个数量级——但随 $ N $ 的**阶仍是 $ O(N^2) $**，不是次二次。

而且对精确 attention 而言，这个复杂度是**下界**（渐近最优），不只是"比标准快"。FA1 论文的 **Proposition 3** 用反证证明：不存在一个算法，能对 $ M $ 的整个区间 $ [d, Nd] $ **同时**做到 $ o(N^2 d^2 M^{-1}) $ 次 HBM 访问。证明思路：取极端情形 $ M = \Theta(Nd) $，此时 $ N^2 d^2 M^{-1} = \Theta(Nd) $；但输入 $ Q,K,V $（各 $ N\times d $）和输出 $ O $（$ N\times d $）本来就躺在 HBM 里，任何精确算法至少要把它们各读写一遍，所以 HBM 访问注定 $ \Omega(Nd) $。于是没有算法能对所有 $ M $ 同时超越 $ \Theta(N^2 d^2 M^{-1}) $——精确计算下这份 IO 复杂度是**渐近最优**的，能优化的只剩常数因子。

> 注：这是流式算法风格的"对 $ M $ 的整个区间都紧"的下界——靠对抗性地取 $ M=\Theta(Nd) $，让阶掉到必须读写输入/输出的 $ \Omega(Nd) $。见 FlashAttention 论文 **Proposition 3**（arXiv:2205.14135，<https://arxiv.org/abs/2205.14135>）。

论文里 GPT-2 medium（seq 1024、head dim 64、16 heads、batch 64）的实测直接印证：

| | GFLOPs | HBM R/W（GB） | Runtime（ms） |
| --- | ---: | ---: | ---: |
| 标准 attention | 66.6 | 40.3 | 41.7 |
| FlashAttention | 75.2 | **4.4** | **7.3** |

注意 FA1 的 GFLOPs 反而更高（75.2 > 66.6），因为反向要重算。**但 HBM 读写从 40.3 GB 掉到 4.4 GB，时间从 41.7 ms 掉到 7.3 ms。** ,极大缓解了memory bound。

## 重计算：反向不用存 S、P

标准实现反向需要 $ S, P $ 来算梯度，于是 forward 时把它们写进 HBM，或者用梯度检查点换 $ S $。FA1 的做法：**forward 只存 $ O $ 和统计量 $ (m, \ell) $，反向时按块重新算出 $ S = Q K^\top $、$ P = \exp(S - m) $。**

这多出来的 FLOPs 并不亏。反向 pass 的 GEMM 反而因为 HBM 访问更少而更快。完整推导在 [[learning/flash-attention/05-backward-kernel|反向内核逐行读]]。重计算不是"省存 S 的显存"，而是把 $ O(N^2) $ 的显存需求压成 $ O(N) $（只存 $ O $ 和 $ L $），同时反向还更快。

后面三篇往里加的东西（FA2 的并行、FA3 的 TMA/FP8）都是这个骨架上的加速，算法本身没变。

## Reference

- FlashAttention（arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- Self-attention Does Not Need $O(n^2)$ Memory（arXiv:2112.05682）：<https://arxiv.org/abs/2112.05682>
- Online normalizer calculation for softmax（Milakov & Gimelshein）：<https://arxiv.org/abs/1805.02867>
- Reformer: The Efficient Transformer（分块 softmax 前驱）：<https://arxiv.org/abs/2001.04451>
