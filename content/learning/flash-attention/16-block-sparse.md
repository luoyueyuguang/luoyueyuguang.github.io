FA1 论文除了"IO-aware 精确 attention"，还有**一个重要的近似扩展**：把 FA 推广到 **block-sparse attention**。[[learning/flash-attention/02-online-softmax|FA1 算法篇]] 只在开头提了一句，这篇展开。它不在"四代 FlashAttention"的优化线路上，但它是 FA1 论文的正式贡献之一，也是"用稀疏省 IO"这条路线的起点。

> **block-sparse FA 是"精确 attention 当稀疏、省 IO 当近似"的折中：注意力矩阵按块稀疏化，只在非零块上跑 FA，IO 复杂度随稀疏度下降。**

## 问题定义

给定 `$ Q, K, V \in \mathbb{R}^{N \times d} $` 和一个**块稀疏掩码** `$ M \in \{0,1\}^{N/B_r \times N/B_c} $`，我们想算：

$$
S = QK^\top, \qquad P = \mathrm{softmax}(S \odot \mathbb{1}_{M}), \qquad O = P V
$$

其中 `$ (S \odot \mathbb{1}_M)_{kl} = S_{kl} $` 若 `$ M_{kl}=1 $`，否则 `$ -\infty $`。

关键约束：**掩码必须是"块形式"**。也就是说 `$ M $` 只在 `$ B_r \times B_c $` 的粒度上给 0/1，不能精细到单个 `$ (k,l) $`。这样 FA 的分块才和稀疏对齐：一个块要么全算、要么全跳过。

## 算法：和 FA 一样，只跳过零块

`BlockSparse FlashAttention` 的算法**和 [[learning/flash-attention/02-online-softmax|FA 的算法]] 完全一样**，唯一区别是**跳过 `$ M_{ij}=0 $` 的块**。内层循环遍历 col block `$ j $` 时，先查 `$ M_{ij} $`，是 0 就 `continue`。于是：

- 计算的块数 ∝ 非零块比例 `$ s $`。
- softmax 的在线归一化照常（因为零块被跳过，softmax 分母只来自非零块，天然正确）。这其实是"因果 mask"的推广：因果就是上三角全 0 的块稀疏。

## IO 复杂度：随稀疏度下降

FA1 证明了 block-sparse FA 的 IO 复杂度：

$$
\Theta\!\left(Nd + N^2 d^2 M^{-1} s\right)
$$

`s = 非零块比例`。对比 [[learning/flash-attention/02-online-softmax|FA]] 的 `$ \Theta(N^2 d^2 M^{-1}) $`，**大宗项直接乘了 `$ s $`**。常见的稀疏设置：

- `$ s = N^{-1/2} $` → `$ \Theta(N\sqrt{N}) $`
- `$ s = N^{-1} \log N $` → `$ \Theta(N \log N) $`

对长序列，`$ N $` 巨大时这两种都能把 IO 从二次降到近线性。

## 稀疏模式：butterfly

论文下游实验用的是**固定的 butterfly 稀疏模式**（来自 Pixelated 那篇 `dao2021pixelated`）。butterfly 是"稀疏矩阵 → 两两块相乘再加"的结构，已经被证明能**逼近任意稀疏度**，而且天然适合 FA 的分块。选它是为了"固定模式可复用 + 表达力够 + 能落成现成 kernel"。

## 结果

论文里的关键结果：

- **LRA 基准**：block-sparse FA 相比标准 attention **快 2.8×**，效果和标准 attention **相当**（这是个"近似但几乎无损"的结果）。
- **长上下文**：块稀疏让模型能跑更长序列。FA1 的摘要提到长文档分类 lift 6.4 分、Path-X（seq 16K）首次超过随机（61.4%）、Path-256（seq 64K，63.1%）。

## 仓库实现

`flash-attention` 仓库里有：

- `flash_blocksparse_attn_interface.py`：`flash_blocksparse_attn_func`，把 block mask（`row_count`/`col_count`/块索引）传给 CUDA kernel。
- `flash_blocksparse_attention.py`：`FlashBlocksparseAttention` / `FlashBlocksparseMHA` 两个 `nn.Module` 封装，内部注册稀疏布局为 buffer、掉 `flash_blocksparse_attn_func`。

mask 在 PyTorch 侧用块稀疏布局表达（每行非零块数 + 列索引），kernel 侧按"非零块"遍历，跳过零块。这和 [[learning/flash-attention/14-launch-scheduling|调度层]] 里因果跳过一半块的逻辑同源，只是 mask 从"因果"换成"任意块稀疏"。

## 一句话

block-sparse FA 把"精确的 FA"改成"按块稀疏的近似"：**掩码必须块形式，非零块照常跑在线 softmax，零块跳过，IO 复杂度乘非零比例 `$ s $`**。它是 FA1 论文"除 IO-aware 外的另一半贡献"，也是后续围绕稀疏做长上下文（[[learning/flash-attention/13-mla|MLA]] 的压缩、各种 block-sparse 路线）的起点。

## Reference

- FlashAttention（arXiv:2205.14135，block-sparse 扩展与 IO 复杂度）：<https://arxiv.org/abs/2205.14135>
- Pixelated Butterfly（butterfly 稀疏模式）：<https://arxiv.org/abs/2112.00029>
- flash-attention 仓库（flash_blocksparse_attention.py、flash_blocksparse_attn_interface.py）：<https://github.com/Dao-AILab/flash-attention>
- Long Range Arena（LRA 基准）：<https://arxiv.org/abs/2011.04006>
