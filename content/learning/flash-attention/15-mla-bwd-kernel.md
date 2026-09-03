[[learning/flash-attention/13-mla|MLA 前向]] 讲了算法和 absorbed 思路，这一篇讲反向。MLA 反向的**梯度链和普通 attention 一样**（重算 `$ S $`、`$ P = e^{S-L} $`、`$ dS = P \circ (dP - D) $`，再得 dQ/dK/dV），**但 FA 仓库把它拆成了三个 kernel**，因为 `$ dS $` 是共享中间量，而 dQ、dK、dV 需要的 `$ dS $` 布局各不相同。注意这是 FA 仓库对**稠密 MLA** 的反向实现；DeepSeek-V3.2 的 token 级稀疏 DSA 是 [[learning/flash-attention/13-mla|另一套 FlashMLA]] 核（DeepSeek 自己维护），不在这里。

## 反向梯度链（和普通 attention 同源）

给定 forward 的 `$ O, L $` 和反向传入的 `$ dO $`：

$$
D = \mathrm{rowsum}(dO \circ O), \qquad
dP = dO\, V^\top, \qquad
dS = P \circ (dP - D)
$$

然后三个梯度 GEMM：

$$
dQ = dS\, K, \qquad dV = P^\top dO, \qquad dK = dS^\top Q
$$

和 [[learning/flash-attention/04-backward-kernel|FA 反向]] 一模一样。区别只在 MLA 的 `$ Q, K, V $` 都是 latent 展开的、且全 head 共享 `$ c^{KV} $`。所以 `$ dQ, dK, dV $` 是**对展开后的 content 张量的梯度**，之后还要流回 `$ W^{UQ}, W^{UK}, W^{UV} $`（那一步在模型的线性层反向里做，不在 attention kernel 内）。

## 仓库的三个 kernel

`flash_attn/cute/` 里 MLA 反向是这三个：

| 文件 | 类 | 算的 |
| --- | --- | --- |
| `flash_bwd_mla_sm100.py` | `FlashAttentionSparseMLABackwardSm100` | `$ dS $`、`$ dP $`、`$ dV $`（主核，重算 softmax） |
| `flash_bwd_mla_dq_dqv_sm100.py` | `dQdQvGemmKernel` | `$ dQ = dS\,K $` 和 `$ dQv = dS\,V $` |
| `flash_bwd_mla_dk_sm100.py` | `dKGemmKernel` | `$ dK = dS^\top Q $` |

**拆分的原因**是 `$ dS $` 作为 A 操作数时，dQ 要配 `$ K $`（`dS · K`）、dQv 要配 `$ V $`（`dS · V`）、dK 要配 `$ Q $`（`dS^\top Q`）。三者的归约维和 B 操作数不同，硬塞进一个核会让 MMA 模板爆炸、寄存器也扛不住，所以拆开，各自喂不同的 `tiled_mma`。

## dQdQvGemmKernel：dS 的 TMA multicast

`dQdQvGemmKernel` 的开头注释就点明了：

```python
# Performs both dQ = dS @ K and dQv = dS @ V, where K and V are ...
# covers the full dQv mma. dS is loaded via TMA and multicast across the CTAs.
# Cluster 0 also performs the dQ mma with tile size 128x64.
```

```python
self.mma_tiler_dQ = (self.nheads, self.head_dim_k, self.tile_k)
self.mma_tiler_dQv = (self.nheads, self.head_dim_v // 2, self.tile_k)
self.compute_dQ = const_expr(mK is not None)   # 可只算 dQv
```

- `dS` 用 **TMA 加载并 multicast（跨 CTA 广播）**：因为 MLA 里 `$ c^{KV} $` 全 head 共享，`$ dS $` 是多个 head 共用的中间量，一个 CTA 加载、cluster 里广播，省重复 HBM 读。
- `mma_tiler_dQ = (nheads, head_dim_k, tile_k)`、`mma_tiler_dQv = (nheads, head_dim_v//2, tile_k)`：dQ 和 dQv 用不同的 tile，`dQv` 的 head_dim 减半（`// 2`）是因为它和 `dQ` 分块并行。
- `compute_dQ = mK is not None`：有些场景只反传 value（`.alibi`/只有 `$ dV $` 路径），此时只有 `dQv` 没有 `dQ`，`mK` 未传就跳过 `$ dQ $`。

## dKGemmKernel：dK 的单位阵布局

`dKGemmKernel` 算 `$ dK = dS^\top Q $`，输出是 **dim-major** 的 `dKaccum`：

```python
# Output dKaccum: (total_q, seqlen_k, dim), dim-major
```

```python
dKaccum = cute.group_modes(dKaccum, 0, 2)
dKaccum_nl = cute.make_tensor(dKaccum.iterator, cute.select(dKaccum.layout, [1, 0]))
self.tiled_mma_dK = utils.sm100.make_trivial_tiled_mma(..., self.mma_tiler_dK[:2], ...)
```

- `dS^\top Q` 的输出形状是 `(seqlen_k, head_dim)`，所以 `dKaccum` 按 **dim-major**（`dim` 是快变维）排，方便后续投影的反向。
- 为什么要单独一个 kernel：`$ dK $` 的输入 `$ dS^\top $` 和 `$ Q $` 都要转置/换布局（`group_modes`、`make_tensor(..., [1,0])` 交换顺序），和 `$ dQ $` 的布局差异很大。

## dV 在主核里

主核 `FlashAttentionSparseMLABackwardSm100` 里 `$ dV $` 用的是 `$ tiled_mma $` 配合 `tmem_offsets_dV`（`self.tmem_offsets_dV = [offset_dV0, offset_dV1]`），`num_stages_dV = 2`（== hdimv splits），`num_epi_stages_dV = 8`。`$ dV = P^\top dO $` 走 `tiled_mma`，累加器拆成两块（`tile_dV = (tile_n, 32)`）分阶段写。它和 `$ dS $` 的 softmax 重叠，复用 [[learning/flash-attention/10-flashattention4-bwd-kernel|FA4 反向]] 那套 TMEM 管理。

## 一句话

MLA 反向就是"普通 FA 反向 + 拆成三个核"：主核重算 `$ S, P $` 得 `$ dS $`（顺带 `$ dV $`），`dQdQvGemmKernel` 用 `dS·K`、`dS·V` 得 `$ dQ, dQv $`（dS 走 TMA multicast），`dKGemmKernel` 用 `dS^\top Q` 得 dim-major 的 `$ dK $`。拆开是因为 `$ dS $` 对三个梯度的布局要求不同；而 `$ W^{UK}, W^{UV}, W^{UQ} $` 的梯度在模型线性层反向，不在这个 kernel 里。

## Reference

- flash-attention 仓库（flash_attn/cute/flash_bwd_mla_sm100.py、flash_bwd_mla_dq_dqv_sm100.py、flash_bwd_mla_dk_sm100.py）：<https://github.com/Dao-AILab/flash-attention>
- DeepSeek-V2 MLA 论文：<https://arxiv.org/abs/2405.04434>
- DeepSeek-V3.2-Exp / FlashMLA（DSA 稀疏核，与本文的稠密 MLA 反向不同）：<https://github.com/deepseek-ai/FlashMLA>
- FlashAttention 反向算法（dS/dQ/dK/dV，arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
