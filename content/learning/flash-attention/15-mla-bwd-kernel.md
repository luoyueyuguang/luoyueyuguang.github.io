[[learning/flash-attention/13-mla|MLA 前向]] 讲了算法和 absorbed 思路，这一篇讲反向。MLA 反向**复用了普通 attention 的梯度链**（重算 $ S $、$ P = e^{S-L} $、$ dS = P \circ (dP - D) $），**但 FA 仓库把它拆成了三个 kernel**，因为 $ dS $ 是共享中间量，而 dQ、dK、dV 各自要配不同的 B 操作数、归约维也不同。

需要先说清范围：仓库里这条 MLA 反向线是 **top-k 稀疏（DSA 式）的 absorbed MLA 反向**——入口 `_flash_attn_bwd_sparse_mla` 断言 `gather_kv_indices is not None`（"require gather kv indices for backward"），主核类名就叫 `FlashAttentionSparseMLABackwardSm100`，三个文件头都署名 Colfax International。也就是说：**FA 仓库的 MLA 训练路径是"latent + 稀疏 gather"，不是全稠密 MLA 反向**。DeepSeek 自己维护的 [[learning/flash-attention/13-mla|FlashMLA]] 是同一思路的另一套实现，两者别混。

## 反向梯度链

给定 forward 的 $ O, L $ 和反向传入的 $ dO $：

$$
D = \mathrm{rowsum}(dO \circ O), \qquad
dP = dO\, V^\top, \qquad
dS = P \circ (dP - D)
$$

$ V $ 在这里是 **latent**（$ d_c = 512 $ 维），不是展开后的 value。然后各梯度 GEMM：

$$
dV = P^\top dO + dS^\top Q_v, \qquad
dQ_v = dS\, V, \qquad
dQ = dS\, K, \qquad
dK = dS^\top Q
$$

和 [[learning/flash-attention/05-backward-kernel|FA 反向]] 的区别只有两处，都来自 absorbed MLA 的结构（下标沿用 [[learning/flash-attention/13-mla|MLA 前向]]：$ Q $ 是 rope 那一路的 query，$ Q_v $ 是 absorbed 后的 content query，$ K $ 是共享的 rope key）：

- **$ dV $ 有两项。** 前向的 latent $ V $ 一份数据当两个 operand 用——既是 $ Q_v V^\top $ 里的"K"（content 得分），又是 $ P V $ 里的"V"（输出）。所以它的梯度要把两条路的贡献相加：$ P^\top dO $（输出路）加 $ dS^\top Q_v $（得分路）。主核的注释就是这么写的：

  ```python
  # dP.T = V    @ dO.T , N x M x dv
  # dV  += P.T  @ dO   , N x dv x M
  # dV  += dS.T @ Qv   , N x dv x M
  ```

- **$ dQ $ 分成两份。** query 侧有两个投影，所以反向也分两条：$ dQ_v = dS\,V $（content 路，$ d_c = 512 $ 维）和 $ dQ = dS\,K $（rope 路，$ d_h^R = 64 $ 维）。

这些梯度最终还要流回线性层（$ W^{UQ}, W^{UK}, W^{UV}, W^O $ 那一侧）——那一步在模型的反向里做，不在 attention kernel 内。

## 仓库的三个 kernel

`flash_attn/cute/` 里 MLA 反向是这三个：

| 文件 | 类 | 算的 |
| --- | --- | --- |
| `flash_bwd_mla_sm100.py` | `FlashAttentionSparseMLABackwardSm100` | $ dS $ 与 $ dV = P^\top dO + dS^\top Q_v $（主核，重算 $ S, P $，跑 softmax 反向） |
| `flash_bwd_mla_dq_dqv_sm100.py` | `dQdQvGemmKernel` | $ dQ_v = dS\,V $ 和 $ dQ = dS\,K $（两个 query 梯度一起出） |
| `flash_bwd_mla_dk_sm100.py` | `dKGemmKernel` | $ dK = dS^\top Q $（rope key 梯度，按 top-k 索引散射相加） |

**为什么拆**：$ dS $ 的"行"是 query 位置、"列"是 top-k 里选中的 KV 位置，而三个下游梯度要的 operand 完全不一样——$ dQ_v $ 配 $ V $（512 维 latent）、$ dQ $ 配 $ K $（64 维 rope key）、$ dK $ 配 $ Q $。B 操作数的宽度和 gather 方式都不同（dK 还要把结果散射回原序列位置），塞进一个核只会让 MMA 的 tile 形状和 epilogue 分支爆炸；拆开后各自用一个 `tiled_mma` 走最紧的 tile。

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

- `dS` 用 **TMA 加载并 multicast（跨 CTA 广播）**：cluster 形状是 `(1, 2)`，两个 CTA 合起来才覆盖完整的 $ dQ_v = dS\,V $（$ 128 \times 512 $ 的输出，每 CTA 算一半 $ 128 \times 256 $），所以同一块 `dS` tile 用一次 TMA 加载、在 cluster 内广播给两个 CTA，省掉重复的 HBM 读。`create_tma_multicast_mask(..., mcast_mode=2)` 就是沿 N 方向广播。
- `mma_tiler_dQ = (nheads, head_dim_k, tile_k)`、`mma_tiler_dQv = (nheads, head_dim_v // 2, tile_k)`：`// 2` 来自上面那个 `(1, 2)` cluster——dQv 的 N 维（$ d_c = 512 $）被两个 CTA 对半分，每个 CTA 的 tile 是 `(128, 256, 128)`。**只有 cluster 里的 0 号 CTA 额外做 dQ**（注释："Cluster 0 also performs the dQ mma with tile size 128x64"），所以 dQ 的 tile 是 `(128, 64, 128)`，正好是 rope 维 $ d_h^R = 64 $。
- `compute_dQ = const_expr(mK is not None)`：`mK` 是 rope key，只有 absorbed 形状里 query 有单独 rope 一路（`head_dim == 64`）时才传。不传就说明 $ S $ 只剩 $ Q_v V^\top $ 一项（`head_dim == head_dim_v == 512` 的退化情形），此时 `head_dim_k = 0`、只算 `dQv`，省掉整个 dQ 的 MMA 和 epilogue。

## dKGemmKernel：dK 的 dim-major 布局与散射累加

`dKGemmKernel` 算 $ dK = dS^\top Q $。它的输出注释是：

```python
# Output dKaccum: (total_q, seqlen_k, dim), dim-major
```

（这里第一个 mode 写 `total_q` 是沿用了 dS 的行分组；真正落到显存里的 `dk` 是 `(*total_k, dim)`，见 `interface.py` 里 `dk = dk.squeeze(-2)`。之所以要区分，是因为 dS 的列索引经过 top-k gather，不是直接的 KV 行。）

```python
dKaccum = cute.group_modes(dKaccum, 0, 2)
dKaccum_nl = cute.make_tensor(dKaccum.iterator, cute.select(dKaccum.layout, [1, 0]))
self.tiled_mma_dK = utils.sm100.make_trivial_tiled_mma(..., self.mma_tiler_dK[:2], ...)
```

- **dim-major**（`dim` 是快变维）让 contiguous 的那一维正好是"要累加到同一位置的连续元素"——epilogue 才能用宽向量化的 `atomic_add`。

**为什么必须原子加**：$ dK $ 的位置是"top-k 选中的 KV 行"，同一行 KV 会被很多 (query, head) 对选中（MQA 下 128 个 head 共享同一份 KV），每个贡献都要累加到同一个 `dK[seqlen_k]`。所以 epilogue 不是直接写回，而是按索引散射 + 原子加，并跳过 `-1` 哨兵（padding 的无效 top-k 槽位）：

```python
seqlen_k_idx_in_batch = sI_tile[topk_idx]
...
# Skip -1 sentinel slots (invalid top-k entries)
if seqlen_k_idx_in_batch >= 0:
    ptr = elem_pointer(tCgC, (i, j, subtile_idx, seqlen_k_idx))
    cute.arch.atomic_add(ptr=ptr, val=tCrC[i, j])
```

这也是它必须独立成核的另一个原因：epilogue 的"索引 gather + 原子加"模式，跟 $ dQ_v $ 那种规规矩矩的连续写回完全不同。

顺带一个布局细节：喂 MMA 前都要换序——`group_modes(..., 0, 2)` 把 (batch, seqlen) 合成 token 维得到 `(_, heads, tokens)`，再对 dS/Q 用 `cute.select(layout, [2, 1, 0])` 把 head 挪到中间（`dS_mkl`、`Q_nkl`），对 `dKaccum` 用 `select(layout, [1, 0])`（`dKaccum_nl`），让 A/B/C operand 各自满足 `make_trivial_tiled_mma` 要求的 major 模式。此外，`topk % 256 == 1` 时这个核走 2-CTA 指令（`cluster_shape_mn = (2, 1)`、`cta_group = CtaGroup.TWO`），否则单 CTA。

## dV 在主核里

主核 `FlashAttentionSparseMLABackwardSm100` 里 $ dV $ 走 `tiled_mma`，累加器放在 TMEM 的两块偏移上（`self.tmem_offsets_dV = [offset_dV0, offset_dV1]`，对应 `tmem_cols_dVi = (hdimv / num_hdimv_splits) / cta_group_size`），`num_stages_dV = 2`（注释写着 `== hdimv splits`）、`num_epi_stages_dV = 8`（注释写着 `== 2 splits x 4 slots/split`）。`tile_dV = (tile_n, 32)` 说明最内层按 32 列分块写回——因为 $ dV $ 有两个来源（$ P^\top dO $ 和 $ dS^\top Q_v $），分块写能让两块累加与 softmax 的 exp/重缩放重叠，这正是 [[learning/flash-attention/12-flashattention4-bwd-kernel|FA4 反向]] 那套 TMEM 管理思路的复用。

## 一句话

MLA 反向是"稀疏（top-k gather）+ absorbed"的 FA 反向：主核重算 $ S, P $ 得 $ dS $ 和 $ dV = P^\top dO + dS^\top Q_v $，`dQdQvGemmKernel` 沿 $ dS $ 的 KV 轴乘出 $ dQ_v = dS\,V $ 与 $ dQ = dS\,K $（`dS` 走 TMA multicast），`dKGemmKernel` 用 $ dS^\top Q $ 得 dim-major 的 $ dK $ 并原子散射回原序列。拆开是因为三个梯度想要的 B 操作数、宽度和 epilogue 模式都不同；而 $ W^{UK}, W^{UV}, W^{UQ} $ 那几层投影的梯度在模型线性层反向，不在这些 kernel 里。

## Reference

- flash-attention 仓库（flash_attn/cute/flash_bwd_mla_sm100.py、flash_bwd_mla_dq_dqv_sm100.py、flash_bwd_mla_dk_sm100.py）：<https://github.com/Dao-AILab/flash-attention>
- DeepSeek-V2 MLA 论文：<https://arxiv.org/abs/2405.04434>
- DeepSeek-V3.2-Exp / FlashMLA（DSA 稀疏核，与本文的稠密 MLA 反向不同）：<https://github.com/deepseek-ai/FlashMLA>
- FlashAttention 反向算法（dS/dQ/dK/dV，arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
