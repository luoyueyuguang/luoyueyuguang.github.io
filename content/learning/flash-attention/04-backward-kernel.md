上一篇读了 forward，这一篇读 backward。反向的难点不是梯度公式，而是**显存**：标准实现反向要 $ S $ 和 $ P $ 来算梯度，这俩都是 $ N \times N $，加上 $ dO, O, V $ 就爆了。FlashAttention 的反向靠**重计算**解决：forward 只存 $ O $ 和 $ L $（logsumexp），反向时每个 block 现场重算 $ S, P $。

> **重计算的本质是"用 FLOPs 换 HBM 访问"。** 反向会多算一遍 $ Q K^\top $ 和 $ e^{S-L} $，但这些都在 SRAM 里，不碰 HBM；而省的是一次性写读 $ N \times N $ 的 $ S, P $。对 memory-bound 的 attention，这笔交换划算。

## 先看标准实现的反向要什么

注意力正向（逐行）是 $ O = \mathrm{softmax}(QK^\top) V $。设 $ P = \mathrm{softmax}(S) $，$ S = QK^\top $。反传要算 $ dQ, dK, dV $，链条是：

$$
D = \mathrm{rowsum}(dO \circ O)
$$

$$
dP = dO\, V^\top, \qquad dS = P \circ (dP - D)
$$

$$
dV = P^\top dO, \qquad dQ = dS\, K, \qquad dK = dS^\top Q
$$

这里的 $ D $ 是 $ dO \odot O $ 的行和。$ dV $、$ dQ $、$ dK $ 都是 GEMM or 内积。**标准实现把 $ S, P $ 留着，就是为了算 $ dS $。**

## 反向只存三样东西

FA 的 forward 存了 $ O $（输出）和 $ L $（logsumexp）。反向开始时多算一个 $ D = \mathrm{rowsum}(dO \circ O) $。

有 $ O, L $ 就够重算 $ P $ 吗？够。因为：

$$
P = \mathrm{softmax}(S) = \frac{e^{S - m}}{\ell} = e^{S - (m + \log \ell)} = e^{S - L}
$$

所以**每个 block 重算出 $ S = QK^\top $ 后，直接 $ e^{S - L} $ 就是 $ P $**，不需要再存 $ m, \ell $ 两个。这就是 [[learning/flash-attention/05-flashattention2|FA2]] 说的"只存 logsumexp，不存 max 和 sum 两个"。FA1 还要存 $ m, \ell $，FA2 把它俩合并成 $ L $。

`flash_attn/ops/` 里有一个 `flash_attn_backward` 的前置 kernel `flash_bwd_preprocess_kernel.h`，专门算 $ D $。它读 $ O $ 和 $ dO $，逐 token 求 $ \sum_d dO_{i,d} O_{i,d} $，写出 `softmax_d`（就是 $ D $）。这个 kernel 只有 elementwise + reduction，是 memory-bound 的，单独跑一次也很快。

## 主循环：重算 S 和 P

核心循环在 `flash_bwd_kernel.h` 的 `compute_dq_dk_dv_1colblock`（遍历 `m_block`，每个 block 处理一段行）。先看重算段：

```cpp
for (; m_block >= m_block_min; --m_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    cute::cp_async_wait<0>();
    __syncthreads();

    // S = Q K^T
    FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    ...
    // P = exp(S - L)，用存储的 L（logsumexp）
    FLASH_NAMESPACE::scale_apply_exp2<scale_max=...>(scores, lse, params.scale_softmax_log2);
```

逐行：

1. `clear(acc_s)` + `cp_async_wait` + `__syncthreads` 先把累加器清空、等 K/V 到位。
2. `gemm(acc_s, tSrQ, tSrK, ...)` 重算 $ S = Q_i K_j^\top $。**这是重计算多出来的一遍 GEMM。**
3. `scores` 是 $ S $（重排成 row/col 布局，方便逐行操作）。
4. `scale_apply_exp2(scores, lse, ...)` 用存储的 $ L $ 做 $ e^{S - L} $，得到 $ P $。注意这里传的是 `lse`，不是行最大，所以它直接就是 $ P $，不用再除以 $ \ell $。

## dS = P ∘ (dP − D)

$dP = dO V^\top$ 用第二个 GEMM（`acc_dp`）算，然后就地做 `pointwise_mult`：

```cpp
// dP = dO · V^T
FLASH_NAMESPACE::gemm(acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp, ...);

Tensor dS = make_tensor(acc_dp.data(), scores.layout());
auto pointwise_mult = [](float p, float dp, float d) {
    return p * (!Is_dropout || p >= 0 ? dp - d : d);
};
for (int mi = 0; mi < size<0>(dS); ++mi) {
    for (int ni = 0; ni < size<1>(dS); ++ni) {
        float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
        dS(mi, ni) = scaled_ds;
    }
}
```

- 复用了 `acc_dp` 的内存来放 `dS`，省寄存器。
- `pointwise_mult(p, dp, d) = p * (dp - d)`，就是 $ dS = P \circ (dP - D) $。
- `dP_sum(mi)` 就是 $ D $（从 `softmax_d` 读进来的，`gdPsum`）。逐行减 $ D $：这是 softmax 的 Jacobian 那一步，$ P $ 对 $ S $ 的导是 $ P $ 减去一个 $ P D $ 修正，而 $ D = \mathrm{rowsum}(dO \circ O) $ 正是那个修正。

## dQ、dK、dV：三个 GEMM

$ dQ = dS K $，$ dK = dS^\top Q $，$ dV = P^\top dO $。三行代码：

```cpp
// dV += P^T · dO
FLASH_NAMESPACE::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv, ...);

// dQ += dS · K
FLASH_NAMESPACE::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq, ...);

// dK += dS^T · Q
FLASH_NAMESPACE::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv, ...);
```

- `tdVrPt` 是 $ P $（A，不过这里把 $ P^\top $ 当 A 用），`tdVrdO` 是 $ dO $（B）。`gemm` 算 $ P^\top dO $ 累进 `acc_dv`。
- `acc_dq`、`acc_dk` 都带 `+=` 的累加语义，分别累进各自的 fp32 accumulator。
- `tiled_mma_dkv` 同时服务 $ dV $ 和 $ dK $ 两个 GEMM（A、B 互补），各用一次 `gemm` 分别更新 `acc_dv`、`acc_dk`；`tiled_mma_dq` 单独算 $ dQ $。

## 序列维并行时的原子加

FA2 反向按**列块**并行（每个 thread block 负责一段 $ K/V $ 列块），而不是行块。这样 $ dK, dV $ 可以各自独立累加，但 $ dQ $ 会被多个 block 都贡献一部分（不同列块都往同一段 `dQ` 加）。所以在 `Seq_parallel` 路径里用原子加：

```cpp
if (!Seq_parallel) {
    cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);
} else {
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
}
```

这就是 [[learning/flash-attention/05-flashattention2|FA2]] 说的"反向按列块并行，用 atomic add 在 block 之间合并 $ dQ $"。

## 为什么重算反而更快

论文里 GPT-2 medium 那个微基准（见 [[learning/flash-attention/02-online-softmax|在线 softmax 与分块]]）：反向把 $ S, P $ 落在 SRAM 里重算，HBM 读写从标准实现的 40.3 GB 掉到 4.4 GB（这里是 forward + backward 合计），总时间 41.7 ms → 7.3 ms。虽然 FLOPs 变多（66.6 → 75.2 GFLOPs），但 attention 是 memory-bound，**省的 36 GB HBM 读写远远值回多出来的几十 GFLOPs。**

用一句话收尾这个系列的后向：**forward 用 $ O, L $ 换掉 $ S, P $ 的 $ O(N^2) $ 显存；backward 用 $ L $ 和 $ D $ 现场重算 $ P $，把重算的 FLOPs 花在 SRAM 里，换来 HBM 访问从 $ \Theta(N^2) $ 降到 $ \Theta(N^2 d^2 / M) $。**

## Reference

- flash-attention 仓库（csrc/flash_attn/src/flash_bwd_kernel.h、flash_bwd_preprocess_kernel.h）：<https://github.com/Dao-AILab/flash-attention>
- FlashAttention 论文（forward/backward 算法）：<https://arxiv.org/abs/2205.14135>
- FlashAttention-2 论文（logsumexp $ L $、反向按列块并行）：<https://arxiv.org/abs/2307.08691>
