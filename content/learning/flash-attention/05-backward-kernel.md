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

这里的 $ D $ 是 $ dO \circ O $ 的行和。$ dV $、$ dQ $、$ dK $ 都是 GEMM。**标准实现把 $ S, P $ 留着，就是为了算 $ dS $。**

$ dV, dQ, dK $ 三式都是一眼可见的链式法则：$ O = PV $ 对 $ V $ 是线性的，所以 $ dV = P^\top dO $；$ S = QK^\top $ 对 $ Q $、$ K $ 也是线性的，所以 $ dQ = dS\,K $、$ dK = dS^\top Q $。把指标写开更清楚——$O_{id} = \sum_j P_{ij} V_{jd}$、$S_{ij} = \sum_d Q_{id} K_{jd}$，于是

$$
dV_{jd} = \sum_i \frac{\partial O_{id}}{\partial V_{jd}}\, dO_{id} = \sum_i P_{ij}\, dO_{id},
\qquad
dQ_{id} = \sum_j \frac{\partial S_{ij}}{\partial Q_{id}}\, dS_{ij} = \sum_j dS_{ij}\, K_{jd},
\qquad
dK_{jd} = \sum_i dS_{ij}\, Q_{id}
$$

对线性项来说偏导就是"另一个输入矩阵"本身，全是 GEMM，而且都能分块做：$ dV $、$ dK $ 沿列块 $ j $ 可加，$ dQ $ 沿行块 $ i $ 可加。（softmax 的 $ 1/\sqrt d $ 缩放折在 $ S $ 里：$ dV $ 不经过 $ S $、不带它，$ dQ $ 和 $ dK $ 才要乘回 `scale_softmax`；这就是 kernel 里 `acc_dk *= params.scale_softmax_rp_dropout` 的来历。）真正需要展开的只有 $ dS $——softmax 是非线性的。逐行看，$ p = \mathrm{softmax}(s) $ 的 Jacobian 是 $ \mathrm{diag}(p) - p p^\top $，所以

$$
dS_{ij} = P_{ij}\Big(dP_{ij} - \sum_k P_{ik}\, dP_{ik}\Big)
$$

右边那个行和换个写法就能省掉一次 $ N\times N $ 的乘法：

$$
\sum_k P_{ik}\, dP_{ik}
= \sum_k P_{ik} \sum_d dO_{id} V_{kd}
= \sum_d dO_{id} \underbrace{\sum_k P_{ik} V_{kd}}_{O_{id}}
= \sum_d dO_{id} O_{id} =: D_i
$$

第一步代入 $ dP = dO\,V^\top $，第二步交换求和次序，第三步用 $ O = PV $。所以只要 forward 的 $ O $（本来就得存）和反向进来的 $ dO $，一次 elementwise 点积再按行求和就得到 $ D $——这正是 $ D = \mathrm{rowsum}(dO \circ O) $ 的来源，也解释了为什么 FA 的反向不需要重算 $ dP $ 的第二项。

## 反向只存三样东西

FA 的 forward 存了 $ O $（输出）和 $ L $（logsumexp）。反向开始时多算一个 $ D = \mathrm{rowsum}(dO \circ O) $。

有 $ O, L $ 就够重算 $ P $ 吗？够。因为：

$$
P = \mathrm{softmax}(S) = \frac{e^{S - m}}{\ell} = e^{S - (m + \log \ell)} = e^{S - L}
$$

所以**每个 block 重算出 $ S = QK^\top $ 后，直接 $ e^{S - L} $ 就是 $ P $**，不需要再存 $ m, \ell $ 两个。这就是 [[learning/flash-attention/06-flashattention2|FA2]] 说的"只存 logsumexp，不存 max 和 sum 两个"。FA1 还要存 $ m, \ell $，FA2 把它俩合并成 $ L $。

$ D $ 不需要单独物化 $ dP $：kernel 里就是读 $ O $ 和 $ dO $，逐 token 求 $ \sum_d dO_{i,d} O_{i,d} $，写出 `softmax_d`（即 $ D $）。

注意这个点积在代码里有两条路径：

- **非并行路径**（`compute_dq_dk_dv`，一个 thread block 包办一个 (batch, head) 的所有列块）：根本不另开 kernel。主反向 kernel 在第一个列块（模板参数 `Is_first`，传的是 `n_block = n_block_max - 1`）里逐个 $ m $ 块调 `dot_do_o`（见 `flash_bwd_kernel.h`）就地算完 $ D $，同时用 `clear(acc_dq)` 把**寄存器里的 dQ 累加器**清零。清的是寄存器而不是全局缓冲：$ dQ $ 要跨列块累加，后续列块会用 `cute::copy` 从 `dQ_accum` 读回上一轮的结果再累加（`Is_first` 之后走的就是这个 `else` 分支），所以只有第一轮需要清零。
- **序列维并行路径**（`run_flash_bwd_seqk_parallel`；当前启动模板 `run_flash_bwd` 走的都是这条）：$ dQ $ 要跨 block 原子加，主循环里 `Is_first` 是编译期常量 `false`、算不了 $ D $，于是先单独跑一个 `flash_bwd_dot_do_o_kernel</*Clear_dQaccum=*/true>`（定义在 `flash_bwd_launch_template.h`，内部调 `csrc/flash_attn/src/flash_bwd_preprocess_kernel.h` 的 `compute_dot_do_o`）把 $ D $ 算出来、**顺便把 `dQ_accum` 清零**（后面每个列块都要往同一块缓冲原子加）。源码注释写得很直白：*"Just compute dot(do, o) and write the result (softmax_d) to global memory as a separate kernel. This is used in the case where we want to parallelize the backward across seqlen_k."* 这个 kernel 只有 elementwise + reduction，是 memory-bound 的，单独跑一次也很快。

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
4. `flash_bwd_kernel.h:536` 实际写的是 `FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2)`。这个模板参数控制 `max` 怎么用：`scale_max=true` 时算 $ \mathrm{exp2}(\text{scores}\cdot\text{scale} - m\cdot\text{scale}) $（即 $ e^{S-m} $），`false` 时改用 `M_LOG2E` 缩放 $ L $，算 $ \mathrm{exp2}(\text{scores}\cdot\text{scale} - L\log_2 e) $，也就是 $ e^{S - L} $。一个参数就切换了"减行最大"和"减 logsumexp"两种语义——$ L $ 已经等于 $ m + \log \ell $，所以后者直接就是 $ P $，不用再除 $ \ell $。
5. `P` 转成 fp16/bf16 后还要经 `smem_tiled_copy_PdS` 写回共享内存：后面 $ dV = P^\top dO $ 和 $ dK = dS^\top Q $ 两个 GEMM 要把 $ P $、$ dS $ 当 A 操作数从 smem 读（$ dS $ 同理，先 `convert_type` 再写 smem）。**每轮循环里 $ S, P, dS $ 都只在片上转一圈，一个都不落 HBM。**

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

- 复用了 `acc_dp` 的内存来放 `dS`（`make_tensor(acc_dp.data(), scores.layout())`），省寄存器。
- `pointwise_mult(p, dp, d) = p * (dp - d)`，就是 $ dS = P \circ (dP - D) $。
- `dP_sum(mi)` 就是 $ D $（从 `softmax_d` 读进来的 `gdPsum`），前面推过它等于 $ \sum_d dO_{id}O_{id} $。
- 那个 `p >= 0` 分支是给 dropout 用的：反向的 dropout mask 被编码进 `P` 的**符号位**（`apply_dropout</*encode_dropout_in_sign_bit=*/true>`），所以 `p < 0` 表示这个位置被 drop 掉，此时按 $ dS = P\cdot d $ 而不是 $ P \circ (dP - D) $ 处理。

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
- `tiled_mma_dkv` 同时服务 $ dV $ 和 $ dK $ 两个 GEMM：两者的 C 都是 $ B_c \times d $，A 操作数又都从转置视图的 smem 读（`smem_tiled_copy_PdSt` 用的是 `SmemCopyAtomTransposed`），所以能共用同一个 mma 形状；`tiled_mma_dq` 的 C 是 $ B_r \times d $，单独一份。

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

这就是 [[learning/flash-attention/06-flashattention2|FA2]] 说的"反向按列块并行，用 atomic add 在 block 之间合并 $ dQ $"。

## 为什么重算反而更快

论文里 GPT-2 medium 那个微基准（论文 Figure 2 左，另见 [[learning/flash-attention/02-online-softmax|online softmax 与分块]]）：反向把 $ S, P $ 落在 SRAM 里重算，HBM 读写从标准实现的 40.3 GB 掉到 4.4 GB（这里是 forward + backward 合计），总时间 41.7 ms → 7.3 ms。虽然 FLOPs 变多（66.6 → 75.2 GFLOPs），但 attention 是 memory-bound，**省的约 36 GB HBM 读写远远值回多出来的 8.6 GFLOPs。**

用一句话收尾这个系列的后向：**forward 用 $ O, L $ 换掉 $ S, P $ 的 $ O(N^2) $ 显存；backward 用 $ L $ 和 $ D $ 现场重算 $ P $，把重算的 FLOPs 花在 SRAM 里，换来 HBM 访问从 $ \Theta(N^2) $ 降到 $ \Theta(N^2 d^2 M^{-1}) $（论文 Theorem 5）。**

## Reference

- flash-attention 仓库（csrc/flash_attn/src/flash_bwd_kernel.h、flash_bwd_preprocess_kernel.h）：<https://github.com/Dao-AILab/flash-attention>
- FlashAttention 论文（forward/backward 算法）：<https://arxiv.org/abs/2205.14135>
- FlashAttention-2 论文（logsumexp $ L $、反向按列块并行）：<https://arxiv.org/abs/2307.08691>
