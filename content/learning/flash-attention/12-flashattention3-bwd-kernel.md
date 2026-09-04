[[learning/flash-attention/04-backward-kernel|FA1/FA2 反向]] 讲的是数学和 Ampere 内核，这一篇读 **FA3 的 Hopper 反向内核** `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` 的 `CollectiveMainloopBwdSm90`。数学和 04 一模一样（还是 $ P = e^{S-L} $、$ dS = P \circ (dP - D) $、dQ/dK/dV 三个 GEMM），但 FA3 把它塞进了 warp specialization，并且**复用同一个 MMA tiling 同时算 S 和 dP**。

## 与 FA2 反向的差别

FA3 反向的"算子"没变，变的是**怎么调度**：

| | FA2（04） | FA3 反向 |
| --- | --- | --- |
| S 与 dP | 两个独立 MMA tiling | `tiled_mma_SdP`（一个 tiling 复用，`SwapAB` 控制哪个当 A） |
| L / D | 从 smem 读 | 小 head dim 用 `__shfl_sync`（`ShuffleLSE`/`ShuffledPsum`） |
| 调度 | 无 warp specialization | producer（TMA 搬 Q/K/V/dO）+ consumer（`bwd_step`） |
| dQ 归约 | `atomicAdd` | `atomicAdd`，或 head dim<256 时走 TMA reduce |
| hdim 256 | — | `Slice_dQKV_Mma` 拆开 dQ/dKV 的 MMA 降寄存器压力 |

## bwd_step：一次算一个 m_block

`bwd_step` 是 consumer warpgroup 的主循环体（`for (; m_block < m_block_max; ++m_block)` 里调用）。先看重算 S 和 dP（用同一个 `tiled_mma_SdP`）：

```cpp
auto bwd_step = [&](int m_block, auto mask_fn) {
    Tensor tSrS = partition_fragment_C(tiled_mma_SdP, ...);
    consumer_wait(pipeline_q, smem_pipe_read);                       // 等 Q 到位
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
        tiled_mma_SdP, tSrQ(..., smem_pipe_read.index()), tSrK, tSrS);   // S = Q K^T

    Tensor tLSErLSE = ...;                                            // 读 logsumexp L
    ...
    consumer_wait(pipeline_do, smem_pipe_read_do_cur);                 // 等 dO 到位
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
        tiled_mma_dP, tdPrdO(..., smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);  // dP = dO V^T
    warpgroup_wait<1>();
```

- `consumer_wait(pipeline_q)` / `pipeline_do`：FA3 的 producer warpgroup 用 TMA 把 Q 和 dO 搬进 smem，这里等它们到位。
- **`tiled_mma_SdP` 同时算 S 和 dP**：`SwapAB=SdP_swapAB` 决定哪个操作数当 A。这是因为 S（$ QK^\top $）和 dP（$ dO\,V^\top $）的形状互补，复用同一个 tiling 能少一套 MMA 模板。`wg_wait=-1` 是异步发出，不等。
- `tLSErLSE` 读的是 forward 存的 **logsumexp $ L $**（FA2 微调 2 的产物），后面直接 $ P = e^{S - L} $。

## P = exp(S − L)，再算 dS

```cpp
scoremod_premask_fn(tSrS);            // softcap（如果启用）
mask_fn(tSrS, m_block);               // 因果/local mask
for (int mi ...) {
    float const lse_scaled = ...;      // L
    for (int ni ...) {
        scores(mi, ni) = exp2f(scores(mi, ni) * params.softmax_scale_log2 - lse_scaled);   // e^{S·log2e − L·log2e}
    }
}
Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
for (int mi ...) {
    float const dP_sum_cur = ...;      // D = rowsum(dO ∘ O)
    for (int ni ...) {
        dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur);    // dS = P ∘ (dP − D)
        if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
    }
}
```

- `exp2f(scores * scale_log2 - lse_scaled)`：$ e^{S - L} = P $。用 `exp2f` 是因为前向在 [[learning/flash-attention/03-forward-kernel|softmax_rescale_o]] 里也这么干，$ e^x = 2^{x\log_2 e} $ 能合成 `ffma`。
- `dS = P ⊙ (dP − D)`：和 [[learning/flash-attention/04-backward-kernel|FA2 反向]] 的 `pointwise_mult` 一模一样。`dP_sum_cur` 是 $ D = \mathrm{rowsum}(dO \circ O) $。
- softcap 时 `dS *= dtanh`：softmax 的链式法则多一项 $ dtanh $（`tanh` 的导）。

## P、dS 转精度 + 三个梯度 GEMM

```cpp
Tensor rP = make_tensor_like<Element>(tSrS);
flash::convert_type_out(tSrS, rP);              // P → fp16/bf16
Tensor rdS = make_tensor_like<Element>(tdPrdP);
flash::convert_type_out(tdPrdP, rdS);            // dS → fp16/bf16
...
if constexpr (!Slice_dQKV_Mma) {
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(
        tiled_mma_dKV, tdVrP_cur, tdVrdO(..., smem_pipe_read_do_cur.index()), tdVrdV);   // dV += P^T·dO
    ...
    flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(
        tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);                                          // dQ = dS·K（+=）
    pipeline_do.consumer_release(smem_pipe_read_do_cur);                                  // 释放 dO
    ...
    flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(
        tiled_mma_dKV, tdKrdS_cur, tdKrQ(..., smem_pipe_read.index()), tdKrdK);            // dK += dS^T·Q
    ...
}
```

- `convert_type_out` 把 fp32 的 $ P $ 和 $ dS $ 转成 fp16/bf16，喂给后面的 MMA。
- 三个 GEMM 顺序：**$ dV = P^\top dO $**（`tiled_mma_dKV`）、**$ dQ = dS\,K $**（`tiled_mma_dQ`）、**$ dK = dS^\top Q $**（`tiled_mma_dKV`，`SwapAB` 切换）。`dKV_swapAB` / `dQ_swapAB` 是 tiling 级的选择（谁当 A、谁当 B），非运行时决定。
- `wg_wait=1`：等前一个 MMA 完成再进入下一步，因为存在数据依赖（比如 `dQ` 用 `dS`，`dS` 由前面的 `exp2f` 算出）。
- `pipeline_do.consumer_release`：dO 这块用完，可以还给 producer 复用。

## dQ 的原子归约（或 TMA reduce）

`dQ` 需要在一个 `m_block` 的范围内累加（按 [[learning/flash-attention/04-backward-kernel|FA2 反向]] 的说法，反向按列块并行，dQ 靠原子加合并）：

```cpp
if constexpr (dQacc_use_TMA) {           // head dim < 256
    // 把 tdQrdQ 写进 smem，再交给 TMA 原子 reduce 到全局
    cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
    ...
} else {                                  // head dim >= 256
    Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
    Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
    #pragma unroll
    for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
}
```

- `dQacc_use_TMA = (kHeadDim < 256)`：head dim 小、dQ 一块装得下，走 smem + TMA 的 `cp.async.bulk reduce`；否则直接 `atomicAdd`（`recast<float4>` 让原子按 128-bit 粒度加，省一半原子操作）。
- `tdQgdQaccum(_, _, _, m_block)`：给当前 `m_block` 的那块 $ dQ $ 累加器加。这就是 FA3 反向的"列块并行，dQ 用原子合并"。

## hdim 256：Slice_dQKV_Mma

`Slice_dQKV_Mma` 分支只在 `kHeadDim == 256 && !dQacc_use_TMA && dQ_swapAB && AtomLayoutMdQ==1 && NumMmaWarpGroups==2` 时走。它把 `dQ` / `dKV` 的 MMA 按 `M_slice` 拆成两半（`M_slice=0/1`），中间穿插写 smem，**降低寄存器峰值**（head dim 256 时累加器太大，不分片会 spill）。这就是 [[learning/flash-attention/06-flashattention3|FA3 算法篇]] 说的"寄存器压力和大 block size 的权衡"在反向的实现。

`ShuffleLSE` / `ShuffledPsum`（head dim ≤64 时）用 `__shfl_sync` 把 L 和 D 从"某个线程"广播给同 warp，省掉从 smem 读统计量那一跳，因为 $ d \le 64 $ 时一个 warp 够装。

"针对 head dim 挑 tiling 配置"具体到 C++ FA3 的常见取值（非因果）：

| head dim | 反向 tiling | 寄存器/线程 | 为什么 |
| --- | --- | ---: | --- |
| 128 | `tile_m=80, tile_n=128`，`SdP_swap=T, dKV_swap=F, dQ_swap=T` | 208 | `tile_m=80` 不是 64 的倍数，靠 swap 把 M 维换成能整除 64 的那一面 |
| 192 | 3 个 MMA warpgroup，`tile_m=64, tile_n=96` | 128 | 3 WG 才有足够 M 向并行，代价是每线程寄存器被压到 128，`tile_n` 只好缩 |
| 256 | 走 `Slice_dQKV_Mma`（上文） | — | 累加器太大，分片降峰值 |

## 一句话

FA3 反向逐行读下来，数学就是 [[learning/flash-attention/04-backward-kernel|FA2 反向]]，但实现上多了三层：**warp specialization 让 TMA 搬 Q/dO 和 MMA 重叠**、**`tiled_mma_SdP` 一个 tiling 复用算 S 和 dP**、**dQ 的原子（或 TMA reduce）按 `m_block` 累加**。它把 04 那套"重算 + 三个梯度 GEMM"搬到了 Hopper，并针对 head dim 64/128/256 分别挑 tiling 配置。

## Reference

- flash-attention 仓库（hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp、flash_bwd_kernel_sm90.h）：<https://github.com/Dao-AILab/flash-attention>
- SM90 调参笔记（反向 tile/swap/warpgroup 配置、寄存器预算）：<https://github.com/Dao-AILab/flash-attention/blob/main/AI/SM90_BLOCK_SIZE_TUNING.md>
- FlashAttention-3 论文（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- FlashAttention 论文（backward 算法，arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- PTX ISA（WGMMA、TMA、named barrier、atomic）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
