[[learning/flash-attention/05-backward-kernel|FA1/FA2 反向]] 讲的是数学和 Ampere 内核，这一篇读 **FA3 的 Hopper 反向内核** `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` 的 `CollectiveMainloopBwdSm90`。数学和 [[learning/flash-attention/05-backward-kernel|05]] 一模一样（还是 $ P = e^{S-L} $、$ dS = P \circ (dP - D) $、dQ/dK/dV 三个 GEMM），但 FA3 把它塞进了 warp specialization，并且**复用同一个 MMA tiling 同时算 S 和 dP**。

## 与 FA2 反向的差别

FA3 反向的"算子"没变，变的是**怎么调度**：

| | FA2（[[learning/flash-attention/05-backward-kernel|05]]） | FA3 反向 |
| --- | --- | --- |
| S 与 dP | 两个独立 MMA tiling | `tiled_mma_SdP`（一个 tiling 复用，`SwapAB` 控制哪个当 A） |
| L / D | 从 smem 读 | 小 head dim 用 `__shfl_sync`（`ShuffleLSE`/`ShuffledPsum`） |
| 调度 | 无 warp specialization | producer warpgroup（TMA 搬 Q/K/V/dO + 一个 warp 做 dQ 归约写回）+ consumer（`bwd_step`） |
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
- **`tiled_mma_SdP` 同时算 S 和 dP**：$ S = QK^\top $ 的输出形状是 $ B_r \times B_c $，$ dP = dO\,V^\top $ 也是 $ B_r \times B_c $，两者的 operand 长宽正好互补（A 是 $ B_r \times d $ 的 Q/dO，B 是 $ B_c \times d $ 的 K/V）。所以同一份 WGMMA tiling（`AtomLayoutSdP` + `TileShapeAtomSdP`）配 `SwapAB` 就能吃下两个 GEMM，省掉一整套 MMA 模板和 atom layout。源码里 `TiledMmadP` 默认就等于 `TiledMmaSdP`（只在 `Mma_dP_is_RS` 这种特殊配置下才换成 `TiledMmadPRS`，因为那时 dP 的 A 操作数（dO）来自寄存器而不是 smem）。
- `wg_wait=-1` 是异步发出，不等；紧接着的 `warpgroup_wait<1>()` 意味着**只等先发的 $ S = QK^\top $**，把后发的 $ dP = dO V^\top $ 继续挂在队列里——下一个环节正是要用 $ S $ 的 softmax，见下。
- `tLSErLSE` 读的是 forward 存的 **logsumexp $ L $**（FA2 微调 2 的产物），后面直接 $ P = e^{S - L} $。

## P = exp(S − L)，再算 dS

```cpp
if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }   // softcap（如果启用）
Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
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

- `exp2f(scores * scale_log2 - lse_scaled)`：$ e^{S - L} = P $。用 `exp2f` 而不是 `expf`，因为 $ e^x = 2^{x\log_2 e} $ 能把"乘 $ \log_2 e $、减 $ L\log_2 e $"合成一条 `ffma`（源码注释原话：allow the compiler to use the ffma instruction instead of fadd and fmul separately）；FA2/FA1 的 forward 里 `softmax_rescale_o` 与 FA3 的 `flash::scale_apply_exp2` 都是这个写法。注意 `lse_scaled` 里已经带上了 $ \log_2 e $，即它存的是 $ L \log_2 e $。
- $ dS = P \circ (dP - D) $：和 [[learning/flash-attention/05-backward-kernel|FA2 反向]] 的 `pointwise_mult` 一模一样。`dP_sum_cur` 是 $ D = \mathrm{rowsum}(dO \circ O) $。
- softcap 时 `dS *= dtanh`：softcap 的链式法则多一项 $ 1 - \tanh^2 $。`calculate_dtanh(scores)` 必须**在 mask 之前**算：mask 会把越界元素写成 $ -\infty $，之后再算 $ 1 - \tanh^2(-\infty) $ 就成了 NaN（源码里专门留了这条注释）。

## 反向也有一层 GEMM-softmax 重叠

前向的 2 级流水线是"把 GEMM1 挪到上一块"，反向没有 $ P V $ 那种结构，但它有另一处可藏：**$ dP = dO V^\top $ 可以藏在对 $ S $ 的 softmax 后面**。看代码顺序：

```text
发 GEMM-S：S = Q K^T          （wg_wait=-1）
发 GEMM-dP：dP = dO V^T       （wg_wait=-1）
warpgroup_wait<1>()           ← 只等 S；dP 还在张量核上跑
对 S 做 mask、算 exp2f(S - L) → P   ← 这段时间和张量核上的 dP 重叠
warpgroup_wait<0>()           ← 现在再等 dP
dS = P ∘ (dP - D)
```

这不是巧合，而是必须的：$ dS $ 的计算要同时用到 $ P $（来自 $ S $）和 $ dP $，两边都得等；但 `exp2f` 那一整圈是纯 MUFU / CUDA core 的活，正好插在两段 GEMM 之间。`warpgroup_wait<1>` 与 `warpgroup_wait<0>` 的前后位置就是把这条依赖切成"先等 S、后等 dP"的手段。

这也能解释为什么反向的寄存器比前向更紧：$ S $ 与 $ dP $ 两个 fp32 累加器必须**同时**活着（$ S $ 在算 exp 时，$ dP $ 还在被写入），之后 $ dQ $ 的累加器才能复用它们腾出来的寄存器。SM90 调参笔记里给反向的峰值估算是 `max(2 * regs_SdP, regs_dQ) + regs_dK + regs_dV`。

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

- `convert_type_out` 把 fp32 的 $ P $ 和 $ dS $ 转成 fp16/bf16，喂给后面的 MMA（`rP`/`rdS` 是 fp16 的副本，`tdPrdP` 那个 fp32 片段还要留着当 $ dS $ 的累加空间用）。
- 三个 GEMM 的顺序是 **$ dV \mathrel{+}= P^\top dO $**（`tiled_mma_dKV`，`wg_wait=-1`）、**$ dQ \mathrel{+}= dS\,K $**（`tiled_mma_dQ`）、**$ dK \mathrel{+}= dS^\top Q $**（`tiled_mma_dKV`）。注意 $ dV $ 和 $ dK $ 共用**同一个 tiling**，靠 `SwapAB` 决定把 $ P^\top/dS^\top $ 还是另一个操作数当 A——因为 $ dV = P^\top dO $ 和 $ dK = dS^\top Q $ 的输出形状相同（都是 $ B_c \times d $），只是"谁乘谁"不同。`dKV_swapAB` / `dQ_swapAB` 是编译期常量，不是运行时分支。
- `wg_wait=1` 的含义要按 `flash::gemm` 的定义读：commit 之后**等到未完成的 WGMMA 只剩 1 组**。此刻队列里有两组，所以它是"等**先发出的那一组**做完"。这里有两处硬需求：(a) $ dV $ 读的是 smem 里的 dO，紧跟其后的 `pipeline_do.consumer_release` 要把 dO 还给 producer，所以 $ dQ $ 那行的 `wg_wait=1` 实际保证了 $ dV $ 已经读完 dO；(b) $ dQ $ 的累加器 `tdQrdQ` 紧接着要被写到 smem（TMA 路径）或直接 `atomicAdd`，所以 $ dK $ 那行的 `wg_wait=1` 保证 $ dQ $ 已经算完。最后还有一个 `warpgroup_wait<0>()`，它等的是 $ dK $——因为 $ dK $ 读的是 $ Q^\top $，之后才能 `pipeline_q.consumer_release`。

## dQ 的原子归约（或 TMA reduce）

$ dQ $ 需要在一个 `m_block` 的范围内累加（按 [[learning/flash-attention/05-backward-kernel|FA2 反向]] 的说法，反向按列块并行，$ dQ $ 靠原子加合并）：

```cpp
if constexpr (dQacc_use_TMA) {           // head dim < 256
    // 先和 producer 侧同步（dQEmptyWG1：这块 sdQ 已被 TMA 清空）
    cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);   // 累加器 → smem
    cutlass::arch::fence_view_async_shared();
    // 再通知 dQFullWG1：sdQ 满了，producer 可以发 bulk reduce
    ...
} else {                                  // head dim >= 256
    Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
    Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
    #pragma unroll
    for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
}
```

- `dQacc_use_TMA = (kHeadDim < 256)`：head dim 小、$ dQ $ 一块装得下，走 smem 中转：MMA warpgroup 把 $ dQ $ 累加器写进 smem（`r2s_tiled_copy_dQaccum`），producer warpgroup 里的另一个 warp 跑 `store_dq`，用 `cp.reduce.async.bulk`...`add.f32` 把整块**并行归约加**到全局 dQaccum（`hopper/copy_sm90_bulk_reduce.hpp` 的 `SM90_BULK_REDUCE_ADD` 就是这条指令；`flash_bwd_kernel_sm90.h` 里 WG0 的 warp 0 负责搬 Q/K/V/dO，warp 1 负责这一步）。两边用 `BwdNamedBarriers::dQEmptyWG1 / dQFullWG1` 交接，WG 号拼在 barrier id 里，所以每个 MMA warpgroup 各有一对。
- head dim ≥ 256 时 smem 预算不够（smem 里还要放 Q 的双缓冲、K/V/dO/P/dS，而 dQaccum 自己是 $ B_r \times d \times 4 $ 字节；源码在 `!dQacc_use_TMA` 时干脆把 `SmemdQacc_t` 定义成空数组），于是退回逐元素 `atomicAdd`：`recast<float4>` 把 4 个 fp32 打成一个 128-bit 原子，原子操作次数降到 1/4。
- `tdQgdQaccum(_, _, _, m_block)`：给当前 `m_block` 的那块 $ dQ $ 累加器加。这就是 FA3 反向的"列块并行，$ dQ $ 用原子合并"。

## hdim 256：Slice_dQKV_Mma

`Slice_dQKV_Mma` 分支只在 `kHeadDim == 256 && !dQacc_use_TMA && dQ_swapAB && AtomLayoutMdQ==1 && NumMmaWarpGroups==2` 时走。它把 `dQ` / `dKV` 的 MMA 按 `M_slice` 拆成两半（`M_slice=0/1`），中间穿插写 smem，**降低寄存器峰值**（head dim 256 时累加器太大，不分片会 spill）。这就是 [[learning/flash-attention/07-flashattention3|FA3 算法篇]] 说的"寄存器压力和大 block size 的权衡"在反向的实现。

`ShuffleLSE` / `ShuffledPsum`（源码条件是 `SdP_swapAB && kHeadDim <= 64`）解决的是统计量的寄存器占用：`SdP_swapAB` 时每个线程要维护 $ B_r/4 $ 行的统计量（源码注释原话），$ L $ 和 $ D $ 各存一份就很吃寄存器。于是改成让共享同一批行的 8 个线程**分摊**——每人只留 `kStatsPerThread = ceil_div(size(tLSEsLSE), 8)` 个值，用的时候 `__shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4))` 从该行的"所有者"线程广播过来。代价是每条统计量多一次 shuffle，收益是省下寄存器。顺带一个细节：`SmemLayoutLSE` 的第二个 stride 被 `round_up(kBlockM, 64)` 撑大，这样个别线程越界读 sLSE 也落在合法 smem 地址上（源码注释写明这是故意的）。

"针对 head dim 挑 tiling 配置"具体到 C++ FA3 的常见取值（非因果）：

| head dim | 反向 tiling | 寄存器/线程 | 为什么 |
| --- | --- | ---: | --- |
| 128 | `tile_m=80, tile_n=128`，`SdP_swap=T, dKV_swap=F, dQ_swap=T`，`aSdP=1, adKV=2`，`mma_dkv_is_rs=True` | 208 | `tile_m=80` 不是 64 的倍数，靠 swap 把 M 维换成能整除 64 的那一面；RS 让 $ P $/$ dS $ 留在寄存器，省掉写入 smem 的那一趟 |
| 192 | 3 个 MMA warpgroup，`tile_m=64, tile_n=96`，`SdP_swap=F, dKV_swap=T` | 128 | 3 WG 才有足够 M 向并行，代价是每线程寄存器被压到 128，`tile_n` 只好缩（这是 hdim=192 下唯一可行的 `tile_n > 64` 配置） |
| 256 | 走 `Slice_dQKV_Mma`（上文） | — | 累加器太大，分片降峰值 |

## 一句话

FA3 反向逐行读下来，数学就是 [[learning/flash-attention/05-backward-kernel|FA2 反向]]，但实现上多了几层：**warp specialization 让 TMA 搬 Q/dO 和 MMA 重叠**、**`tiled_mma_SdP` 一个 tiling 复用算 S 和 dP**、**$ dP $ 的 GEMM 被塞进 $ S $ 那圈 exp2f 后面**、**dQ 的原子（或 TMA reduce）按 `m_block` 累加**。它把 [[learning/flash-attention/05-backward-kernel|05]] 那套"重算 + 三个梯度 GEMM"搬到了 Hopper，并针对 head dim 64/128/192/256 分别挑 tiling + swap 配置。

## Reference

- flash-attention 仓库（hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp、hopper/flash_bwd_kernel_sm90.h、hopper/utils.h、hopper/mask.h）：<https://github.com/Dao-AILab/flash-attention>
- SM90 调参笔记（反向 tile/swap/warpgroup 配置、寄存器预算）：<https://github.com/Dao-AILab/flash-attention/blob/main/AI/SM90_BLOCK_SIZE_TUNING.md>
- FlashAttention-3 论文（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- FlashAttention 论文（backward 算法，arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- PTX ISA（WGMMA、TMA、named barrier、atomic）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
