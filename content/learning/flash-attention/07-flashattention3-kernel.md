[[learning/flash-attention/06-flashattention3|FA3 算法篇]] 讲了 warp specialization、pingpong、2 级流水线，这一篇真的逐行读 Hopper 前向内核。文件是 `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` 的 `CollectiveMainloopFwdSm90::mma`（consumer 端），加上 `load`（producer 端）。

先说明：FA3 用的是 Hopper 的两样新家伙。**TMA**（`SM90_TMA_LOAD`）批量异步拷贝，**WGMMA**（`tiled_mma`）异步张量核 GEMM。它俩都是"发出即返回"，所以能靠 `pipeline.consumer_wait / producer_acquire` 这类 barrier 把"搬"和"算"叠起来。

## consumer 端 setup：把 tile 和 warp 组接起来

`mma` 函数开头建立共享内存 tile 和 warp 组切片：

```cpp
Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
Tensor sK = make_tensor(... smem_k ...), SmemLayoutK{});
Tensor sV = make_tensor(... smem_v ...), SmemLayoutVtMma{});
Tensor sP = ...;   // P 的 smem 位置（RS 时复用 sQ，免一块）
```

然后按 warp group 切片：

```cpp
int warp_group_idx = __shfl_sync(..., thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
TiledMmaQK tiled_mma_qk;
TiledMmaPV tiled_mma_pv;
auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
```

- `warp_group_idx`：当前线程属于哪个 warpgroup。Fa3 的 mma（consumer）warpgroup 数由 tiling 决定，常见为 2 个做 pingpong；producer 是独立的：FP16 用 TMA 时是 1 个 warp（`NumProducerThreads = cutlass::NumThreadsPerWarp`），FP8 要转置 V 时是 1 个 warpgroup（`NumThreadsPerWarpGroup`）。
- `tiled_mma_qk` 是 `$ QK^\top $` 的 WGMMA，`tiled_mma_pv` 是 `$ P V $` 的 WGMMA。`tSrQ`（A 片段）、`tSrK`（B 片段）喂 `$ QK^\top $`，`tOrV` + `tOsP` 喂 `$ PV $`。

因为 producer 只发 TMA、几乎不占寄存器，FA3 用 `setmaxnreg` 把寄存器预算从 producer 匀给 consumer 的 MMA warpgroup。Hopper 下来就是 "用几个 MMA warpgroup" 直接决定每线程的寄存器天花板：2 个 warpgroup 时每线程 240 个、扣 24 个固定开销后约 **216 个可用**；3 个时压到 160 个、扣 32 个后约 **128 个可用**。所以多一个 warpgroup 意味着更大的 `tile_m`（[[learning/flash-attention/06-flashattention3|算法篇]] 说的 pingpong 好处），但每线程能用的寄存器变少——这就是"pingpong 和大 block size 都耗寄存器、权衡更难"的落地数字。

## producer：TMA 加载

`load` 函数（producer warpgroup）发 TMA。核心是"填一个 stage，通知 consumer，重复"：

```cpp
auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
    pipeline_k.producer_acquire(smem_pipe_write);
    // TMA 描述符 + mcast + 缓存提示 EVICT_LAST
    copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
         tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
};
```

- `pipeline_k.producer_acquire(smem_pipe_write)`：等这个 stage 有空。
- `params.tma_load_K.with(...)`:这是一个 `SM90_TMA_LOAD` 的拷贝原子，`with` 绑定 barrier 和缓存提示。`EVICT_LAST` 是 TMA 缓存提示：把这块数据标记为「最后淘汰」（尽量留在 L2），配合 `mcast_mask_kv` 的 cluster multicast，让同 cluster 的其他 CTA 复用同一份 K/V。
- 真正执行是 `copy(...)`：TMA 异步拷贝，**立刻返回**。
- `producer_get_barrier` → 这块 load 完成时让 `pipeline_k` 的 barrier 到达，consumer 那边 `consumer_wait` 就醒。

`load_V` 同理。Q 也用 TMA 加载（`Use_TMA_Q`），但 Q 是每个 CTA 只加载一次，所以有个独立的 barrier。

## consumer 主循环：`fwd_step` 是核心

`IntraWGOverlap` 为真时（2 级 GEMM-softmax 流水线），主循环 `fwd_step`（一个 lambda）一次处理一个 `$ n\_block $`，但**同时发起两个 GEMM**：

```cpp
auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
    PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
    ++smem_pipe_read;
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
    if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_k, smem_pipe_read); }
    warp_scheduler_barrier_sync();

    // GEMM0：Q K^T → S（异步 WGMMA，不立即等）
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

    if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
    // GEMM1：P · V → O（这是"上一轮的 P"乘"这一轮的 V"，和 GEMM0 并行）
    if (!HasQv) {
        if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_v, smem_pipe_read_v); }
    }
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, ...(tOrP/tOsP), tOrV(..., smem_pipe_read_v.index()), tOrO);

    warp_scheduler_barrier_arrive();
    warpgroup_wait<1>();                 // 等一下两个 WGMMA 里较晚的那个
    pipeline_k.consumer_release(smem_pipe_read);  // 释放 K，让 producer 复用
    ...
    mask_fn(tSrS, n_block);               // 因果 / local mask（用 `scores` 的 identity 做 predication）

    cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
    softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);   // 在线 softmax：m、l 更新 + 重缩 O
    ...
    convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);       // fp32 S → fp16/bf16 P
    if (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
    if constexpr (!RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }  // 把当前轮要的 rescale 补上
    if (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
};
```

逐行对应的算法（[[learning/flash-attention/06-flashattention3|算法篇]] 的 2 级流水线）：

1. `++smem_pipe_read`：推进 K 的流水线 stage。
2. `consumer_wait(pipeline_k, smem_pipe_read)`：等这块 K 的 TMA 拷贝完成。
3. `gemm(... tiled_mma_qk, tSrQ, tSrK ...)`：**GEMM0**，`$ S = Q K_j^\top $`，`wg_wait=-1` 表示不等，异步发出去。
4. `if RescaleOBeforeGemm: softmax.rescale_o(tOrO, scores_scale)`：把 `$ O $` 按 `scores_scale`（`$ e^{m_{old}-m_{new}} $`）重缩。**这个顺序很关键**：GEMM1 用的是"重缩后的 O"，所以必须放在 GEMM1 之前。
5. `consumer_wait(pipeline_v, smem_pipe_read_v)`：等这块 V 拷贝完成。
6. `gemm(... tiled_mma_pv, tOrP/tOsP, tOrV ...)`：**GEMM1**，`$ O += P_{j-1} \cdot V_{j-1} $`。注意它用的 P 和 V 都**慢半拍**：`tOrP` 是上一轮 softmax 算出的 `$ P_{j-1} $`，`tOrV` 用 `smem_pipe_read_v`（在 `++smem_pipe_read` 之前捕获的那一个阶段，即上一块的 V）。它和 GEMM0 的 `$ Q K_j^\top $` 并行——这就是算法篇说的"GEMM1(j) ∥ GEMM0(j+1)"，错开一块，本轮只算 `$ S_j $`，上一块的 `$ P V $` 挪到本轮。
7. `warpgroup_wait<1>()`：等两个 WGMMA 都算完（GEMM0 和 GEMM1）。
8. `pipeline_k.consumer_release`：释放 K 这个 stage，让 producer 接着填。
9. `mask_fn(tSrS, n_block)`：因果/local 掩码，把越界位置设 `$ -\infty $`。
10. `max_get_scale(...)`：算出这一轮的 rescale 因子（新行最大 vs 旧行最大）。`Check_inf` 处理"整行全是 `$ -inf $`"的边界。
11. `online_softmax(...)`：更新 `$ m $`、`$ \ell $`，并重缩 `$ O $`。
12. `convert_type_out(...)`：`$ S $` 是 fp32 累加器，转成 fp16/bf16 的 `$ P $`，喂给下一个 GEMM1。
13. `write_P_to_smem(tOrP)`：`MmaPV_is_RS` 为假时把 P 写到 smem（SS GEMM）；为真时 `$ P $` 留在寄存器（RS GEMM），省这块 smem。
14. `if !RescaleOBeforeGemm: rescale_o(...)`：两种调度二选一，把 `$ O $` 的重缩要么放 GEMM1 前（`RescaleOBeforeGemm`），要么放 softmax 后。

**为什么这样就能重叠**：GEMM1 用的 `$ P_{j-1} $` 和 `$ V_{j-1} $` 都是上一轮就算好/搬好的；这一轮只算 `$ S_j $`（GEMM0），然后 softmax 出 `$ P_j $`，为下一轮的 GEMM1 做准备。于是"本轮 softmax"和"本轮 GEMM1（用上轮 P、上轮 V）"以及"下轮 GEMM0"在时间上错开，**exp 单元算 `$ P_j $` 的时候，张量核在算 `$ P_{j-1} V_{j-1} $` 和 `$ Q K_{j+1}^\top $`**。

`mask_fn`/`check_inf` 区分了三种循环：因果/local 掩码段（`check_inf=true`）、无掩码段（`check_inf=false`）、local 左掩码段。因果时只对"碰对角"的块做掩码，别的块整块跳过（`n_block_min / n_block_max` 已把区间算好）。

## pingpong 从哪来

`fwd_step` 是**一个 warpgroup 内**的 2 级流水线。pingpong 是**两个 consumer warpgroup 之间**：一个 warpgroup 的 `fwd_step` 跑 `$ Q K_j^\top $` + softmax，另一个跑另一组行块的 `$ P V $`。`warp_scheduler_barrier_sync` / `warp_scheduler_barrier_arrive` 就是 FA3 用来强制两边的 GEMM 交错的 barrier。

## FP8 的额外一脚

`if (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }` 和 `permute_Aregs_fp8(tOrP)`：这就是 [[learning/flash-attention/06-flashattention3|FA3 算法篇]] 说的"FP32 累加器布局和 operand A 布局不同，要 byte-permute"。FP8 时 `$ S $` 从累加器（C 布局）转成 `$ P $`（A 布局）之前，必须重排寄存器里的元素顺序，否则 WGMMA 算错。

- `write_P_to_smem` / `arrive_on_P_write_barrier` 用的是 `FwdNamedBarriers::PEmpty / PFull`，配合 `LargeHeadDimV`（`kHeadDimV > 256`，源码第 61 行）时 `$ P $` 太大放不下寄存器，必须走 smem，所以多一层 named barrier 同步。

## 一句话

FA3 Hopper 前向逐行读下来，就是**"把两个 WGMMA 和一次 softmax 交错成一个 `fwd_step`"**：GEMM0 算 `$ S_j $`（异步），GEMM1 用上轮 `$ P $` 算 `$ P V $`（异步），中间的空档给 softmax 的 exp 用。TMA 在 producer 端持续喂 K/V，`pipeline` barrier 控制 stage 复用。FA3 把一个 warpgroup 都算得"没有一个空等张量核的时刻"。

## Reference

- flash-attention 仓库（hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp、flash_fwd_kernel_sm90.h）：<https://github.com/Dao-AILab/flash-attention>
- SM90 调参笔记（`setmaxnreg` 寄存器预算、warpgroup 数、tile 形状）：<https://github.com/Dao-AILab/flash-attention/blob/main/AI/SM90_BLOCK_SIZE_TUNING.md>
- FlashAttention-3 论文（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- NVIDIA CUTLASS（SM90 pipeline / TMA / GMMA）：<https://github.com/NVIDIA/cutlass>
- PTX ISA（WGMMA、TMA、named barrier）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
