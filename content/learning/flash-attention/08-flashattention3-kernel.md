[[learning/flash-attention/07-flashattention3|FA3 算法篇]] 讲了 warp specialization、pingpong、2 级流水线，这一篇真的逐行读 Hopper 前向内核。文件是 `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` 的 `CollectiveMainloopFwdSm90::mma`（consumer 端），加上 `load`（producer 端）。

先说明：FA3 用的是 Hopper 的两样新家伙。**TMA** 批量异步拷贝（Q 用 `cute::SM90_TMA_LOAD`；K/V 的拷贝原子由 cluster 形状决定——cluster 只有 1 个 CTA 时是 `SM90_TMA_LOAD`，多 CTA 时是 `SM90_TMA_LOAD_MULTICAST`，见 `sm90_cluster_shape_to_tma_atom`），**WGMMA** 异步张量核 GEMM（由 `tiled_mma_qk/pv` 上的 `GMMA::ss_op_selector` 选的 atom——$ PV $ 在 RS 配置下换成 `rs_op_selector`，因为那时 A 操作数来自寄存器而不是 smem）。它俩都是"发出即返回"，所以能靠 `pipeline.consumer_wait / producer_acquire` 这类 barrier 把"搬"和"算"叠起来。这里所有 MMA 都走同一个封装 `flash::gemm<zero_init, wg_wait, SwapAB>`：它以 `warpgroup_arrive()` 开始、以 `warpgroup_commit_batch()` 结束——**每次调用自成一个 WGMMA commit group**；`wg_wait >= 0` 时在 commit 后接一个 `warpgroup_wait<wg_wait>()`，而 `wg_wait = -1` 表示发出去就走。下面看到的"GEMM 不等"和"等第 N 个"全都由此而来。

## consumer 端 setup：把 tile 和 warp 组接起来

`mma` 函数开头建立共享内存 tile 和 warp 组切片：

```cpp
Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
Tensor sK = make_tensor(... smem_k ...), SmemLayoutK{});
Tensor sV = make_tensor(... smem_v ...), SmemLayoutVtMma{});
Tensor sP = ...;   // P 的 smem 位置（RS 时借用 smem_q 当占位，实际不写）
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

- `warp_group_idx`：当前线程属于哪个 warpgroup。FA3 的 mma（consumer）warpgroup 数由 tiling 决定，`tile_m = 128` 时是 2 个（正好做 pingpong）；producer 是独立的：FP16 走 TMA 时只需要 1 个 warp（`NumProducerThreads = cutlass::NumThreadsPerWarp`，因为 TMA 指令单线程就能发起），FP8 要在 kernel 内转置 V 时是 1 个 warpgroup（`NumThreadsPerWarpGroup`，LDSM/STSM 是一个 warp 协作的指令，转置还要配合寄存器搬运）。
- `tiled_mma_qk` 是 $ QK^\top $ 的 WGMMA，`tiled_mma_pv` 是 $ P V $ 的 WGMMA。`tSrQ`（A 片段）、`tSrK`（B 片段）喂 $ QK^\top $，`tOrV` + `tOsP` 喂 $ PV $；注意 $ PV $ 的 A 操作数由 `cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP)` 二选一——RS 时 $ P $ 直接用寄存器里的 `tOrP`，SS 时才用 smem 的 `tOsP`。

因为 producer 只发 TMA、几乎不占寄存器，FA3 用 `setmaxnreg` 把寄存器预算从 producer 匀给 consumer 的 MMA warpgroup（源码里是 `cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>()` 和 `warpgroup_reg_alloc<MmaRegisterRequirement>()`，前向取 `Load=24、Mma=240`（2 个 MMA warpgroup + TMA 时）、`Load=32、Mma=160`（3 个））。Hopper 上一个 CTA 的寄存器总量是固定的，所以"用几个 MMA warpgroup"直接决定每线程的寄存器天花板：2 个 warpgroup 时每线程 240 个、扣 24 个固定开销后约 **216 个可用**；3 个时压到 160 个、扣 32 个后约 **128 个可用**（这两个净数来自官方 SM90 调参笔记）。所以多一个 warpgroup 意味着更大的 `tile_m`（[[learning/flash-attention/07-flashattention3|算法篇]] 说的 pingpong 好处），但每线程能用的寄存器变少——这就是"pingpong 和大 block size 都耗寄存器、权衡更难"的落地数字。

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
- `params.tma_load_K.with(...)`：一个 TMA 拷贝原子 + 属性，`with` 把 **completion barrier**（`producer_get_barrier`）、cluster 广播掩码、缓存提示一起绑上。`EVICT_LAST` 是 TMA 的 L2 淘汰优先级提示：这块数据标记为"最后淘汰"，尽量留在 L2；`mcast_mask_kv` 只在 `SM90_TMA_LOAD_MULTICAST` 时非零，作用是让 cluster 内需要同一份 K/V 的 CTA 从一次 TMA 里各取一份。
- 真正执行是 `copy(...)`：TMA 异步拷贝，**立刻返回**。
- `producer_get_barrier` → 这块 load 完成时让 `pipeline_k` 的 barrier 到达，consumer 那边 `consumer_wait` 就醒。

注意这里**没有显式的 `producer_commit`**：TMA 路径下 `producer_acquire` 已经把这次拷贝的字节数（expect-tx）登记到 mbarrier 上，由 TMA 引擎拷完以后自己把 barrier 打到完成态；`producer_commit` 只出现在 `PagedKVNonTMA` 的 `cp.async` 分支（`pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive)`），因为 `cp.async` 需要手动 `cp.async.mbarrier.arrive` 才会有这个信号。这也解释了论文 Algorithm 1 里为什么 producer 的"commit"看起来是免费的一步。

`load_V` 同理。Q 也用 TMA 加载（`Use_TMA_Q`），但 Q 是每个 CTA 只加载一次、不进 K/V 那条循环流水线，所以有自己的 completion barrier（`barrier_Q`），缓存提示是 `EVICT_FIRST`（除非是 split 场景需要复用，才改成 `EVICT_LAST`）。

## consumer 主循环：`fwd_step` 是核心

`IntraWGOverlap` 为真时（2 级 GEMM-softmax 流水线），主循环 `fwd_step`（一个 lambda）一次处理一个 $ n\_block $，但**同时发起两个 GEMM**：

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
    // GEMM1：O += P_prev · V_prev（上一轮的 P 乘上一块的 V，和 GEMM0 并行）
    if (!HasQv) {
        if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_v, smem_pipe_read_v); }
    }
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, ...(tOrP/tOsP), tOrV(..., smem_pipe_read_v.index()), tOrO);

    warp_scheduler_barrier_arrive();
    warpgroup_wait<1>();                 // 手上只留 1 个未完成的 WGMMA → 先发的 GEMM0 已经算完
    pipeline_k.consumer_release(smem_pipe_read);  // GEMM0 读完 K 了，释放这个 stage
    ...
    mask_fn(tSrS, n_block);               // 因果 / local mask：越界位置写 -INFINITY

    cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
    softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);   // e^{S-m} 写回 S，并更新行和 l
    ...
    convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);       // fp32 S → fp16/bf16 P
    if (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
    if constexpr (!RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }  // 把当前轮要的 rescale 补上
    if (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
};
```

逐行对应的算法（[[learning/flash-attention/07-flashattention3|算法篇]] 的 2 级流水线）：

1. `++smem_pipe_read`：推进 K 的流水线 stage。
2. `consumer_wait(pipeline_k, smem_pipe_read)`：等这块 K 的 TMA 拷贝完成。
3. `gemm(... tiled_mma_qk, tSrQ, tSrK ...)`：**GEMM0**，$ S = Q K_j^\top $，`wg_wait=-1` 表示不等，异步发出去。
4. `if RescaleOBeforeGemm: softmax.rescale_o(tOrO, scores_scale)`：把 $ O $ 按 `scores_scale`（$ e^{m_{old}-m_{new}} $）重缩。**这个顺序很关键**：GEMM1 用的是"重缩后的 O"，所以必须放在 GEMM1 之前。
5. `consumer_wait(pipeline_v, smem_pipe_read_v)`：等这块 V 拷贝完成。
6. `gemm(... tiled_mma_pv, tOrP/tOsP, tOrV ...)`：**GEMM1**，$ O \mathrel{+}= P_{j-1} V_{j-1} $。注意它用的 P 和 V 都**慢半拍**：`tOrP` 是上一轮 softmax 算出的 $ P_{j-1} $，`tOrV` 用 `smem_pipe_read_v`（在 `++smem_pipe_read` 之前捕获的那个阶段，即上一块的 V）。所以它和本轮 GEMM0 的 $ Q K_j^\top $ 之间**没有任何数据依赖**，两个 GEMM 可以一起排在 tensor core 队列里——这就是算法篇的 2 级流水线：本轮只算 $ S_j $，上一块的 $ P V $ 挪到本轮来算。
7. `warpgroup_wait<1>()`：此刻有两个未完成的 WGMMA（GEMM0、GEMM1），`wait<1>` 等到手上只剩一个——也就是**先发出去的 GEMM0 已经算完**，所以下一步能安全释放 K。GEMM1 这时还在跑，正好被下面的 softmax 盖住；它的收尾在 `online_softmax` 之后（`warpgroup_wait<0>()`，然后才 `consumer_release` 掉那块 V）。
8. `pipeline_k.consumer_release`：释放 K 这个 stage，让 producer 接着填。
9. `mask_fn(tSrS, n_block)`：因果/local 掩码。`flash::Mask` 先用 `cute::make_identity_tensor` 造一张 $ (m, n) $ 坐标表、按 MMA 的 C 布局切给每个线程，于是每个元素的行列号在编译期就已知；再按 `row_idx`、`seqlen_k` 等算出这一行允许的列上界，把越界元素写成 $ -\text{INFINITY} $（不是写 0——写 $ -\infty $ 才能让随后的 exp 得到 0，同时不把行 max 拉低）。
10. `max_get_scale(...)`：算出这一轮的 rescale 因子（新行最大 vs 旧行最大）。`Check_inf` 处理"整行全是 $ -inf $"的边界。
11. `online_softmax(...)`：对 $ S $ 做 $ e^{S - m} $（`flash::scale_apply_exp2`，用 `exp2f` 合成 `ffma`），并把结果累加进行和 `row_sum`（$ \ell $）。注意它**不管 $ O $ 的重缩**——`max_get_scale` 已经把旧 `row_sum` 乘过 rescale 因子，而 $ O $ 的重缩由第 14 步的 `rescale_o` 单独负责（FA2 的 `softmax_rescale_o` 是把这三件事揉在一起的，FA3 拆开了）。
12. `convert_type_out(...)`：$ S $ 是 fp32 累加器，转成 fp16/bf16 的 $ P $，喂给下一个 GEMM1。
13. `write_P_to_smem(tOrP)`：`MmaPV_is_RS` 为假时把 P 写到 smem（SS GEMM）；为真时 $ P $ 留在寄存器（RS GEMM），省这块 smem。
14. `if !RescaleOBeforeGemm: rescale_o(...)`：两种调度二选一，把 $ O $ 的重缩要么放 GEMM1 前（`RescaleOBeforeGemm`），要么放 softmax 后。

**为什么这样就能重叠**：GEMM1 要的 $ P_{j-1} $ 和 $ V_{j-1} $ 都是上一轮就算好 / 搬好的，所以它和本轮 GEMM0 的 $ Q K_j^\top $ 之间没有依赖，两个 WGMMA 会先后进入 tensor core 队列；`warpgroup_wait<1>` 只等 GEMM0，于是接下来这段 softmax（读 $ S_j $、算 $ e^{S - m} $、重缩 $ O $）**跑在 GEMM1 还在算的时候**——exp 单元和 tensor core 同时忙。softmax 出 $ P_j $ 之后，它又成为下一轮 GEMM1 的输入，就这样一块一块错位下去。再叠上 pingpong，另一个 warpgroup 的 GEMM 也会落在同一时间窗口里，整个 CTA 的 tensor core 基本不空。

`mask_fn`/`check_inf` 区分了三种循环：因果/local 掩码段（`check_inf=true`）、无掩码段（`check_inf=false`）、local 左掩码段。因果时只对"碰对角"的块做掩码，别的块整块跳过（`n_block_min / n_block_max` 已把区间算好）。

## pingpong 从哪来

`fwd_step` 是**一个 warpgroup 内**的 2 级流水线。pingpong 是**两个 consumer warpgroup 之间**的错相位，两件事共用同一份 `fwd_step` 代码：

- **分工**：`AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>` 把 tile 的 M 维切给各 WG——`kBlockM = 128` 时 WG0 管第 0–63 行、WG1 管第 64–127 行，索引 head 与 $ K/V $ 完全相同（`tSrQ` 由 `wg_mma_qk` 切出来）。所以两个 WG 跑的是同一个 $ Q_i $ 的行块，不是不同的 CTA 任务。
- **错相位**：如果两边各跑各的，它们的 softmax 会撞到同一个 MUFU 上。`warp_scheduler_barrier_sync()` / `warp_scheduler_barrier_arrive()` 就是用来强制次序的：`arrive` 把"我这个 WG 的 GEMM 都发出去了"通知**下一个** WG（两 WG 时 `next_WG = 1 - cur_WG`，三 WG 时环形传递），`sync` 则在 `WarpSchedulerWG1/WG2/WG3` 这组 named barrier 上等（每个 WG 用自己的 id，由上一个 WG 的 `arrive` 唤醒）。于是 WG0 的 GEMM0+GEMM1 先发，WG1 的 GEMM 排在后面——WG0 落到 softmax 时，WG1 正好在跑 GEMM。
- **开关**：`UseSchedulerBarrier` 的条件是——开了 2 级流水线（`IntraWGOverlap`）时要求 `NumMmaWarpGroups >= 2`，且 FP16 要 `kHeadDim <= 128`、FP8 要 `kHeadDim >= 128`；没开 2 级流水线时要求正好 2 个 warpgroup；两种情况都排除 `LargeHeadDimV`。不满足时 `sync`/`arrive` 在 `if constexpr (UseSchedulerBarrier)` 下退化成空操作，各 WG 各跑各的。

注意这个 barrier 只约束**指令发出顺序**，不搬运数据：GEMM 是异步的，softmax 也不读另一个 WG 的东西。它唯一的作用就是让两个 WG 的执行相位错开一个块，把 MUFU 和 tensor core 的占用时间错开。

## FP8 的额外一脚

FP8 的两条分支各有一处寄存器重排（`hopper/utils.h`）：`if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }` 和 `if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }`。

- `!V_colmajor`（也就是 FA3 走的"kernel 内用 LDSM/STSM 转置 V"那条路）：$ S $ 还在累加器里，就要先在累加器片段上做重排（`permute_Cregs_fp8` 把片段 recast 成 `uint2` 再成对交换），效果就是论文描述的 `{d0 d1 d4 d5 d2 d3 d6 d7}`；换完才能当 $ P $ 用。
- `V_colmajor`（V 已经在全局内存里按列主序排好，不需要 kernel 内转置）：累加器不用动，改成在 $ P $ 的 A 片段上调 `permute_Aregs_fp8`。

这就是 [[learning/flash-attention/07-flashattention3|FA3 算法篇]] 说的"FP32 累加器布局和 operand A 布局不同，要 byte-permute"：论文给的重排顺序是 `d0 d1 d4 d5 d2 d3 d6 d7`（每 8 字节重复一次），并且要求 kernel 内转置 V 时写出对应的行置换，两边合起来才能让第二个 WGMMA 算出正确的输出。不做这一步，WGMMA 会安静地算错。

- `write_P_to_smem` / `arrive_on_P_write_barrier` 用的是 `FwdNamedBarriers::PEmpty / PFull` 这对 named barrier：`LargeHeadDimV`（`kHeadDimV > 256`，源码里 `static constexpr bool LargeHeadDimV = kHeadDimV > 256;`）时 $ O $ 的累加器已经把寄存器吃掉太多，源码里直接 `static_assert(!LargeHeadDimV || !MmaPV_is_RS)` 强制 $ P $ 走 smem 的 SS GEMM，于是 P 的"读完/可写"要多一层同步；同时 O 的重缩系数也要经 `sScale` 这块 smem 在 WG 之间传（`store_scales`），不再只是寄存器里的事。

## 一句话

FA3 Hopper 前向逐行读下来，就是**"把两个 WGMMA 和一次 softmax 交错成一个 `fwd_step`"**：GEMM0 算 $ S_j $（异步），GEMM1 用上轮 $ P $ 算 $ P V $（异步），中间的空档给 softmax 的 exp 用。TMA 在 producer 端持续喂 K/V，`pipeline` barrier 控制 stage 复用。FA3 把一个 warpgroup 都算得"没有一个空等张量核的时刻"。

## Reference

- flash-attention 仓库（hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp、hopper/flash_fwd_kernel_sm90.h、hopper/utils.h、hopper/softmax.h、hopper/mask.h）：<https://github.com/Dao-AILab/flash-attention>
- SM90 调参笔记（`setmaxnreg` 寄存器预算、warpgroup 数、tile 形状）：<https://github.com/Dao-AILab/flash-attention/blob/main/AI/SM90_BLOCK_SIZE_TUNING.md>
- FlashAttention-3 论文（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- NVIDIA CUTLASS（SM90 pipeline / TMA / GMMA）：<https://github.com/NVIDIA/cutlass>
- PTX ISA（WGMMA、TMA、named barrier）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
