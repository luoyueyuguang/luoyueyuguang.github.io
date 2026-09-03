FA4 的反向比 forward 更值钱，因为 [[learning/flash-attention/08-flashattention4|算法篇]] 里提到的"2-CTA MMA、DSMEM 换 dS、确定性归约"全在这个 kernel 里。它在 `flash_attn/cute/flash_bwd_sm100.py`，类 `FlashAttentionBackwardSm100`。

上一轮我读的是 [[learning/flash-attention/09-flashattention4-kernel|前向]]，这轮补反向。反向的核心不是"重算 S / 算梯度"，FA3 早就做了；FA4 的新东西是**怎么在 Blackwell 上把 5 个 MMA 和非 MMA 操作重叠，以及怎么把 dQ 的全局原子归约减半再做成确定性的**。

## 5 个 MMA + TMEM 共享

反向每轮做 5 个 MMA（`$ S^\top = KQ^\top $`、`$ dP^\top = V\,dO^\top $`、`$ dQ = dS\,K $`、`$ dV = P^\top dO $`、`$ dK = dS^\top Q $`）。其中 3 个是 SS（双操作数读 smem），2 个是 TS（A 从 TMEM，B 从 smem）。

TMEM 只够放 4 个 128×128 累加器 tile，所以必须共享：

```python
self.tmem_dS_offset = self.tmem_dP_offset    # dS 与 dP 共用一块 TMEM
```

这正是 [[learning/flash-attention/08-flashattention4|算法篇]] 说的：`$ S $` 和 `$ P $` 共用一个 TMEM 块，`$ dP, dS, dQ $` 共用另一个，`$ dV, dK $` 各自占剩下的不能共享。4 个 tile 摆满。

## 流水线：拿上一轮的 dQ/dK MMA 垫 softmax

FA3 反向里，softmax 只和 `$ dP $` 的 MMA 重叠。但 [[learning/flash-attention/06-flashattention3|FA3 篇]] 说过 Blackwell 上 MMA 必须至少两个并发才喂得饱。所以 FA4 的 `compute_loop`（2883 行起）让**上一轮的 `$ dQ $` 和 `$ dK $` 两个 MMA** 和当前轮的 softmax 重叠。

```python
# compute_loop 里（伪代码化）：
#   这一轮算完 dS 之后，把当前 block 的结果倒进 TMEM；
#   紧接着发上一轮就绪的 dQ、dK MMA；然后再做本轮的 softmax 计算
```

代价是要精细管理 TMEM 和 smem 在"加载 / MMA / 计算 / 归约"四类操作之间的复用，因为 dS 写 smem 和 TMEM 的时机对得上才能重叠。

## 2-CTA MMA：dQ 的归约减半

这一节是 FA4 反向最独特的地方。普通 CTA 里，`$ dQ $` 的归约轴是 `$ N $`（KV 维），每个 CTA 都要对整段 KV 做原子 add。FA4 用 Blackwell 的 **2-CTA MMA**（`tcgen05.CtaGroup.TWO`）：

```python
self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
self.cluster_shape_mn = (cluster_size, 1)     # 2 个 CTA 一组
self.Q_stage = 1 if self.use_2cta_instrs else 2
```

2-CTA MMA 把输出累加器在 **M 维**切成两半，两个 CTA 当一个 `$ M=256 $` 的 tile：每个 CTA 只加载并暂存**一半的 operand B**，只保留自己的累加器切片。

**dQ 的归约轴冲突。** dQ MMA 的结构是 `$ dS\,K $`，归约方向是 `$ N $`。2-CTA MMA 只切输出（M 维），不切归约轴；但每个 CTA 又要对自己那部分行做完整归约。FA4 用 **DSMEM**（distributed shared memory）把一半的 `$ dS $` tile 塞给对侧 CTA，让每个 CTA 凑成一个 `$ (M/2 \times 2N) $` 的 operand，去跑 **CTA-pair UMMA（加倍的归约）**。这样整个 `$ N $` 的归约在 CTA pair 内并行完成。

**另一半收益：dQ 的全局原子减半。** 2-CTA 下每个 CTA 只写一半 `$ dQ $` tile，到达全局的原子归约次数减半。这在 `dQacc_reduce` 里对应：

```python
stage_offset = (
    expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0
)
```

`cta_rank_in_cluster` 在 2-CTA 下给每个 CTA 一个偏移：CTA 0 写 stage 0、1，CTA 1 写 stage 2、3，各写 `$ dQ $` 的一半。最终写到全局用：

```python
copy_utils.cpasync_reduce_bulk_add_f32(
    sdQaccum[None, smem_idx].iterator,
    gdQaccum_cur[None, stage + stage_offset].iterator,   # 落到各自的半 tile
    self.tma_copy_bytes["dQ"] // 1,
)
```

`cp.async.bulk` 的 reduce-add 版本，把 smem 里的 partial 原子加到全局。

## 确定性归约：信号量锁

`$ dQ $`（GQA 时还有 `$ dK,dV $`）的全局归约是**非确定性**的：多个 CTA 的 `cp.async.bulk reduce` 到达顺序不定。FA4 的确定性模式用**信号量锁**串行化：

```python
# 获取：等信号量等于 lock_value（即所有"更早的写者"都完成）
barrier.wait_eq(
    mdQ_semaphore_cur[(m_block, None)].iterator,
    tidx, cta_rank_in_cluster, lock_value,
)
...
# 释放：给上一块的信号量 +1；arrive_inc 会发 red_release（内部带 membar）
if const_expr(self.deterministic and stage == 0 and delay_semaphore_release):
    if m_block > m_block_min:
        barrier.arrive_inc(
            mdQ_semaphore_cur[(m_block - 1, None)].iterator,
            tidx, cta_rank_in_cluster, 1,
        )
```

- `lock_value` 由 `_dq_semaphore_lock_value` 决定。**普通模式**：`lock_value = n_block`。**SPT（shortest-processing-time-first）模式**：`lock_value = n_block_max_for_m_block - 1 - n_block`，让"最长的块"先写，减少 CTA 等的次数。
- `wait_eq` = acquire：等到这个 `m_block` 的信号量等于 `lock_value`，也就是按**预定顺序**排队。
- `arrive_inc` = release：写完后给上一块的信号量 +1，允许下一块写。注释明确写了"`arrive_inc` 调用 `red_release` 会发 `membar`"。这就是论文说的"设备级可见性栅栏"的性能代价。

**为什么有代价。** 确定性模式要 (1) 每个写之后发 `membar`（`read=True` 的 `cp_async_bulk_wait_group(0)`），(2) 每个 CTA 等前面 CTA 写完这同一块 `dQ`。负载不均时，等锁能把性能拖垮。所以 **SPT 调度 + CTA swizzle** 是关键：因果 mask 时 KV 块降序、query 块从对角向上、`dQ` 归约按 query 块索引降序，保证没有 CTA 第一次写 `dQ` 就被卡住。

代码里还有一堆针对 `hdim==192`、block-sparsity、`kv_subtile > cta_group_size` 的锁值细节（`_dq_semaphore_lock_value` 里那段 `[NOTE] KV_subtile determ + spt` 的长注释），都是处理"tail 块没被调度、锁会悬空"的边界情况。这属于工程细节，理解到这层就够了。

## epilogue_dKV

`epilogue_dKV`（3846 行）把 `$ dK, dV $` 的累加器从 TMEM 读出、写回全局。GQA 时多个 query head 共享一组 KV，这里要把跨 head 的 `$ dK, dV $` 求和。FA4 用 TMA 把 `$ dK/dV $` 存出去（`epilogue_dK_or_dV_tma`），配合 `use_2cta_instrs` 时同样拆分。

## 一句话

FA4 反向的"逐行"读下来，三个关键改动都在为同一个目标服务：**让 Blackwell 反向的共享内存带宽不再是瓶颈**。TMEM 共享把中间量搬进张量核自带的存储；2-CTA MMA + DSMEM 把 dQ 的归约轴在 CTA pair 内并行、全局原子减半；确定性锁把不可复现的原子归约变成可复现的、代价是 membar 和等锁。前向是躲 exp 单元，反向是躲共享内存。

## Reference

- flash-attention 仓库（flash_attn/cute/flash_bwd_sm100.py）：<https://github.com/Dao-AILab/flash-attention>
- FlashAttention-4 论文（arXiv:2603.05451，backward / 2-CTA / deterministic）：<https://arxiv.org/abs/2603.05451>
- PTX ISA（tcgen05 CtaGroup、cp.async.bulk reduce、cluster DSMEM）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
- CUTLASS cuTe-DSL：<https://github.com/NVIDIA/cutlass/tree/main/python/cutlass/cute_dsl>
