前面讲的全是"一块怎么算"（kernel 内部）。这一篇往上一格：**GPU 上到底启动多少个 CTA、每个 CTA 分到哪一块**。也就是 `flash_fwd_launch_template.h`（决定 grid/block 形状）和 `hopper/tile_scheduler.hpp`（把 CTA 索引映射到 work）加 `hopper/heuristics.h`（决定 split 数）。这三层决定的是**占用率和负载均衡**，正是 [[learning/flash-attention/06-flashattention2|FA2]] 说的"work partitioning"的启动端。

## grid 形状：三维 (M, head, batch)

`flash_fwd_launch_template.h` 里先是块大小：

```cpp
int const num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
```

- `seqlen_q * qhead_per_khead`：query 总行数（`qhead_per_khead` 是 GQA 的 group 数，PACK_GQA 时把同组 head 打包）。
- `ceil_div(..., kBlockM)`：能切成多少 `m` 块；再 `round_up` 到 cluster 大小（2-CTA 时向上取整到 2 的倍数）。

然后 grid 由内核的静态方法算，再交给 CUTLASS 的 launch 封装（集群启动走 `cutlass::launch_kernel_on_cluster`，否则 `cutlass::kernel_launch`；源码里那句 `kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params)` 是注释掉的等价写法）：

```cpp
dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
dim3 block_dims = AttnKernel::get_block_shape();
int smem_size = AttnKernel::SharedStorageSize;
CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, ...));
```

`get_grid_shape`（tile_scheduler.hpp）本质是：

```cpp
return {uint32_t(params.num_blocks),                    // M 块数
        uint32_t((!Split ? 1 : params.num_splits) * params.num_head),   // head × split（fold 进来）
        uint32_t(params.num_batch)};                     // batch
```

grid.x = M 块数，grid.y = head × split（split-KV 时把 split 折进 y），grid.z = batch。**split-KV 不新增 grid 维度，而是把 split 折进 head 那一维**——这样每个 CTA 除了"哪块 M、哪个 head、哪个 batch"，还会多一个"哪一段 KV（split_idx）"。

这是 non-persistent 的 `SingleTileScheduler` 的 grid。persistent 调度器（Hopper 上非 split、非 varlen 的常见路径，即 `StaticPersistentTileScheduler`/`DynamicPersistentTileScheduler`）的 `get_grid_shape` 返回一维 `{num_sm}`：只启动 SM 数那么多 CTA，再用一个全局 tile 计数器从 work 队列里抢下一个 tile，所以 grid 里没有"M 块"那一维。

## Tile scheduler：CTA 索引 → work

`scheduler` 把一维的 `block_idx`（grid 展开后的 CTA 编号）映射到 `(m_block, head, batch, split)`。最常用的两个：

**SingleTileScheduler**（每 CTA 只做一个 tile，non-persistent）：

```cpp
// get_block_coord：坐标就是 grid 的三维索引
return {block_idx, bidh, bidb, !Split ? 0 : split_idx};

// get_initial_work：split 时把 grid.y 再拆成 (bidh, split_idx)
WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
if constexpr (Split) {
    int split_idx;
    work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
    work_info.split_idx = split_idx;
}
```

注意 `WorkTileInfo` 里 `bidh` 对应 grid.y、`bidb` 对应 grid.z，**只有 split 需要额外计算**——因为 split 被折进了 grid.y，所以这里用 `nsplits_divmod` 把 grid.y 再拆成 `(bidh, split_idx)`。

**StaticPersistentTileScheduler**（persistent，一块 CTA 抢多个 tile）：

```cpp
// get_block_coord：从一维 tile_idx 逐级拆
bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
if constexpr (Split) { bidh = params.nsplits_divmod.divmod(split_idx, bidh); }
return {block, bidh, bidb, split_idx};
```

都用 `FastDivmod` 把一维 CTA 索引高效拆成多维坐标。**它省的是整数除法**：GPU 上硬件没有单指令整数除法，编译器要展开成一条几十条指令的序列；而 CUTLASS 的 `FastDivmod` 在构造时预计算两个常量，把除法换成一次乘法取高位：

```cpp
// 构造（host 侧，每个除数算一次）
unsigned p = 31 + find_log2(divisor);
multiplier  = ((1ull << p) + divisor - 1) / divisor;   // ceil(2^p / divisor)
shift_right = p - 32;
// 使用（device 侧，每条 CTA 每级拆分算一次）
quotient  = __umulhi(dividend, multiplier) >> shift_right;   // 64 位乘法取高 32 位，再右移
remainder = dividend - quotient * divisor;
```

因为除数（head 数、batch 数、split 数、swizzle 大小）在 launch 时就知道且固定，这套预计算摊到所有 CTA 上是免费的——代价转移到了 host 侧算 `multiplier`。**这就是"block scheduling"**：决定谁会做哪一块，且不因此拖慢每个 CTA 的启动。

## split-KV 启发式

`heuristics.h` 的 `num_splits_heuristic` 决定要把 KV 切成几段并行：

```cpp
int total_mblocks = ...;   // M 块总数
if (total_mblocks >= 0.8f * num_SMs) {
    // 差不多填满 SM 了，用 1 split；除非 KV 头大到装不进 L2（50MB）
    if (size_one_kv_head > size_l2 && num_m_blocks >= num_SMs * 2 && !is_causal_or_local) {
        return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
    }
    return 1;
}
// 别 split 太少次（num_n_blocks 很小时）
if (num_n_blocks <= 4) { return 1; }
max_splits = std::min({max_splits, num_SMs, num_n_blocks});   // split 不能超过 SM 数和 KV 块数
// 找出让"波次效率"最高的 split 数，再取达到最优 85% 的最小值
float max_efficiency = 0;
for (num_splits = 1; ...; num_splits++) {
    float n_waves = float(total_mblocks * num_splits) / num_SMs;
    float eff = n_waves / ceil(n_waves);
    max_efficiency = max(max_efficiency, eff);
    efficiency.push_back(eff);
}
for (num_splits = 1; ...; num_splits++) {
    if (efficiency[num_splits - 1] >= 0.85f * max_efficiency) { return num_splits; }
}
```

关键指标是**波次效率** `n_waves / ceil(n_waves)`：`n_waves = total_mblocks × num_splits / num_SMs` 是需要的波数，`ceil` 是实际要跑的整数波数，比值就是最后一波的填充率——如果总 tile 数是 SM 数的整数倍，最后一波满载（eff = 1）；否则最后一波空转（`48×2/108 = 0.89`，所以 eff = 0.89）。代码注释里的例子正是 `batch × n_heads = 48`、108 个 SM、每个 (batch,head) 只有一个 M 块：2 split 效率 0.89，3 split 效率 `1.33/2 = 0.67`，取 2。注意这里 2 split 的 eff 0.89 已经不完美（一波只用了 89% 的 SM），但它是所有 split 数里最好的，所以仍被选中——**"波次效率"追求的是最后一波别太空，不是每波都满**。

`total_mblocks` 是"要调度的 M 块总数"，真正算的时候是 `batch × n_heads_kv × num_m_blocks`（kv head 数，不是 q head 数——因为 GQA 下 KV 是共享的），见 `flash_api.cpp`：

```cpp
int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks,
                            size_one_kv_head, params.is_causal || params.is_local, 128);
```

`max_splits` 传的硬上限是 128。

**但是**：如果 M 块已经能填满 SM（`total_mblocks >= 0.8*num_SMs`），就不为波次效率而 split——除非单个 KV 头大到装不进 L2（那样非得拆 KV 不可，否则 cache 疯狂 miss）。这就是"占用率优先、cache 兜底"的策略。

**为什么"wave 效率"这么敏感**：Hopper 上一个 CTA 几乎吃光 SMEM。H100 每个 SM 约 228KB，FA3 留 ~3KB 给 LSE / dPsum / mbarrier 后，张量缓冲还能用约 **224KB**，一个 CTA 就是这样一个大 tile。于是 `num_SMs` 既是可并发 CTA 的上限，又正好是"最后一波是否满载"的分母——`total_mblocks`（乘以 split 数）越接近 SM 数的整数倍，最后的零头越小，空转越少。这也是为什么 `total_mblocks >= 0.8*num_SMs` 时干脆不再为波次而 split：SM 已经被 tile 占满，再切只在给 partial 波次添乱。

## Pack-GQA 启发式

`should_pack_gqa`：GQA 每组的多个 query head 共享 KV。**如果不打包**，每个 head 独立做一个小 M 块，`seqlen_q` 不是 kBlockM 的倍数时末尾浪费一块；**打包**后把 `seqlen_q * qhead_per_khead` 当成一个长 M 维度，跨 head 一起切块：

```cpp
float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency;
```

打包效率高 10% 以上就打包。varlen（变长序列）时直接打包（长度未知，打包更稳）。

## 因果 / local 的 tile 重排

因果时每个 `m_block` 要扫的 KV 块数不一样（靠前的行扫得少，靠后的扫得多）。如果按顺序调度，前面的 CTA 先做完、后面的 CTA 拖到很晚，**负载不均**。`tile_scheduler.hpp` 里专门为因果/local 设计的调度器**按"预计工作量"给 tile 排序**——工作量大的先排：`DynamicPersistentTileScheduler::get_block_coord` 末尾就是 LPT（Longest-processing-time-first），一行把 M 块顺序倒过来：

```cpp
// Longest-processing-time-first
block = params.m_block_divmod.divisor - 1 - block;
```

让扫得最长的行块先被处理。这和 [[learning/flash-attention/12-flashattention4-bwd-kernel|FA4 反向]] 写的是同一类思想：先处理"最长"的块，让所有 CTA 尽量同时收工。varlen 走 `VarlenDynamicPersistentTileScheduler`，排序方式由模板参数 `LPT` / `Sort` 控制，launch 端设的是 `LPT = Is_causal || Is_local`、`Sort = !Is_local`（见 `flash_fwd_launch_template.h`）。

## 一句话

启动这一层逻辑很直白：**grid 是 (M 块, head×split, batch) 三维；tile scheduler 用 FastDivmod 把 CTA 编号拆成坐标；split-KV 看波次效率和 L2；pack-GQA 看切块浪费；因果就把 tile 按工作量重排。** 它不在 kernel 里，却决定了整个 GPU 怎么被占满、最后一批 CTA 空不空转。

## Reference
- flash-attention 仓库（hopper/flash_fwd_launch_template.h、hopper/tile_scheduler.hpp、hopper/heuristics.h）：<https://github.com/Dao-AILab/flash-attention>
- SM90 调参笔记（SMEM 预算、CTA 每 SM 数）：<https://github.com/Dao-AILab/flash-attention/blob/main/AI/SM90_BLOCK_SIZE_TUNING.md>
- CUTLASS 的 tile scheduler / cluster launch：<https://github.com/NVIDIA/cutlass>
- FlashAttention-2（sequence-length 并行与 work partitioning 动机）：<https://arxiv.org/abs/2307.08691>
