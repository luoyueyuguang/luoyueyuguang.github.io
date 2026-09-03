前面讲的全是"一块怎么算"（kernel 内部）。这一篇往上一格：**GPU 上到底启动多少个 CTA、每个 CTA 分到哪一块**。也就是 `flash_fwd_launch_template.h`（决定 grid/block 形状）和 `hopper/tile_scheduler.hpp`（把 CTA 索引映射到 work）加 `hopper/heuristics.h`（决定 split 数）。这三层决定的是**占用率和负载均衡**，正是 [[learning/flash-attention/05-flashattention2|FA2]] 说的"work partitioning"的启动端。

## grid 形状：三维 (M, head, batch)

`flash_fwd_launch_template.h` 里先是块大小：

```cpp
int const num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
```

- `seqlen_q * qhead_per_khead`：query 总行数（`qhead_per_khead` 是 GQA 的 group 数，PACK_GQA 时把同组 head 打包）。
- `ceil_div(..., kBlockM)`：能切成多少 `m` 块；再 `round_up` 到 cluster 大小（2-CTA 时向上取整到 2 的倍数）。

然后 grid 由内核的静态方法算：

```cpp
dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
dim3 block_dims = AttnKernel::get_block_shape();
kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
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
// get_block_coord
return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
// 而 bidh/bidb 从线性索引拆出来：
int split_idx = params.nsplits_divmod.divmod(split_idx, work_info.bidh);  // 先拆 split vs head
```

**StaticPersistentTileScheduler**（persistent，一块 CTA 抢多个 tile）：

```cpp
// 从 block 线性索引逐级拆
bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
```

都用 `FastDivmod`（编译期/运行期都知道除数的快速取模），把一维 CTA 索引高效拆成多维坐标。**这就是"block scheduling"**：决定谁会做哪一块。

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
// 找出让"波次效率"最高的 split 数，再取达到最优 85% 的最小值
float max_efficiency = 0;
for (num_splits = 1; ...; num_splits++) {
    float n_waves = float(total_mblocks * num_splits) / num_SMs;
    float eff = n_waves / ceil(n_waves);
    max_efficiency = max(max_efficiency, eff);
}
for (num_splits = 1; ...; num_splits++) {
    if (efficiency[num_splits - 1] >= 0.85f * max_efficiency) { return num_splits; }
}
```

关键指标是**波次效率** `n_waves / ceil(n_waves)`：如果总 tile 数是 SM 数的整数倍，最后一波满载；否则最后一波空转。例如 48 个 (batch,head)、108 个 SM：2 split 效率 0.89，3 split 0.67，所以选 2。

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

因果时每个 `m_block` 要扫的 KV 块数不一样（靠前的行扫得少，靠后的扫得多）。如果按顺序调度，前面的 CTA 先做完、后面的 CTA 拖到很晚，**负载不均**。`tile_scheduler.hpp` 里专门为因果/local 设计的调度器（非 varlen 用 `DynamicPersistentTileScheduler`，varlen 用 `VarlenDynamicPersistentTileScheduler`）**按"预计工作量"给 tile 排序**——工作量大的先排：`DynamicPersistentTileScheduler` 里就是 LPT（`block = divisor - 1 - block`，注释写着 "Longest-processing-time-first"），让扫得最长的行块先被处理。这和 [[learning/flash-attention/10-flashattention4-bwd-kernel|FA4 反向]] 写的是同一类思想：先处理"最长"的块，让所有 CTA 尽量同时收工。varlen 版本由 `LPT`/`Sort` 模板参数（launch 端设 `LPT = Is_causal || Is_local`、`Sort = !Is_local`）决定排序。

## 一句话

启动这一层逻辑很直白：**grid 是 (M 块, head×split, batch) 三维；tile scheduler 用 FastDivmod 把 CTA 编号拆成坐标；split-KV 看波次效率和 L2；pack-GQA 看切块浪费；因果就把 tile 按工作量重排。** 它不在 kernel 里，却决定了整个 GPU 怎么被占满、最后一批 CTA 空不空转。

## Reference
- flash-attention 仓库（hopper/flash_fwd_launch_template.h、hopper/tile_scheduler.hpp、hopper/heuristics.h）：<https://github.com/Dao-AILab/flash-attention>
- SM90 调参笔记（SMEM 预算、CTA 每 SM 数）：<https://github.com/Dao-AILab/flash-attention/blob/main/AI/SM90_BLOCK_SIZE_TUNING.md>
- CUTLASS 的 tile scheduler / cluster launch：<https://github.com/NVIDIA/cutlass>
- FlashAttention-2（sequence-length 并行与 work partitioning 动机）：<https://arxiv.org/abs/2307.08691>
