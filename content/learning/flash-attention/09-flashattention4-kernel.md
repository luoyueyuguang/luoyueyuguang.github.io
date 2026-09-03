[[learning/flash-attention/08-flashattention4|FA4 算法]] 讲了思路，这一篇真的逐行读 FA4 的前向内核。它在仓库 `flash_attn/cute/flash_fwd_sm100.py`，是 **CuTe-DSL（Python 嵌入）**，不是 C++。`@cute.jit` 装饰的函数会被下沉成 PTX，再经 `ptxas` 出 SASS。

上一篇说过 FA4 有四个 warp 组：`softmax×2`、`correction`、`mma/TMA`。这一篇从配置往下读，直到 `ex2_emulation_2` 那个多项式位技巧。

## 配置：warp 组 + TMEM 布局

`FlashAttentionForwardSm100.__init__` 里，`__config` 字典（约 100–120 行）按 `(use_2cta, is_causal, head_dim_padded, is_sm103)` 查表给参数。以 `(True, False, 128, False)`（2-CTA、非因果、hdim128）为例：

```python
{"ex2_emu_freq": 10, "ex2_emu_start_frg": 1, "num_regs_softmax": 176, "num_regs_correction": 88}
```

- `ex2_emu_freq=10`：每 10 个 fragment 里有若干走多项式模拟（见下文 `apply_exp2_convert`）。
- `num_regs_softmax=176`、`num_regs_correction=88`：每个 softmax warpgroup / correction warpgroup 的寄存器数。
- `num_regs_other` 反推：`512 - num_regs_softmax*2 - num_regs_correction`，即给 mma/TMA warp 的。

warp 分工（302–303 行）：

```python
self.softmax0_warp_ids = (0, 1, 2, 3)
self.softmax1_warp_ids = (4, 5, 6, 7)
self.correction_warp_ids = (8, 9, 10, 11)
self.mma_warp_id = 12
```

TMEM 布局（344–349 行）：

```python
self.tmem_s_offset = [0, self.n_block_size]        # e.g. 0, 128
self.tmem_o_offset = [...]                          # e.g. 256, 384
self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
```

两个 `$ S $` tile 在 TMEM 列 0、128（一个 stage 一个），两个 `$ O $` tile 在 256、384。一个线程能 hold 整行 128 个元素（BF16 输入 128 寄存器、输出可能 64），所以寄存器分配是 FA4 的关键。

![FA4 的 4 个 warp 组和 TMEM 布局](/learning/assets/fa4-tmem.svg)

2-CTA 相关（195–204 行）：

```python
self.cta_group_size = 2 if self.use_2cta_instrs else 1
self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)  # (256, 128, 128)
self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
```

`use_2cta_instrs` 时 `cta_group_size=2`，一个 CTA pair 当一个 `$ M=256 $` 的 tile，`cluster_shape_mn=(2,1)`。

## softmax_loop：主循环

`@cute.jit def softmax_loop(...)` 是 consumer 侧的主循环，核心是 `while work_tile.is_valid_tile:`。每次取一个 work tile（`m_block, head_idx, batch_idx, split_idx`），算出这个 tile 要扫的 KV 块区间 `n_block_min..n_block_max`，再构造 mask：

```python
work_tile = tile_scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
    kv_head_idx = self._kv_head_idx(head_idx)
    n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx=split_idx, ...)
    mask = AttentionMaskCls(seqlen)
    ...
    work_tile = tile_scheduler.advance_to_next_work()
```

- `_kv_head_idx`：GQA 时把 query head 映射到 kv head（FA4 沿用 FA3 的 descale 语义）。
- `get_n_block_min_max`：因果 / 局部窗口 / split-KV 时算这个行块要处理的 KV 块范围。
- `rescale_threshold` 在这里设置（`if q_dtype.width==16: 8.0 else 0.0`），对应上一篇的 `$ \tau $`。

## softmax_step：一步 softmax

这是 FA4 的"行"级核心，和 [[learning/flash-attention/03-forward-kernel|FA1/FA2 的 softmax_rescale_o]] 对应，但换成了 cell 循环加 TMEM。

**① 等 S 到位。**

```python
pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
```

consumer 等 MMA warpgroup 把这一 stage 的 `$ S $`（128×128，fp32 累加器）写进 TMEM。这是 2 级流水线的依赖：softmax 要等 GEMM0 完成。

**② 从 TMEM 把 S 装进寄存器，同时算行最大值。**

```python
tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
hw_row_max = Float32(-Float32.inf)
if const_expr(self.use_ldred_rowmax):
    tSrS_red = cute.make_rmem_tensor(((1,1), *tSrS_t2r.shape[1:]), self.qk_acc_dtype)
    cute.copy(thr_tmem_load, tStS_t2r, (tSrS_t2r, tSrS_red))   # 一次拷贝同时取 S 和每 x32 tile 的 max
    for i in cute.range_constexpr(cute.size(tSrS_red.shape)):
        hw_row_max = cute.arch.fmax(hw_row_max, tSrS_red[i])    # 把各 tile 的 max 归约
else:
    cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
```

- `use_ldred_rowmax`：用 `tcgen05.copy.LdRed32x32bOp`（SM103 的硬件归约版），`ld.red` 在把 `$ S $` 装回寄存器的同时，额外返一个每 x32 tile 的 max 在 `tSrS_red`。省掉一行软件的 fmax 树。
- 前半段 `cute.copy(thr_tmem_load, tStS_t2r, (tSrS_t2r, tSrS_red))` 是"一箭双雕"：S 数据 + 硬件 max 一起拿。

**③ mask + 更新行最大。**

```python
if const_expr(mask_fn is not None):
    mask_fn(tSrS_t2r, n_block=n_block)                          # 因果/local mask，把越界位置置 -inf
if const_expr(self.use_ldred_rowmax and mask_fn is None):
    row_max, acc_scale = softmax.update_row_max_precomputed(hw_row_max, is_first)
else:
    row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
```

关键分支：**mask 之后，硬件的 max 就失效了**（mask 把某些值改成 `$ -\infty $`），所以只要 `mask_fn is None`（本块不需要 mask）才敢用 `update_row_max_precomputed(hw_row_max)`；否则回退到软件的 `update_row_max`（重新 fmax 一遍）。

`update_row_max`（softmax.py）里的条件缩放就藏在 `is_first` 分支：

```python
acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2     # (m_old - m_new)·log2e
acc_scale = cute.math.exp2(acc_scale_, fastmath=True)           # 理论上 = e^{m_old - m_new}
if const_expr(self.rescale_threshold > 0.0):
    if acc_scale_ >= -self.rescale_threshold:                  # m_new - m_old ≤ τ（log2 单位）
        row_max_new = row_max_old                              # 跳过：不更新 max
        row_max_safe = row_max_old
        acc_scale = 1.0                                        # 不重缩放
```

`rescale_threshold=8.0`（FP16/BF16）时，`m_new - m_old`（log2 单位）小于约 8 就**不重缩**，`acc_scale=1.0`。这就是 [[learning/flash-attention/08-flashattention4|条件缩放]]。

**④ 把 acc_scale 交给 correction warpgroup。**

```python
if const_expr(not is_first):
    sScale[thread_idx + stage * self.m_block_size] = acc_scale   # 存到共享/寄存器缓冲
sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)      # 通知 correction：row_max/scale 就绪
```

`acc_scale` 不在这里消费，而是通过 `sScale` 缓冲 + `sm_stats_barrier` 交给 correction warpgroup，让它把 `$ O $` 的旧块重缩（`$ e^{m_{old}-m_{new}} $ O_{old}`），从而**退出 softmax 的关键路径**。

**⑤ 减 max、exp2、转精度。**

```python
softmax.scale_subtract_rowmax(tSrS_t2r, row_max)   # S = S·scale_log2 - row_max·scale_log2
...
softmax.apply_exp2_convert(tSrS_t2r, tSrP_r2t, ex2_emu_freq=self.ex2_emu_freq, ex2_emu_start_frg=self.ex2_emu_start_frg)
```

`scale_subtract_rowmax` 用 `fma_packed_f32x2`（一次对两个 f32 做乘加），`bias = max_offset - row_max_scaled`，正是 `exp2(S·log2e - m·log2e)` 的偏置。`apply_exp2_convert` 里才真正做 exp2（含模拟）。

**⑥ 把 P 分块写回 TMEM，边写边通知 mma。**

```python
for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
    cute.copy(thr_tmem_store, tSrP_r2t_f32[None,None,i], tStP_r2t[None,None,i])
    if const_expr(self.split_P_arrive > 0):
        split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
        if const_expr(i + 1 == split_P_arrive_idx):
            cute.arch.fence_view_async_tmem_store()
            pipeline_s_p_o.consumer_release_w_index(stage)   # 前四分之三块 P 就绪，让 P·V MMA 先跑
cute.arch.fence_view_async_tmem_store()
pipeline_s_p_o.consumer_release_w_index(stage)               # 最后四分之一块 P 也就绪
```

这就是上一篇说的"把 `$ P $` 分成四份存"：`split_P_arrive > 0` 时，存完前四分之三就 `consumer_release`，让 `$ P\cdot V $` 的 MMA 和剩下的 softmax 并行；`fence_view_async_tmem_store` 保证 TMEM 写对 mma 可见。

**⑦ 更新行和。**

```python
pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
```

`update_row_sum` 是 `$ \ell = e^{m_{old}-m_{new}} \ell_{old} + \sum e^{S-m_{new}} $`，即 `init_val = row_sum[0] * row_scale`。

## apply_exp2_convert：模拟 vs 硬件

`apply_exp2_convert`（softmax.py）把 S 的 fragment 切成 32 个一组：

```python
frg_tile = 32
frg_cnt = cute.size(acc_S_row) // frg_tile
acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
for j in cute.range_constexpr(frg_cnt):
    for k in cute.range_constexpr(0, cute.size(acc_S_row_frg, 0), 2):
        if const_expr(ex2_emu_freq == 0):
            exp2(...)                                            # 全走硬件 MUFU.EX2
        else:
            if const_expr(k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                          or j >= frg_cnt - 1
                          or j < ex2_emu_start_frg):
                exp2(...)                                        # 硬件
            else:
                acc_S_row_frg[k,j], acc_S_row_frg[k+1,j] = utils.ex2_emulation_2(...)  # 多项式模拟
        acc_S_row_converted_frg[None, j].store(acc_S_row_frg[None, j].load().to(...))
```

- `k` 每隔 `ex2_emu_freq` 个里约有 `ex2_emu_res` 个走模拟（`k % freq < freq - res` 为真走硬件，否则走模拟），比例约 `ex2_emu_res / ex2_emu_freq`。`freq=16`、`res=4` 约 25% 模拟；`freq=32` 约 12.5%；`freq=10` 约 40%。边界约束（`j >= frg_cnt-1` 强制硬件、`j < ex2_emu_start_frg` 开头跳过）会再降一点。论文说的"10–25%"就是这个比例按 tile 配置调出来的典型区间。
- `acc_S_row_converted_frg.store(... .to(element_type))`：结果从 fp32 转成 fp16/bf16 再写回。

## ex2_emulation_2：多项式位技巧

`utils.ex2_emulation_2`（一个 DSL operator）是核心，`$ 2^x = 2^{\lfloor x \rfloor}\cdot 2^{x - \lfloor x \rfloor} $`：

```python
fp32_round_int = float(2**23 + 2**22)          # 0x4B400000
xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))   # clamp 防下溢
xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm")  # 加 1.5·2^23，rd 向下取整
xy_rounded_back = cute.arch.sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
xy_frac = cute.arch.sub_packed_f32x2(xy_clamped, xy_rounded_back)     # x - floor(x) ∈ [0,1)
xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree])  # 多项式 2^{frac}
x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])           # 拼回 2^x
```

`combine_int_frac_ex2` 的 PTX：

```ptx
shl.b32 x_rounded_e, x_rounded_i, 23;   // floor(x) 移进指数字段
add.s32 out_i, x_rounded_e, frac_ex_i;  // 加上 2^{frac} 的尾数位
```

- `add 2^23+2^22`：把 `x` 的整数部分挤进尾数的低段（`2^22` 保证 round-down 的舍入），`add.rm` 是向下取整。
- `x_frac = x - floor(x)`：`[0,1)`。
- `POLY_EX2[3] = (1.0, 0.69515, 0.22756, 0.07712)`，Horner 法用 `fma_packed_f32x2` 算 `2^{frac}`。
- `shl 23` + `add`：把 floor(x) 变成指数、把 `2^{frac}` 的尾数拼上，得到 `2^x` 的 IEEE 位模式。

## correction_loop / correction_rescale

`correction_loop`（2551 行起）是 correction warpgroup：它从 `sScale` 读回 `acc_scale`，对旧 `$ O $` 做重缩，并负责 epilogue。这就是"重缩放退出关键路径"的实现：softmax warpgroup 只管算 `$ P $`（和 `$ m, \ell $` 更新），`$ O $` 的合并让 correction warpgroup 在别的 warp 做 GEMM 时干。

## 一句话

FA4 前向的"逐行"读下来，本质还是那套在线 softmax，但**载体全换了**：累加器在 TMEM（不是寄存器）、行 max 用硬件 `ld.red` 省软件归约、exp2 按 `ex2_emu_freq` 部分走多项式、`$ P $` 分块写回边写边喂 MMA、`$ O $` 重缩丢给 correction warpgroup。每一条都是为了躲开 Blackwell 上"没涨的 exp 单元"和"爬升的共享内存流量"。

## Reference

- flash-attention 仓库（flash_attn/cute/flash_fwd_sm100.py、cute/softmax.py、cute/utils.py）：<https://github.com/Dao-AILab/flash-attention>
- CuTe-DSL：<https://github.com/NVIDIA/cutlass/tree/main/python/cutlass/cute_dsl>
- FlashAttention-4 论文（arXiv:2603.05451）：<https://arxiv.org/abs/2603.05451>
- PTX ISA（tcgen05.ld/st、fma、ex2）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
