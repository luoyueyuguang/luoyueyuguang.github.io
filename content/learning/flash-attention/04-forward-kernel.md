这一篇逐行读 forward kernel。代码在 `flash-attention` 仓库的 `csrc/flash_attn/src/flash_fwd_kernel.h`（Ampere/FA2 版，函数 `compute_attn_1rowblock`）。它是纯 CUDA + cuTe（CUTLASS 的 tensor 抽象），不是教学玩具，所以先解释三个 cuTe 概念，再逐段读。

先看最核心的循环。中间只做四件事，和 [[learning/flash-attention/02-online-softmax|算法]] 一一对应：

```text
for each column block j (reverse order):
    S = Q_i · K_j^T                    ← GEMM（QK^T）
    softmax_rescale_o(S, O)           ← online softmax 加点
    P = fp16(S)                       ← 转成 tensor core 的输入精度
    O += P · V_j                      ← GEMM（P·V）
```

## 三个 cuTe 概念

**1. Tensor。** `make_tensor(指针, 形状, 步长)` 只是给一块内存描述"怎么看"，不拷贝。`gQ`、`gK`、`gV` 是全局内存里的 tile，`sQ`、`sK`、`sV` 是共享内存里的 tile。

**2. 线程块只负责一个 tile。** 每个 thread block 处理"一个 batch、一个 head、一行块 $ m $"的输出。所以读出 $ gQ $ 用 `local_tile` 切出 `(kBlockM, d)` 这一块，$ gK $、$ gV $ 切成 `(kBlockN, d)` 的第 $ j $ 列块。源码里就是这三行：

```cpp
Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
```

`local_tile` 的最后一个坐标给 `_` 表示"这一维先不切"——于是 `gK` 多出一个列块维度，主循环里 `tVgV(_, _, _, n_block)` 用它索引当前列块。

![一个 thread block 的循环：固定 Q_i，遍历 K_j / V_j](/learning/assets/fa-tiling.svg)

> 自绘示意图

**3. MMA 片段（fragment）与 tile。** 张量核一次 `mma` 处理一个小 tile（如 `m16n8k16`，在 CuTe 里叫 `SM80_16x8x16_F32F16F16F32_TN`），一个 warp 的 32 个线程分别持有这个 tile 的一部分，这叫 fragment。`partition_fragment_A/B/C`（定义在 `cute::ThrMMA`）就是按 mma 的形状把操作数分给每个线程，得到能直接喂给 mma 指令的寄存器片段；A/B 的输入是 smem 里的 tile，C 只需要一个形状（`partition_fragment_C(tiled_mma, Shape<...>{})`）。

## 参数与共享内存

```cpp
extern __shared__ char smem_[];
const int tidx = threadIdx.x;
constexpr int kBlockM = Kernel_traits::kBlockM;   // 例如 64 或 128
constexpr int kBlockN = Kernel_traits::kBlockN;
```

`smem_` 是整个 thread block 的共享内存（192 KB 那块的这一份）。后面所有 `sQ`、`sK`、`sV` 都在里面按偏移分配。

## 分配 smem tile

```cpp
Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                        typename Kernel_traits::SmemLayoutQ{});
Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                        typename Kernel_traits::SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
```

- `SmemLayoutQ` / `SmemLayoutKV` 是**带 swizzle 的布局**，目的是避免共享内存 bank conflict（同一个周期里多个线程撞同一个 bank）。
- `sVt` 是 `V` 的**转置视图**（同一块内存换个布局别名，不搬数据）：第二个 GEMM 是 $ P \cdot V $，mma 的 B 操作数按 $ K \times N $ 的形状读，而 `sV` 里存的是 $ B_c \times d $，所以要一个 $ d \times B_c $ 的视图。代码里其实有两个别名——带 swizzle 的 `sVt` 给 smem→寄存器的 copy atom，`sVtNoSwizzle` 给 `partition_fragment_B`（`gemm_rs` 直接从 smem 读 B）。
- `Share_Q_K_smem` 为真时 `sQ` 和 `sK` 共用同一片内存：`kSmemSize = max(kSmemQSize, kSmemKVSize)` 而不是两者相加，省下 $ B_r \times d $ 的 SRAM。前提是 `Is_Q_in_regs` 也为真（`kernel_traits.h` 里注释写着 *"If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true"*）：Q 整块先搬进寄存器，那片 smem 就空出来给 K/V 循环用了。多出来的 SRAM 直接换成更大的 `kBlockN`。

## 把 tile 从 HBM 拷进 smem

```cpp
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);   // 源：全局
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);   // 目的：共享
```

这是 `cp.async` 的抽象：`partition_S`/`partition_D` 把"整个 block 的拷贝工作"分给每个线程，发出异步的内存拷贝。真正执行在 prologue 里：

```cpp
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                             binfo.actual_seqlen_q - m_block * kBlockM);
```

`tQcQ` 是"坐标张量"（identity tensor，用来把每个 tile 元素映射回它的逻辑行列号），`tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d` 是"第 $ k $ 列是否落在 head dim 内"的谓词——源码里只在 `if (!Is_even_K)` 时才真的去填这两个谓词。越界的列怎么处理由 `copy` 的模板参数决定：`Clear_OOB_K` 默认是 `true`，所以 Q/K/V 拷进 smem 时越界的列会被**清零**（这一点很关键：如果不清零，smem 里留着别的轮次的数据，位模式可能是 NaN，那 $ \times 0 $ 还是 NaN）。只有写回 gmem 的输出（$ O $、$ dQ $）才显式传 `Clear_OOB_K=false`，免得把 0 写进结果。

`copy` 的参数里还有 M/N 方向的边界：Q 传的是 `binfo.actual_seqlen_q - m_block * kBlockM`，K 传的是 `binfo.actual_seqlen_k - n_block * kBlockN`（`Is_even_MN` 为真时这一项就没意义了）。所以 padding 要分两个方向看——**M/N 方向的越界行**（`seqlen` 不是 block 的整数倍，或 varlen 的 ragged 序列）由 `Is_even_MN` + `max_MN` 管，**head dim 方向的越界列**由 `Is_even_K` + `tQpQ` 管。

**为什么 predicated**：`d` 不必等于这个 kernel 实例化的 head dim。启动模板里写死了 `const bool is_even_K = params.d == Kernel_traits::kHeadDim;`，即只有当 $ d $ 正好是实例化的 head dim（32/64/96/128/192/256）时才置真；否则（如 ALiBi/RoPE 一类的 head dim = 40、DeepSeek 的 $ d=192 $ 用 256 的实例化）tensor core tile 的 k 维度固定是 16 的倍数，多出来的列在 smem 操作数里被清零、在写回输出时被谓词挡掉（不写）。这就是 `Is_even_K` 存在的原因。

## GEMM 所需的寄存器片段

```cpp
typename Kernel_traits::TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(tidx);
Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
Tensor tSrK  = thr_mma.partition_fragment_B(sK);
Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);
Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
```

- `tSrQ` 是 A 片段（$ Q_i $），`tSrK` 是 B 片段（$ K_j $），GEMM 算 $ S = Q_i K_j^\top $。
- `tOrVt` 是 $ V $ 的转置 B 片段，给第二个 GEMM（$ P \cdot V $）。
- `acc_o` 是 $ kBlockM \times d $ 的累加器（fp32），就是那个"未归一化的 $ \widetilde{O} $"。

**这一步就体现了 FA2 的 warp 分工。** `tSrQ` 是 A 片段，A 是按 mma 的 M 维分给 warps 的，所以"`tSrQ` 怎么切"直接决定每个 warp 负责哪些 query 行。FA2 把 $ Q $ 切成 4 份分给 4 个 warp、$ K,V $ 对所有 warp 可见（FA2 §3.3 原文：*"we instead split Q across 4 warps while keeping K and V accessible by all warps"*），于是每个 warp 独立算出自己那 4 分之一行块的全部 $ \widetilde{O} $（先算 $ S $ 再算 $ P \cdot V $），**warp 之间不需要任何通信**。FA1 则相反，把 $ K, V $ 切给 warp、$ Q $ 共享，每个 warp 只算出一部分 $ \widetilde{O} $，得写回 smem、`__syncthreads()`、再加起来——论文管这叫 "split-K" 方案。详见 [[learning/flash-attention/06-flashattention2|FA2]]。

## prologue：先拷第一块

```cpp
// 拷 Q（只拷本 block 需要的那块）
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, ...);
if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }
```

然后把 `sQ` 读到寄存器片段里（FA2 的 `Is_Q_in_regs` 为真）：

```cpp
if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    FLASH_NAMESPACE::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
}
```

`Q` 一旦进了寄存器（$ B_r \times d $ 也放得下），整个主循环就不再读它，省下每轮重复读 smem 的开销。这只有 FA2 的"Q 切给 warp"才做得出来；FA1 里 $ Q $ 被所有 warp 共享，不能这么留在寄存器。

## 主循环：S = QK^T

为什么倒着扫？`flash_fwd_launch_template.h` 的 grid 是 `dim3 grid(num_m_block, params.b, params.h)`——`blockIdx.x` 就是行块 `m_block`，每个 block 只跑自己那一段 query。循环从 `n_block_max-1` 递减到 `n_block_min`：源码注释说两个理由，一是**只有最后一个列块需要 mask**（越过 `actual_seqlen_k` 的那部分），先做它、把 K/V 的边界处理集中在一处；二是倒着走可以少留一个寄存器（不必同时保存 `n_block` 和 `n_block_max`）。现在的代码把这段拆成两个循环：前 `n_masking_steps` 步（非 causal 时是 1，causal 时是 `ceil(kBlockM/kBlockN)`，`seqlen_k` 不是 `kBlockN` 整数倍时再多一步）带 mask，后面的循环注释直接写着 *"These are the iterations where we don't need masking on S"*。

```cpp
for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    FLASH_NAMESPACE::cp_async_wait<0>();     // 等 K/V 拷贝完成
    __syncthreads();
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );
```

- `acc_s` 是 $ kBlockM \times kBlockN $ 的 fp32 累加器，就是 $ S_{ij} $。
- `FLASH_NAMESPACE::gemm` 封装了 `mma.sync.aligned.m16n8k16` 这类指令：`tSrQ`（A，寄存器）乘 `tSrK`（B，smem），结果写进 `acc_s`（C，寄存器）。
- `A_in_regs=Is_Q_in_regs` 告诉它 A 操作数在寄存器里还是 smem 里，省一次 `retile`。

## online softmax + 重缩放 O

之前读的是 `flash_fwd_kernel.h`，softmax 的实现在 `csrc/flash_attn/src/softmax.h` 的 `Softmax::softmax_rescale_o`。这正好对应 [[learning/flash-attention/02-online-softmax|02 那篇]] 伪代码的第 11–13 行（行最大合并、$ \ell $ 合并、$ \widetilde{O} $ 重缩）：

```cpp
template<bool Is_first, ...>
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    if (Is_first) {
        reduce_max</*zero_init=*/true>(scores, row_max);
        scale_apply_exp2(scores, row_max, softmax_scale_log2);
        reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        reduce_max</*zero_init=*/false>(scores, row_max);
        // 对 acc_o 先按新旧 max 重缩放
        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_scale = exp2f((scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale;
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        scale_apply_exp2(scores, row_max, softmax_scale_log2);
        reduce_sum</*zero_init=*/false>(scores, row_sum);
    }
}
```

逐行对应：

1. `reduce_max` 算这一块 $ S $ 的行最大 $ \tilde{m} $，并和已有的 $ m $ 合并（`zero_init=false` 时做 `max(m, new)`），即算法里的 $ m_i^{new} = \max(m_i, \tilde{m}_{ij}) $。
2. `cute::copy(row_max, scores_max_prev)` 保存旧 $ m_i $。
3. `scores_scale = exp2f((scores_max_prev - row_max) * softmax_scale_log2)`，就是 $ e^{m_i^{\text{old}} - m_i^{new}} $。**它同时乘到 $ \ell $ 和 $ \widetilde{O} $ 上**：旧的指数和和旧的输出累加器都还挂在旧 max 的尺度上，换了更大的 max，两项要各乘同一个因子才能对齐。（源码里这个因子是从 `scores_max_cur` 算的，而不是直接拿 `row_max`：`Check_inf` 为真（causal / local）时 `row_max` 为 `-INFINITY` 就先换成 0，免得整行被掩码遮掉时出现 $ -\infty - (-\infty) $。它和下面 `scale_apply_exp2` 里的 `max_scaled` 守卫是一对：一个管 $ \widetilde{O} $ 的重缩，一个管 $ \widetilde{P} $。）
4. `scale_apply_exp2(scores, row_max, ...)`：把 $ S $ 按新 max 做 $ \widetilde{P} = e^{S - m} $，即 `exp2(S * scale - m * scale)`。用 `exp2f` 而不是 `expf` 是技巧：$ e^x = 2^{x \log_2 e} $，写成 `exp2f(x * scale)` 编译器能合成 `ffma`（一次乘加），比 `fadd` + `fmul` 各一条省一条指令。指数是特殊函数单元（SFU）干的活，吞吐远低于张量核——FA3 论文给的 H100 数字是矩阵乘 989 TFLOPS 对特殊函数 3.9 TFLOPS，所以 FA4 干脆用多项式把 `exp2` 模拟出来（见 [[learning/flash-attention/10-flashattention4|FA4]]）。
5. `reduce_sum` 把 $ \widetilde{P} $ 的行和加进 $ \ell $。

把第 3、5 步的**记账**写成公式（第 $ j $ 轮，$ \widetilde{P}_{ij} = e^{S_{ij} - m_i^{new}} $）：

$$
\ell_i \leftarrow e^{m_i^{\text{old}} - m_i^{\text{new}}}\,\ell_i + \operatorname{rowsum}(\widetilde{P}_{ij}),
\qquad
\widetilde{O}_i \leftarrow e^{m_i^{\text{old}} - m_i^{\text{new}}}\,\widetilde{O}_i + \widetilde{P}_{ij} V_j
$$

**这里只重缩、不除 $ \ell $。** FA1 论文 Algorithm 1 第 15 行是每一步都用新的 $ \ell_i $ 归一化一次（Triton 版里对应 `acc_scale = l_i / l_i_new * alpha` 那两行），FA2 把这次除法挪到 epilogue 只做一次：$ \widetilde{O}_i $ 全程是"相对当前 max 的未归一化量"，最后才 $ O_i = \widetilde{O}_i / \ell_i $。省掉的是每轮 $ B_r $ 个除法——A100 上非 matmul 的 FP32 只有 19.5 TFLOPs/s、张量核有 312 TFLOPs/s，一个非 matmul FLOP 贵 16 倍（FA2 §3.1），能挪出循环就挪。

顺带两个容易漏的细节：

- 掩码全 `-inf` 的行（因果 attention 里被整块遮掉的 query）会让 max 保持 `-inf`，此时 $ -\infty - (-\infty) $ 是 NaN。`scale_apply_exp2` 里专门写了 `max_scaled = max(mi) == -INFINITY ? 0.f : ...` 把这个 NaN 挡掉。
- 循环里的 `reduce_sum` **只做线程内归约**，不做跨线程的 quad allreduce（源码注释：*"We don't do the reduce across threads here since we don't need to use the row_sum. We do that reduce at the end when we need to normalize."*）。跨线程的部分留到 `normalize_softmax_lse` 里一次算完——省掉每轮 $ T_c $ 次 shuffle。

## 转精度 + 第二个 GEMM：O += P·V

```cpp
Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);   // fp32 -> fp16/bf16
if (Is_dropout) { dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps); }
Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs(rP.layout()));
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

- `convert_type<Element>`：tensor core 的输入是 fp16/bf16，`acc_s` 是 fp32，所以在喂给 $ P \cdot V $ 前要把 $ S $（此时已经是"近似 P"）转成 fp16。这是 attention 唯一的精度损失点之一（softmax 结果存成 fp16）。
- `convert_layout_acc_Aregs`：`acc_s` 作为 C 累加器时，每个线程持有的元素分布，和作为 A 操作数（$ P $）时期望的分布不一样。要重排一下，否则 mma 算错。这就是 FA3 里"FP32 累加器布局和 operand A 布局不同"需要在 FP8 用 byte-permute 解决的同一类问题。
- `FLASH_NAMESPACE::gemm_rs` 是"register × shared" GEMM：$ P $ 在寄存器、$ V $ 在 smem，结果加进 `acc_o`。`rs` 后缀表示 A 从寄存器（register）、B 从共享（shared）取。

`acc_o` 一直攒的是未归一化的 $ \widetilde{O} $，这行就把它和 $ \widetilde{P} \cdot V_j $ 相加，对应算法里的 $ O_i \leftarrow e^{m_i^{old} - m_i^{new}} O_i + \widetilde{P}_{ij} V_j $。

## epilogue：归一化 + 写回

```cpp
Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
```

`normalize_softmax_lse` 做最后一步：

```cpp
for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = row_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    lse(mi) = ... row_max(mi) * softmax_scale + __logf(sum);   // L = m·scale + log ℓ
    acc_o_rowcol(mi, ni) *= inv_sum;                            // O = Õ / ℓ
}
```

- 把 $ \widetilde{O} $ 除以累计的 $ \ell $（`inv_sum`），得到真正的 $ O $。
- 同时算出 logsumexp $ L = m \cdot \mathrm{scale} + \log \ell $，存给反向用。注意这里用的是 `row_max * softmax_scale`（未做 $ \log_2 e $ 换算），因为 `row_max` 存的就是 $ S $ 的原始尺度上的最大值，而 `scores` 里的指数是 $ \mathrm{exp2}(S\cdot s\cdot\log_2 e - m\cdot s\cdot\log_2 e) $（$ s $ 即 `softmax_scale`）；两者合起来，$ L $ 记录的是"自然对数尺度"的 $ \log \sum e^{S\cdot\mathrm{scale}} $，反向下一次 $ e^{S-L} $ 就对得上。
- `inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum`：整行被掩码掉时 $ \ell = 0 $，直接取倒数会得到 inf/NaN，所以退化为 1，同时把该行的 `lse` 写成 `INFINITY`（`Split` 路径写 `-INFINITY`）当哨兵。这样"空行"输出 0 而不是 NaN。
- dropout 时乘 `rp_dropout`，即 FA1 提到的"rescale at the end"。

最后写回 HBM：把 `acc_o` 从寄存器转到 smem（`smem_tiled_copy_O`），再 `copy` 到 `gO`。LSE 单独写：`get_lse_tile` 定位 `softmax_lse_ptr` 的第 $ (b, h, m) $ 块。

## 一个能跑的等价实现

用 Python 把上面主循环的数学等价写出来，验证累计逻辑（FP16 舍入不模拟，只验online softmax）：

```python
import numpy as np
rng = np.random.default_rng(1)
N, d = 8, 64
Q = rng.normal(size=(N, d)); K = rng.normal(size=(N, d)); V = rng.normal(size=(N, d))

def flashattn_forward(Q, K, V, Br=4, Bc=4):
    O_tilde = np.zeros((Q.shape[0], d)); m = np.full(Q.shape[0], -np.inf); l = np.zeros(Q.shape[0])
    for j in range(0, K.shape[0], Bc):     # 列块
        Kj, Vj = K[j:j+Bc], V[j:j+Bc]
        for i in range(0, Q.shape[0], Br): # 行块
            Qi = Q[i:i+Br]
            S = Qi @ Kj.T
            m_new = np.maximum(m[i:i+Br], S.max(axis=1))
            P = np.exp(S - m_new[:, None])
            scale = np.exp(m[i:i+Br] - m_new)
            O_tilde[i:i+Br] = scale[:, None] * O_tilde[i:i+Br] + P @ Vj
            l[i:i+Br] = scale * l[i:i+Br] + P.sum(axis=1)
            m[i:i+Br] = m_new
    return O_tilde / l[:, None]

def attention_ref(Q, K, V):
    S = Q @ K.T; M = S.max(axis=1, keepdims=True)
    P = np.exp(S - M); P /= P.sum(axis=1, keepdims=True)
    return P @ V

print("equal:", np.allclose(flashattn_forward(Q, K, V), attention_ref(Q, K, V)))
```

`O_tilde` 对应内核里的 `acc_o`（未归一化），`l`、`m` 对应 `row_sum`、`row_max`。`scale` 就是 $ \mathrm{exp2f}\big((m_{\text{old}} - m_{\text{new}})\cdot\log_2 e\big) $。运行结果是 `equal: True`（NumPy 1.26.4，与站点内置运行器同版本）。

## 小结

一行行剥下来，forward kernel 就是：**从 HBM 拷块进 smem → $ Q K^\top $ 张量核 → online softmax（$ m, \ell $ 更新 + 重缩放 $ \widetilde{O} $）→ $ P \cdot V $ 张量核 → 最后除 $ \ell $、写 $ O $ 和 $ L $。** 全程只在进/出 kernel 时碰一次 HBM（Q/K/V 进、O/L 出），中间的 $ S, P, \widetilde{O} $ 都留在 SRAM 和寄存器里。

## Reference

- flash-attention 仓库（csrc/flash_attn/src/flash_fwd_kernel.h、softmax.h）：<https://github.com/Dao-AILab/flash-attention>
- CUTLASS cuTe（tensor/mma 抽象）：<https://github.com/NVIDIA/cutlass>
- FlashAttention 论文算法（forward）：<https://arxiv.org/abs/2205.14135>
