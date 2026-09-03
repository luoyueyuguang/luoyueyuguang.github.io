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

**2. 线程块只负责一个 tile。** 每个 thread block 处理"一个 batch、一个 head、一行块 `$ m $`"的输出。所以读出 `$ gQ $` 用 `local_tile` 切出 `(kBlockM, d)` 这一块，`$ gK $`、`$ gV $` 切成 `(kBlockN, d)` 的第 `$ j $` 列块。

![一个 thread block 的循环：固定 Q_i，遍历 K_j / V_j](/learning/assets/fa-tiling.svg)

**3. MMAP 片段（fragment）与 tile。** 张量核一次 `mma` 处理一个小 tile（如 `m16n8k16`），一个 warp 的 32 个线程分别持有这个 tile 的一部分，这叫 fragment。`partition_fragment_A/B/C` 就是按 mma 的形状把 smem 里的 tile 分给每个线程，得到能喂给 mma 指令的寄存器片段。

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
- `sVt` 是 `V` 的转置视图，因为第二个 GEMM 是 `$ P \cdot V $`，P 在寄存器里按"行主序"排，V 需要按"列"读，转置一下更顺。
- `Share_Q_K_smem` 为真时 `sQ` 和 `sK` 共用同一片内存（因为一个不重叠的时机用）。

## 把 tile 从 HBM 拷进 smem

```cpp
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);   // 源：全局
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);   // 目的：共享
```

这是 `cp.async` 的抽象：`partition_S`/`partition_D` 把"整个 block 的拷贝工作"分给每个线程，发出异步的内存拷贝。真正执行在 prologue 里：

```cpp
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, ...);
```

`tQcQ`、`tQpQ` 是谓词。`tQpQ(k) = ... < params.d` 表示第 `$ k $` 列是否在 `$ d $` 之内，超出 `$ d $` 的列被 mask 掉（`Is_even_K` 为假时，即 head dim 不是 128/64 的倍数）。

**为什么 predicated**：`d` 不一定是 64 的倍数（如 RoPE 的 head dim = 40）。但 tensor core tile 固定是 16 的倍数，所以多出来的列要么填 0、要么不写。这就是 `Is_even_K` 存在的原因。

## GEMM 所需的寄存器片段

```cpp
typename Kernel_traits::TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(tidx);
Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
Tensor tSrK  = thr_mma.partition_fragment_B(sK);
Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);
Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
```

- `tSrQ` 是 A 片段（`$ Q_i $`），`tSrK` 是 B 片段（`$ K_j $`），GEMM 算 `$ S = Q_i K_j^\top $`。
- `tOrVt` 是 `$ V $` 的转置 B 片段，给第二个 GEMM（`$ P \cdot V $`）。
- `acc_o` 是 `$ kBlockM \times d $` 的累加器（fp32），就是那个"未归一化的 `$ \widetilde{O} $`"。

**这一步就体现了 FA2 的 warp 分工。** `tSrQ` 是 A 片段，A 是按 mma 的 M 维分给 warps 的。FA2 把 `$ Q $` 切给不同 warp，每个 warp 算出自己那行块的全部 `$ \widetilde{O} $`（先算 `$ S $` 再算 `$ P \cdot V $`），**warp 之间不需要任何通信**。FA1 则相反，把 `$ K, V $` 切给 warp、`$ Q $` 共享，导致每个 warp 算出的 `$ P \cdot V $` 要写回 smem、同步、再加，这就是"split-K"开销。详见 [[learning/flash-attention/05-flashattention2|FA2]]。

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

`Q` 一旦进了寄存器（`$ B_r \times d $` 也放得下），整个主循环就不再读它，省下每轮重复读 smem 的开销。这只有 FA2 的"Q 切给 warp"才做得出来；FA1 里 `$ Q $` 被所有 warp 共享，不能这么留在寄存器。

## 主循环：S = QK^T

```cpp
for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    FLASH_NAMESPACE::cp_async_wait<0>();     // 等 K/V 拷贝完成
    __syncthreads();
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, ...);
    cute::cp_async_fence();

    FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );
```

- `acc_s` 是 `$ kBlockM \times kBlockN $` 的 fp32 累加器，就是 `$ S_{ij} $`。
- `FLASH_NAMESPACE::gemm` 封装了 `mma.sync.aligned.m16n8k16` 这类指令：`tSrQ`（A，寄存器）乘 `tSrK`（B，smem），结果写进 `acc_s`（C，寄存器）。
- `A_in_regs=Is_Q_in_regs` 告诉它 A 操作数在寄存器里还是 smem 里，省一次 `retile`。

## 在线 softmax + 重缩放 O

之前读的是 `flash_fwd_kernel.h`，softmax 的实现在 `csrc/flash_attn/src/softmax.h` 的 `Softmax::softmax_rescale_o`。这正好对应 [[learning/flash-attention/02-online-softmax|算法]] 的第 11–13 行：

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

1. `reduce_max` 算这一块 `$ S $` 的行最大 `$ \tilde{m} $`，并和已有的 `$ m $` 合并（`zero_init=false` 时做 `max(m, new)`），即算法里的 `$ m_i^{new} = \max(m_i, \tilde{m}_{ij}) $`。
2. `cute::copy(row_max, scores_max_prev)` 保存旧 `$ m_i $`。
3. `scores_scale = exp2f((scores_max_prev - row_max) * softmax_scale_log2)`，就是 `$ e^{m_i - m_i^{new}} $`。**它同时乘到 `$ \ell $` 和 `$ \widetilde{O} $` 上**，这就是 FA2 的"不用对两个项都除以 `$ \ell $`"：直接攒"未缩放"的 `$ \widetilde{O} $`，最后再归一化。
4. `scale_apply_exp2(scores, row_max, ...)`：把 `$ S $` 按新 max 做 `$ \widetilde{P} = e^{S - m} $`，即 `exp2(S * scale - m * scale)`。用 `exp2f` 而不是 `expf` 是技巧：`$ e^x = 2^{x \log_2 e} $`，写成 `exp2f(x * scale)` 编译器能合成 `ffma`（一次乘加），比 `fadd` + `fmul` 各一条省一条指令。
5. `reduce_sum` 把 `$ \widetilde{P} $` 的行和加进 `$ \ell $`。

注意这里用 `exp2f` 而不是 `expf`：H100/A100 上特殊函数单元对 `exp2` 有原生命令，且 `exp2(a·b)` 直接是 `ffma`。这是后面 FA3 说"special function 只有 3.9 TFLOPS"的同款硬件约束。

## 转精度 + 第二个 GEMM：O += P·V

```cpp
Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);   // fp32 -> fp16/bf16
if (Is_dropout) { dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps); }
Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs(rP.layout()));
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

- `convert_type<Element>`：tensor core 的输入是 fp16/bf16，`acc_s` 是 fp32，所以在喂给 `$ P \cdot V $` 前要把 `$ S $`（此时已经是"近似 P"）转成 fp16。这是 attention 唯一的精度损失点之一（softmax 结果存成 fp16）。
- `convert_layout_acc_Aregs`：`acc_s` 作为 C 累加器时，每个线程持有的元素分布，和作为 A 操作数（`$ P $`）时期望的分布不一样。要重排一下，否则 mma 算错。这就是 FA3 里"FP32 累加器布局和 operand A 布局不同"需要在 FP8 用 byte-permute 解决的同一类问题。
- `FLASH_NAMESPACE::gemm_rs` 是"register × shared" GEMM：`$ P $` 在寄存器、`$ V $` 在 smem，结果加进 `acc_o`。`rs` 后缀表示 A 从寄存器（register）、B 从共享（shared）取。

`acc_o` 一直攒的是未归一化的 `$ \widetilde{O} $`，这行就把它和 `$ \widetilde{P} \cdot V_j $` 相加，对应算法里的 `$ O_i \leftarrow e^{m_i^{old} - m_i^{new}} O_i + \widetilde{P}_{ij} V_j $`。

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

- 把 `$ \widetilde{O} $` 除以累计的 `$ \ell $`（`inv_sum`），得到真正的 `$ O $`。
- 同时算出 logsumexp `$ L = m \cdot \mathrm{scale} + \log \ell $`，存给反向用。
- dropout 时乘 `rp_dropout`，即 FA1 提到的"rescale at the end"。

最后写回 HBM：把 `acc_o` 从寄存器转到 smem（`smem_tiled_copy_O`），再 `copy` 到 `gO`。LSE 单独写：`get_lse_tile` 定位 `softmax_lse_ptr` 的第 `$ (b, h, m) $` 块。

## 一个能跑的等价实现

用 Python 把上面主循环的数学等价写出来，验证累计逻辑（FP16 舍入不模拟，只验在线 softmax）：

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

`O_tilde` 对应内核里的 `acc_o`（未归一化），`l`、`m` 对应 `row_sum`、`row_max`。`scale` 就是 `exp2f((m_old - m_new)·log2e)`。

## 小结

一行行剥下来，forward kernel 就是：**从 HBM 拷块进 smem → `$ Q K^\top $` 张量核 → online softmax（`$ m, \ell $` 更新 + 重缩放 `$ \widetilde{O} $`）→ `$ P \cdot V $` 张量核 → 最后除 `$ \ell $`、写 `$ O $` 和 `$ L $`。** 全程只在进/出 kernel 时碰一次 HBM（Q/K/V 进、O/L 出），中间的 `$ S, P, \widetilde{O} $` 都留在 SRAM 和寄存器里。

## Reference

- flash-attention 仓库（csrc/flash_attn/src/flash_fwd_kernel.h、softmax.h）：<https://github.com/Dao-AILab/flash-attention>
- CUTLASS cuTe（tensor/mma 抽象）：<https://github.com/NVIDIA/cutlass>
- FlashAttention 论文算法（forward）：<https://arxiv.org/abs/2205.14135>
