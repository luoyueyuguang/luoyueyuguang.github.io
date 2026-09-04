为什么 `flash-attention` 仓库里有**几百个 `.cu` 文件**、每个长得都一样？因为 FA 的内核是一个**模板**，而它的"变体空间"是硬件（架构）× 精度（dtype）× head 维 × 特性（causal/varlen/paged/GQA/…）的**笛卡尔积**。每个变体都要实例化一份，才能在编译期把那些分支消除掉、吃到最紧的指令。这一篇专门厘清这套"变体"怎么组织。

## 模板参数：变体从哪来

`flash_fwd_kernel.h` 是 `compute_attn_1rowblock<...>` 这样一个模板：

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap,
         bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(...)
```

`Is_causal`、`Is_local`、`Is_softcap` 等都是 **`constexpr bool`**，编译器在实例化时就定型；`Kernel_traits` 把"用什么 dtype、什么 block 大小、什么 MMA atom"打包。所以每个组合都生成一份完全特化的 SASS，**运行时没有任何 `if` 分支**。

变体空间大致是：

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 架构 | sm80 / sm86–89 / sm90 / sm100 / sm120 | Ampere → Hopper → Blackwell，硬件能力不同 |
| dtype | fp16 / bf16 / fp8(e4m3 / e5m2) / 更高 | tensor core 输入精度 |
| head dim | 32 / 64 / 96 / 128 / 192 / 256 | 非倍数要 padding（`Is_even_K`） |
| 特性 | causal / local / varlen / paged / GQA / split-KV / softcap / alibi / append-KV / block-sparse / MLA | 每个加一个模板标志或预处理 |

## 架构变体：从 Ampere 到 Blackwell

同一份数学，硬件不同就得换实现：

- **sm80（Ampere）**：`csrc/flash_attn/src/flash_fwd_kernel.h`。累加器在**寄存器**，用 `cp.async` 搬 tile，`mma.sync` 张量核。
- **sm90（Hopper）**：`hopper/flash_fwd_kernel_sm90.h`、`mainloop_fwd_sm90_tma_gmma_ws.hpp`。用 **TMA** 批量拷贝、**WGMMA** 异步张量核、**warp specialization**。P 走 smem/寄存器，累加器仍在寄存器。
- **sm100/sm120（Blackwell）**：`flash_attn/cute/flash_fwd_sm100.py`、`flash_fwd_sm120.py`（CuTe-DSL）。累加器在 **TMEM**、2-CTA MMA、`tcgen05`。完全换一套抽象。

同一型号的"小改"也有：sm86/sm89（Ampere 消费级）、sm103（B300，FA4 里 `is_sm103` 会**关掉 exp2 模拟**，因为 B300 有原生快 exp2）。

## dtype 变体

实例化命名里直接体现：`flash_fwd_hdim128_{fp16|bf16|e4m3|e5m2}_sm90.cu`。FA4 还支持**更高精度**（在低精度核上模拟），这是 [[learning/flash-attention/08-flashattention4|FA4]] 提到的。dtype 决定：

- 喂给 MMA 的操作数位宽（16-bit vs 8-bit）。
- 是否要额外 descaled（FP8 的块 scale，`ptr_q_descale` 等）。
- FA4 里是否开 `ex2_emu`（FP8 更吃 MUFU，模拟更激进）。前面看 `flash_fwd_sm100.py` 的 `__config` 就是按 `(use_2cta, causal, head_dim, is_sm103)` 查表，FP8 单独一份表。

## head-dim 变体：padding

MMA tile 固定是 16 的倍数（`m16n8k16`），但 head dim 不一定（如 RoPE 后顶 `d=40`、DeepSeek `d=192`）。所以：
- 不为 `Is_even_K` 时，多出来的列用**predication**（`tQpQ(k) = ... < params.d`）跳过，读越界当作 0/不写。
- hdim 192/256 是专门的 `hdim192` / `hdim256` 变体（FA4 里 `flash_fwd_sm100.py` 的 hd256 配置、`slice_dQKV_Mma` 等），因为太大了得拆。

## 特性变体

每个特性都对应一套模板参数 + 预处理：

| 特性 | 模板/预处理 | kernel 里的落点 |
| --- | --- | --- |
| **causal** | `Is_causal` | 跳过一半列块（`n_block_min/max`），只对最后一个块加 mask |
| **local window** | `Is_local` + window size | 只算 `[l, r]` 窗口内的块 |
| **varlen** | `cu_seqlens_q/k` | 每条序列长度不同，动态调度 |
| **paged KV** | `block_table` / `PagedKV` | `paged_kv.h` 按页表索引，decode 时不连续 |
| **GQA / MQA** | `PackGQA` / `h_h_k_ratio` | `pack_gqa.h` 打包同组 head，`bidh / h_h_k_ratio` 定位 KV head |
| **split-KV** | `Split` / `num_splits` | 每个 CTA 只算一段 KV，出 partial O/L，再 combine |
| **softcap** | `Has_softcap` | `softcap_val`，`tanh` 缩放 |
| **alibi** | `Has_alibi` | `alibi.h` 加位置偏置 |
| **append-KV** | `AppendKV` | 新的 KV 块追加到 cache |
| **block-sparse** | block mask | [[learning/flash-attention/16-block-sparse|跳过零块]] |
| **MLA** | 单独的 cute 内核 | [[learning/flash-attention/13-mla|latent KV]] |

## 实例化文件：一个 `.cu` 一个组合

`csrc/flash_attn/src/` 下就这样：

```cpp
// flash_fwd_hdim128_fp16_causal_sm80.cu
#include "flash_fwd_kernel.h"
#include "flash_fwd_launch_template.h"
// ... 用 CUDA_ARCH + 模板参数实例化 flash_attn_fwd<...>
```

命名是 `op_hdim{dtype}_{features}_{arch}.cu`。`hopper/instantiations/` 里更细：`flash_fwd_hdimdiff_fp16_split_softcap_sm90.cu`、`flash_fwd_hdimall_bf16_packgqa_sm90.cu`……这里 `hdimdiff`/`hdimall` 是"head dim 由 kernel 内分派"的变体（省得为每个 hdim 都编一份），`split`/`softcap`/`packgqa` 是特性。统计一下：`hopper/instantiations/` 有 **451 个**文件，其中 310 个 sm90、140 个 sm80。

**编译时间**就是这样爆炸的。这也是 FA4 改用 CuTe-DSL 的动机之一（[[learning/flash-attention/08-flashattention4|FA4]]：C++ 模板要预编译几百个、fwd 55s，CuTe-DSL JIT 降到 2.5s）。

而且现在这套 CuTe-DSL 打法已经收敛成独立的 `flash-attn-4` 发行版（`pip install flash-attn-4`），`flash_attn/cute/interface.py` 里把 sm80/sm90/sm100/sm120 的 forward/backward、MLA forward/backward、combine、block-sparse 全部从一个包分派出去——对 Blackwell 而言，C++ 那套"一个 `.cu` 一个实例化"的爆炸被换成了 JIT 编译的单一 Python 包，`interface.py` 开头就写着 "[2025-07-04] Version in Cute-DSL, for Hopper and Blackwell"。

## 运行时怎么选

launcher（`flash_fwd_launch_template.h` 的 `run_flash_fwd`）在运行时按 `(head_dim, dtype, seqlen, causal, varlen, paged, ...)` **switch** 到对应的实例化函数。这就是一个巨大的 `if/switch` 分派表。编译期把分支消掉，运行期只跑一种。

## split-KV 的 combine 内核

split-KV 时（长序列 / 小 batch，KV 拆给多个 CTA），每个 CTA 只算一段 KV，产出**partial $ O $ 和 partial logsumexp $ L $**。`flash_fwd_combine_kernel.h` 把它合并：

```cpp
using ShapeOPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;   // (seqlen, d, num_splits, head, batch)
using ShapeLSEPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t>;          // (seqlen, num_splits, head, batch)
```

合并公式（online softmax 的跨 CTA 版）：

$$
L_{\text{final}} = \log\sum_s e^{L_s}, \qquad
O = \sum_s e^{L_s - L_{\text{final}}}\, O_s
$$

$O_s$ 是第 `s` 个 split 的未归一化累积，$L_s$ 是它的 logsumexp。combine 核读各 split 的 partial，做 logsumexp 合并再归一化。**这就是 [[learning/flash-attention/09-flashattention4-kernel|FA4]] 里 `flash_fwd_combine` 的对应，也是 split-KV 的收尾。**

## 一句话

FA 的"变体"是**编译期模板参数的笛卡尔积**：架构 × dtype × head dim × 特性。每个组合实例化成一个 `.cu`，launcher 在运行期 switch 选一个，所以 SASS 里零运行时分支。代价是编译时间爆炸（FA4 因此改 CuTe-DSL）。split-KV 时主核出 partial O/L，再由 `flash_fwd_combine_kernel.h` 做跨 CTA 的 logsumexp 合并。

## Reference

- flash-attention 仓库（csrc/flash_attn/src/*.cu、hopper/instantiations/、flash_attn/cute/interface.py）：<https://github.com/Dao-AILab/flash-attention>
- `flash-attn-4` PyPI 发行版（CuTe-DSL，sm80–sm120 单包分派）：<https://pypi.org/project/flash-attn-4/>
- CUTLASS cuTe（模板化 MMA / tile）：<https://github.com/NVIDIA/cutlass>
- FlashAttention-4（CuTe-DSL、JIT 编译）：<https://arxiv.org/abs/2603.05451>
