FlashAttention 系列有四篇正式论文：`FlashAttention`（NeurIPS 2022）、`FlashAttention-2`（2023）、`FlashAttention-3`（NeurIPS 2024）、`FlashAttention-4`（2026，arXiv 2603.05451）。加上一篇被它反复引用的前驱，Rabe 和 Staats 的 `Self-attention Does Not Need $O(n^2)$ Memory`。

这一整套技术解决的只有一句话：attention 是 memory-bound 的，你算得再快也没用，数据搬不动。

> **FlashAttention 的核心不是发明了新的数学，而是把已有的 online softmax + 分块，重构成了一个 IO-aware 的 CUDA kernel，让 attention 在 GPU 上不再被 HBM 带宽卡死。**

## 为什么 attention 慢

先看 attention 的标准做法。给定 `$ Q, K, V \in \mathbb{R}^{N \times d} $`，输出 `$ O = \mathrm{softmax}(QK^\top) V $`。朴素实现把它拆成三步，每一步都在 HBM 读写：

$$
S = Q K^\top, \qquad P = \mathrm{softmax}(S), \qquad O = P V
$$

第 1 步把 `$ S $` 写进 HBM，第 2 步把 `$ S $` 读出来算出 `$ P $` 再写回去，第 3 步把 `$ P $` 读出来乘 `$ V $`。`$ S $` 和 `$ P $` 都是 `$ N \times N $` 的矩阵，所以**显存占用是 `$ O(N^2) $`，HBM 读写也是 `$ O(N^2) $` 次**。

问题就在这。attention 大部分操作是 reduction（softmax、sum），属于 memory-bound；矩阵乘那一点计算量反而被吞没了。序列一长，不是算力不够，是带宽不够。

## GPU 的存储层次

一块 GPU 有多个层次的存储，越小越快：

| 层次 | 大小（A100 为例） | 带宽 | 说明 |
| --- | --- | --- | --- |
| HBM（高带宽内存） | 40–80 GB | 1.5–2.0 TB/s | 大，慢 |
| 片上 SRAM（共享内存） | 每个 SM 192 KB | 约 19 TB/s | 每个 SM 一块，快一个数量级 |
| 寄存器 | 每个 SM 256 KB | 最快 | 一个线程私有 |

A100 有 108 个 SM。SRAM 比 HBM 快约一个数量级，但每个 SM 只有 192 KB。**attention 慢的原因，就是数据在 HBM 和 SRAM 之间来回倒腾，倒腾的次数太多。**

## 计算强度与两种瓶颈

用**算术强度**（arithmetic intensity，FLOPs / 字节）区分：

- 计算密集（compute-bound）：算术操作多，HBM 访问少。典型是内维很大的矩阵乘。
- 内存密集（memory-bound）：HBM 访问多，算术操作少。典型是 elementwise 和 reduction，softmax 就是。

attention 属于后者。所以**消除 attention 的瓶颈，关键是减少 HBM 读写，而不是减少 FLOPs。**

## FlashAttention 的三件套

FA1 用了三个技巧，把 HBM 访问从 `$ O(N^2) $` 压到次二次：

1. **分块（tiling）**：把 `$ Q, K, V $` 切成 block，一次只把一块搬进 SRAM。
2. **在线 softmax（online softmax）**：softmax 是逐行做的，分母 `$ \ell $` 和最大值 `$ m $` 可以一边算一边更新，不用等整行算完。
3. **重计算（recomputation）**：反向传播不存 `$ S, P $` 这两个 `$ N \times N $` 矩阵，而是存输出 `$ O $` 和统计量 `$ L $`（logsumexp），反向时在 SRAM 里重新算出 `$ S, P $`。

这三个都不新。分块是 GEMM 的常规操作，在线 softmax 来自 Milakov、Rabe 等人，重计算就是梯度检查点。FA1 的特殊之处是把它们揉进一个 kernel，让**整个 forward / backward 各跑一个 CUDA kernel，中间不落 HBM**（除了存 `$ O $` 和 `$ L $`）。

![FlashAttention 分块：一个 thread block 处理一个 Q 行块，沿 K/V 列块走一遍](/learning/assets/fa-tiling.svg)

## 系列地图

这套技术从 Ampere（FA1/FA2）到 Hopper（FA3），再到 Blackwell（FA4），每篇侧重不同：

| 文章 | 内容 |
| --- | --- |
| [[learning/flash-attention/02-online-softmax|在线 softmax 与分块]] | FA1 的算法本体：online softmax 的数学、分块、IO 复杂度；以及前驱 Rabe & Staats |
| [[learning/flash-attention/03-forward-kernel|前向内核逐行读]] | `flash_fwd_kernel.h`：HBM→SRAM 拷贝、QK^T、online softmax、P·V、写回 |
| [[learning/flash-attention/04-backward-kernel|反向内核逐行读]] | `flash_bwd_kernel.h`：怎么用 `$ L $` 重算 `$ P $`，以及 dQ/dK/dV 的推导 |
| [[learning/flash-attention/05-flashattention2|FlashAttention-2]] | 两个算法微调 + 序列维并行 + warp 内分工，把 FA1 的算力利用率翻倍 |
| [[learning/flash-attention/06-flashattention3|FlashAttention-3]] | Hopper 上：warp specialization、TMA、pingpong 调度、2 级流水线、FP8 块量化 |
| [[learning/flash-attention/07-flashattention3-kernel|FA3 前向内核逐行读]] | `mainloop_fwd_sm90_tma_gmma_ws.hpp`：TMA 的 producer、`fwd_step` 的 GEMM-softmax 交错、FP8 permute |
| [[learning/flash-attention/12-flashattention3-bwd-kernel|FA3 反向内核逐行读]] | `mainloop_bwd_sm90_tma_gmma_ws.hpp`：`tiled_mma_SdP` 复用、`$ P=e^{S-L} $`、dQ 原子、hdim256 切片 |
| [[learning/flash-attention/08-flashattention4|FlashAttention-4]] | Blackwell 上：非对称硬件缩放、指数模拟、条件缩放、TMEM + 2-CTA MMA、CuTe-DSL |
| [[learning/flash-attention/09-flashattention4-kernel|FA4 前向内核逐行读]] | `flash_fwd_sm100.py`：TMEM 的 S/P 布局、`ld.red` 行最大、`apply_exp2_convert`、correction warpgroup |
| [[learning/flash-attention/10-flashattention4-bwd-kernel|FA4 反向内核逐行读]] | `flash_bwd_sm100.py`：5 个 MMA 的 TMEM 共享、2-CTA + DSMEM 归约、确定性信号量锁 |
| [[learning/flash-attention/11-triton-reference|Triton 参考实现]] | `flash_attn_triton_og.py`：只用 `tl.dot`/`tl.exp` 的 30 行前向，附 FA1 vs FA2 方案对比 |
| [[learning/flash-attention/13-mla|MLA：压缩 KV 的 Latent Attention]] | **扩展**（非 FlashAttention 论文）：DeepSeek-V2/V3 把 K/V 压成 latent `$ c^{KV} $`，absorbed 技巧不物化大 K/V；FA 仓库用 CuTe-DSL 实现 |
| [[learning/flash-attention/14-launch-scheduling|调度与启动层]] | `flash_fwd_launch_template.h`、`tile_scheduler.hpp`、`heuristics.h`：grid 形状、FastDivmod 的 CTA→work、split-KV/pack-GQA 启发式、因果 tile 重排 |
| [[learning/flash-attention/15-mla-bwd-kernel|MLA 反向内核]] | `flash_bwd_mla_*`：主核算 dS/dV，`dQdQvGemmKernel` 算 dS·K 与 dS·V，`dKGemmKernel` 算 dS^T·Q |
| [[learning/flash-attention/16-block-sparse|Block-sparse FlashAttention]] | FA1 论文另一半贡献：按块跳过零块，IO 复杂度 ×稀疏度 `$ s $`，butterfly 模式，LRA 2.8× |
| [[learning/flash-attention/17-variants|变体与实例化]] | 几百个 `.cu` 是"架构 × dtype × head dim × 特性"的笛卡尔积；split-KV 的 `flash_fwd_combine_kernel.h` 做跨 CTA logsumexp 合并 |

先给一张符号表，后面反复用：

| 符号 | 含义 |
| --- | --- |
| `$ N $` | 序列长度 |
| `$ d $` | head 维度（head dim） |
| `$ H $` | head 数 |
| `$ M $` | 片上 SRAM 大小 |
| `$ B_r $` | query row block size |
| `$ B_c $` | key/value column block size |
| `$ T_r = \lceil N/B_r \rceil $` | row block 数 |
| `$ T_c = \lceil N/B_c \rceil $` | column block 数 |
| `$ S_{ij} $` | 第 `$ i $` 行块和第 `$ j $` 列块的分数矩阵 |
| `$ m_i $` | 第 `$ i $` 行块的当前行最大值 |
| `$ \ell_i $` | 第 `$ i $` 行块的指数和 |
| `$ L_i $` | logsumexp `$= m_i + \log \ell_i $` |
| `$ \tilde{O}_i $` | 未除 `$ \ell $` 的输出累计 |

## 几个数字

先摆出关键结果，感受一下量级：

- FA1 在 A100 上，训练 GPT-2（seq len 1K）比最优 baseline 快约 `3×`，BERT-large（512）端到端快 `15%`。
- FA1 把 `$ S, P $` 的 `$ O(N^2) $` 显存降到 `$ O(N) $`（多存 `$ O $` 和统计量 `$ (m, \ell) $`，FA2 起合并为一个 `$ L $`），这是长序列能跑起来的前提。
- FA2 是纯速度优化：FA1 的 forward 只到 30–50%、backward 只有 25–35% 的理论峰值算力，FA2 提到 forward 最高 73%、backward 最高 63%，训练 GPT 式模型到 225 TFLOPs/s（72% 模型 FLOPs 利用率）。
- FA3 是 Hopper 优化 + FP8。H100 上 FA2 只有 35% 利用率，FA3 的 FP16 到 740 TFLOPs/s（75%），FP8 接近 1.2 PFLOPs/s。
- FA4 是 Blackwell 优化。B200 上 BF16 到 1613 TFLOPs/s（71%），比 cuDNN 9.13 快 1.3×、比 Triton 快 2.7×；整个 kernel 用 CuTe-DSL（Python）写，编译快 22–32×。

## Reference

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（arXiv:2307.08691）：<https://arxiv.org/abs/2307.08691>
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- Self-attention Does Not Need $O(n^2)$ Memory（arXiv:2112.05682）：<https://arxiv.org/abs/2112.05682>
- 官方代码：<https://github.com/Dao-AILab/flash-attention>
- Online normalizer calculation for softmax（Milakov & Gimelshein, 2018）：<https://arxiv.org/abs/1805.02867>
