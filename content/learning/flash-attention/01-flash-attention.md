FlashAttention 系列有四篇正式论文：`FlashAttention`（NeurIPS 2022）、`FlashAttention-2`（2023）、`FlashAttention-3`（NeurIPS 2024）、`FlashAttention-4`（2026，arXiv 2603.05451）。除了这四篇外，被引用的Rabe 和 Staats 的 *Self-attention Does Not Need $O(n^2)$ Memory* 也值得重视。

flash attention主要是解决memory bound的问题。在不同代际的nv上有不同的做法。

> **FlashAttention 并没有发明新的数学算法，而是把已有的 online softmax + 分块结合成了一个 IO-aware 的 CUDA kernel，让 attention 在 GPU 上不再被 HBM 带宽bound。**

## 为什么 attention 慢

先看 attention 的标准做法。给定 $ Q, K, V \in \mathbb{R}^{N \times d} $，输出 $ O = \mathrm{softmax}(QK^\top) V $。朴素实现把它拆成三步，每一步都在 HBM 读写：

$$
S = Q K^\top, \qquad P = \mathrm{softmax}(S), \qquad O = P V
$$

1. 把 $ S $ 写进 HBM；
2. 把 $ S $ 读出来算出 $ P $ 再写回去；
3. 把 $ P $ 读出来乘 $ V $。

$ S $ 和 $ P $ 都是 $ N \times N $ 的矩阵，所以**显存占用是 $ O(N^2) $，HBM 读写也是 $ O(N^2) $ 次**。

![FlashAttention 论文 Figure 1（左）：朴素实现会把 N×N 的注意力矩阵（虚线框，即 S、P）物化到较慢的 HBM；FlashAttention 用分块把它们留在片上 SRAM，右图是 GPT-2 上的加速](/learning/assets/fa-fig1.png)

> 图源：Dao-AILab《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（arXiv:2205.14135）Figure 1

问题就在这。attention 大部分操作是 reduction（softmax、sum），属于 memory-bound；矩阵乘那一点计算量反而被吞没了。长序列下 memory bound 更为明显。

## GPU 的存储层次

从 SM（streaming multiprocessor）往外看，从寄存器 → 共享内存 → HBM，访存延迟越来越高。

![GPU 存储层次：寄存器（线程私有，最快）→ 共享内存（block 共享）→ HBM（grid 共享，最慢）](/learning/assets/gpu-memory-hierarchy.svg)

> 自绘示意图

- **寄存器（register）**：**线程私有**，延迟最低。A100 每个 SM 的寄存器文件共 256 KB（64K 个 32 位寄存器），但被该 SM 上所有线程瓜分——单线程最多约 255 个寄存器（≈1 KB），**凑不成一整块可共享的缓冲**。
- **共享内存（shared memory / SRAM）**：**block 内共享**、整块可寻址。A100 每个 SM 约 164 KB（L1+共享共 192 KB），带宽约 19 TB/s。**attention 分块要存的 $ S $、$ P $、$ O $ 就放在这里。**
- **HBM（全局内存）**：**grid 级共享**，40–80 GB，1.5–2.0 TB/s，最慢。

再说一遍容易混的一点：register 是**按线程私有**的，不是整个 block 都能访问的缓冲；真正能当一整块共享缓冲的是 shared memory。

FA 的 kernel 把分块的 $ S $、$ P $、$ O $ 全留在共享内存，只在进出 kernel 时碰一次 HBM。于是瓶颈落在 HBM 带宽上——这就是 attention 慢、memory-bound 的来源。

## Roofline 模型：算力与带宽两种瓶颈

用**算术强度**（arithmetic intensity，$ I = \text{FLOPs} / \text{字节} $）衡量一次操作里算力和内存的比例。一个平台有两个天花板：

- **计算峰值** $ R_{\text{peak}} $：每秒最多做多少 FLOPs。
- **内存带宽** $ B_{\text{peak}} $：每秒最多读写多少字节。

任何 kernel 都可能被这两者中较小的那个卡住：

$$
\text{performance} = \min\left(R_{\text{peak}},\; B_{\text{peak}} \times I\right)
$$

$ I $ 大时被计算峰值卡住（compute-bound），$ I $ 小时被带宽卡住（memory-bound）。两条线的交点叫 **ridge point**：
$$ 
I_{\text{ridge}} = R_{\text{peak}} / B_{\text{peak}} 
$$
A100（80GB SXM）上 BF16 张量核的稠密峰值约 $ \pi = 312 $ TFLOPS、HBM 峰值约 $ \beta = 2.0 $ TB/s，所以 $ I_{\text{ridge}} = 312 / 2.0 \approx 156 $ FLOPs/byte。

attention 落在左侧。head dim $ d $ 不大（几十），标准实现又把 $ N \times N $ 的 $ S $、$ P $ 写进 HBM 再读出来：一行 query 花 $ 4Nd $ FLOPs（两次 $ N \times d $ 的 GEMM），却要搬 $ 8N $ 字节（$ S $、$ P $ 各写一次读一次，fp16），算术强度只有 $ \approx d/2 $（$ d = 64 $ 时约 32）——远低于 ridge。所以它是 memory-bound：**瓶颈是 HBM 带宽，不是算力。**

FlashAttention 把 $ S $、$ P $ 留在 SRAM，HBM 只搬 $ Q,K,V,O $。但流量**并没有降到 $ \Theta(Nd) $**：分块之后每个块对 $ (i,j) $ 都要重新读写一次 $ Q_i $、$ O_i $（换个说法就是每个 $ Q $ 行块都得把整条 $ K,V $ 扫一遍），块对总数是 $ T_r T_c $，于是 HBM 访问是 $ \Theta(N^2 d^2 M^{-1}) $ 次——论文 Theorem 2 是按**元素个数**数访问次数的，另外还有 $ \Theta(Nd) $ 的输入输出项。按这个阶算算术强度：$ 4N^2 d $ FLOPs 对 $ 2N^2 d^2 M^{-1} $ 字节（每个 fp16 元素 2 字节），约 $ 2M/d $——只由 SRAM 大小 $ M $ 和 head dim $ d $ 决定，与 $ N $ 无关；A100 上 $ M \approx 10^5 $ 个 fp16 元素（192 KB）、$ d = 64 $，就是三千多 FLOPs/byte，已经越过 ridge。

但这个阶要 $ N $ 足够大才成立。在 $ N = 1024 $、$ d = 64 $ 这类常用规模下，$ \Theta(Nd) $ 那一项和 $ N^2d^2M^{-1} $ 同量级，实测远不到三千：论文 Figure 2（左）那张表的数字是 75.2 GFLOPs 对 4.4 GB，$ I \approx 17 $；标准实现同表是 66.6 GFLOPs 对 40.3 GB，$ I \approx 1.65 $——比上面按单行 forward 估的 32 低得多，因为那是 forward + backward 的端到端平均，反向还要多搬 $ dO $、$ dS $、$ dP $ 好几个 $ N\times N $ 张量。抬了一个数量级，但仍低于 156。

注意 roofline 示意图画的是**渐近意义**上的位置（分块把 $ I $ 抬过 ridge）；实测的 FA1 仍落在 ridge 左边——这也正是 FA2 继续死磕算力利用率的原因。

所以**消除 attention 的瓶颈，关键是减少 HBM 读写、提高算术强度，而不是减少 FLOPs。**

![Roofline：朴素 attention 落在memory bound，FlashAttention 提高算术强度后进入compute bound](/learning/assets/roofline.svg)

> 自绘示意图

## FlashAttention 的三件套

FA1 用了三个技巧，关键是**不让 $ N\times N $ 的 $ S $、$ P $ 落回 HBM**。结果是：

- 显存从 $ O(N^2) $ 降到 $ O(N) $；
- HBM 读写的**阶**仍是 $ O(N^2) $，只是**常数**从约 $ 4 $（标准做法写、读 $ S,P $ 各一次）降到约 $ d^2/M $（$ M $ 为片上 SRAM 大小，$ d^2/M\ll1 $）：理论上少约 $ 4M/d^2 $ 倍，GPT-2 medium 实测 40.3 GB → 4.4 GB（约 9×）。

三个技巧是：

1. **分块（tiling）**：把 $ Q, K, V $ 切成 block，一次只把一块搬进 SRAM。
2. **online softmax**：softmax 是逐行做的，分母 $ \ell $ 和最大值 $ m $ 可以一边算一边更新，不用等整行算完。
3. **重计算（recomputation）**：反向传播不存 $ S, P $ 这两个 $ N \times N $ 矩阵，而是存输出 $ O $ 和统计量 $ m, \ell $（FA2 起合并成一个 $ L $，见 [[learning/flash-attention/05-backward-kernel|反向内核]]），反向时在 SRAM 里重新算出 $ S, P $。

这些想法都不是fa原生提出的，但fa将他们组合了起来。分块是 GEMM 的常规操作，online softmax 来自 Milakov、Rabe 等人，重计算就是梯度检查点。FA1 的特殊之处是把它们揉进一个 kernel，让**整个 forward / backward 各跑一个 CUDA kernel，中间尽量不落 HBM**（除了存 $ O $ 和统计量 $ m, \ell $）。

![FlashAttention 论文 Figure 1（左）：K/V 按块拷进 SRAM（Copy Block to SRAM），在片上 Compute Block on SRAM 里算，输出回 HBM](/learning/assets/fa-fig1.png)

> 图源：Dao-AILab《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（arXiv:2205.14135）Figure 1

## 系列地图

这套技术从 Ampere（FA1/FA2）到 Hopper（FA3），再到 Blackwell（FA4），每篇侧重不同。作者本人才疏学浅，且大量使用了ai辅助，如果出现了问题，您可以在仓库提出 [issue](https://github.com/luoyueyuguang/luoyueyuguang.github.io/issues) 或 [email to me](mailto:luoyuelight@outlook.com)。

| 文章 | 内容 |
| --- | --- |
| [[learning/flash-attention/02-online-softmax|online softmax 与分块]] | FA1所使用的算法：online softmax 的数学、分块、IO aware |
| [[learning/flash-attention/03-triton-reference|Triton 参考实现]] | `flash_attn_triton_og.py`：只用 `tl.dot`/`tl.max`/`tl.exp` 的 70 行前向（核心主循环 30 多行），附 FA1 vs FA2 方案对比 |
| [[learning/flash-attention/04-forward-kernel|前向kernel代码逐行读]] | `flash_fwd_kernel.h`：HBM→SRAM 拷贝、QK^T、online softmax、P·V、写回 |
| [[learning/flash-attention/05-backward-kernel|反向kernel代码逐行读]] | `flash_bwd_kernel.h`：怎么用 $ L $ 重算 $ P $，以及 dQ/dK/dV 的推导 |
| [[learning/flash-attention/06-flashattention2|FlashAttention-2]] | 两个算法微调 + 序列维并行(CP) + warp 内分工，把 FA1 的算力利用率翻倍 |
| [[learning/flash-attention/07-flashattention3|FlashAttention-3]] | Hopper 上：warp specialization、TMA、pingpong 调度、2 级流水线、FP8 块量化 |
| [[learning/flash-attention/08-flashattention3-kernel|FA3 前向kernel代码逐行读]] | `mainloop_fwd_sm90_tma_gmma_ws.hpp`：TMA 的 producer、`fwd_step` 的 GEMM-softmax 交错、FP8 permute |
| [[learning/flash-attention/09-flashattention3-bwd-kernel|FA3 反向kernel代码逐行读]] | `mainloop_bwd_sm90_tma_gmma_ws.hpp`：`tiled_mma_SdP` 复用、$ P=e^{S-L} $、dQ 原子、hdim256 切片 |
| [[learning/flash-attention/10-flashattention4|FlashAttention-4]] | Blackwell 上：非对称硬件缩放、指数模拟、条件缩放、TMEM + 2-CTA MMA、CuTe-DSL |
| [[learning/flash-attention/11-flashattention4-kernel|FA4 前向kernel代码逐行读]] | `flash_fwd_sm100.py`：TMEM 的 S/P 布局、`ld.red` 行最大、`apply_exp2_convert`、correction warpgroup |
| [[learning/flash-attention/12-flashattention4-bwd-kernel|FA4 反向kernel代码逐行读]] | `flash_bwd_sm100.py`：5 个 MMA 的 TMEM 共享、2-CTA + DSMEM 归约、确定性信号量锁 |
| [[learning/flash-attention/13-mla|MLA：压缩 KV 的 Latent Attention]] | **扩展**（非 FlashAttention 论文）：DeepSeek-V2/V3 把 K/V 压成 latent $ c^{KV} $，absorbed 技巧不物化大 K/V；FA 仓库用 CuTe-DSL 实现 |
| [[learning/flash-attention/14-launch-scheduling|调度与启动层]] | `flash_fwd_launch_template.h`、`tile_scheduler.hpp`、`heuristics.h`：grid 形状、FastDivmod 的 CTA→work、split-KV/pack-GQA 启发式、因果 tile 重排 |
| [[learning/flash-attention/15-mla-bwd-kernel|MLA 反向kernel]] | `flash_bwd_mla_*`：主核算 dS/dV，`dQdQvGemmKernel` 算 dS·K 与 dS·V，`dKGemmKernel` 算 dS^T·Q |
| [[learning/flash-attention/16-block-sparse|Block-sparse FlashAttention]] | FA1 论文另一半贡献：按块跳过零块，IO 复杂度 ×稀疏度 $ s $，butterfly 模式，LRA 2.8× |
| [[learning/flash-attention/17-variants|变体与实例化]] | 几百个 `.cu` 是"架构 × dtype × head dim × 特性"的笛卡尔积；split-KV 的 `flash_fwd_combine_kernel.h` 做跨 CTA logsumexp 合并 |

先给一张符号表，后面反复用：

| 符号 | 含义 |
| --- | --- |
| $ N $ | 序列长度 |
| $ d $ | head 维度（head dim） |
| $ H $ | head 数 |
| $ M $ | 片上 SRAM 大小 |
| $ B_r $ | query row block size |
| $ B_c $ | key/value column block size |
| $ T_r = \lceil N/B_r \rceil $ | row block 数 |
| $ T_c = \lceil N/B_c \rceil $ | column block 数 |
| $ S_{ij} $ | 第 $ i $ 行块和第 $ j $ 列块的分数矩阵 |
| $ m_i $ | 第 $ i $ 行块的当前行最大值 |
| $ \ell_i $ | 第 $ i $ 行块的指数和 |
| $ L_i $ | logsumexp $= m_i + \log \ell_i $ |
| $ \tilde{O}_i $ | 未除 $ \ell $ 的输出累计 |

## 几个数字

先摆出几个结果，感受一下fa的加速比：

- FA1 在 A100 上训练 GPT-2（seq len 1K）比 HuggingFace 实现快最多 3×、比 Megatron-LM 快最多 1.7×；BERT-large（512）端到端比 MLPerf 1.1 的训练速度记录快 15%（8×A100，10 次平均）。
- FA1 把 $ S, P $ 的 $ O(N^2) $ 显存降到 $ O(N) $（多存 $ O $ 和统计量 $ (m, \ell) $，FA2 起合并为一个 $ L $），这是长序列能跑起来的前提。
- FA2 是纯速度优化：FA1 的 forward 只到 30–50%、backward 只有 25–35% 的理论峰值算力，FA2 提到 forward 最高 73%、backward 最高 63%，训练 GPT 式模型到 225 TFLOPs/s（72% 模型 FLOPs 利用率）。
- FA3 是 Hopper 优化 + FP8。H100 上 FA2 只有 35% 利用率，FA3 的 FP16 到 740 TFLOPs/s（75%），FP8 接近 1.2 PFLOPs/s。
- FA4 是 Blackwell 优化。B200 上 BF16 到 1613 TFLOPs/s（71%），比 cuDNN 9.13 快 1.3×、比 Triton 快 2.7×；整个 kernel 用 CuTe-DSL（Python）写，编译快 20–30×。

## Reference

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（arXiv:2307.08691）：<https://arxiv.org/abs/2307.08691>
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- Self-attention Does Not Need $O(n^2)$ Memory（arXiv:2112.05682）：<https://arxiv.org/abs/2112.05682>
- 官方代码：<https://github.com/Dao-AILab/flash-attention>
- Online normalizer calculation for softmax（Milakov & Gimelshein, 2018）：<https://arxiv.org/abs/1805.02867>
