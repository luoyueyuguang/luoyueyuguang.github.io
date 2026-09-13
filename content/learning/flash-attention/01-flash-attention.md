FlashAttention 系列有四篇正式论文：`FlashAttention`（NeurIPS 2022）、`FlashAttention-2`（2023）、`FlashAttention-3`（NeurIPS 2024）、`FlashAttention-4`（2026，arXiv 2603.05451）。除了这四篇外，Rabe 和 Staats 的 *Self-attention Does Not Need $O(n^2)$ Memory* 也常被引用。

FlashAttention 要解决的是 attention 的 memory bound 问题，不同代际的 NVIDIA GPU 上做法不同。

> **FlashAttention 没有发明新的数学算法，它把已有的 online softmax 和分块组合成一个 IO-aware 的 CUDA kernel，把 attention 的 HBM 读写降了一个数量级。**

## 为什么 attention 慢

先看 attention 的标准做法。真实的张量是四维的：批 $ b $、头数 $ h $、序列长度 $ N $、head dim $ d $，即 $ Q, K, V \in \mathbb{R}^{b \times h \times N \times d} $。论文为了记号干净，按单个 $ (b, h) $ 切片写成 $ \mathbb{R}^{N \times d} $——下面沿用这个约定，只在涉及总量时把 $ b \cdot h $ 乘回来。

单个切片内，输出是 $ O = \mathrm{softmax}(QK^\top) V $。朴素实现把它拆成三步，每一步都在 HBM 读写：

$$
S = Q K^\top, \qquad P = \mathrm{softmax}(S), \qquad O = P V
$$

1. 把 $ S $ 写进 HBM；
2. 把 $ S $ 读出来算出 $ P $ 再写回去；
3. 把 $ P $ 读出来乘 $ V $。

**$ N^2 $ 出现在 $ S $ 和 $ P $ 的最后两维上**，把每步的形状写全就很清楚：

| 步骤 | 运算 | 切片内形状 | 四维形状 | 元素个数 |
| --- | --- | --- | --- | --- |
| 1 | $ S = QK^\top $ | $ N \times N $ | $ b \times h \times N \times N $ | $ b h N^2 $ |
| 2 | $ P = \mathrm{softmax}(S) $ | $ N \times N $ | $ b \times h \times N \times N $ | $ b h N^2 $ |
| 3 | $ O = PV $ | $ N \times d $ | $ b \times h \times N \times d $ | $ bhdN $ |

输入和输出都只有 $ Nd $（每行一个 token），涨到 $ N^2 $ 的只有 $ S $ 和 $ P $ 这两个中间结果。$ b $ 和 $ h $ 不改变形状，只是把同一份切片复制 $ b \cdot h $ 次：**显存占用是 $ O(bhN^2) $，HBM 读写也是同样的量级。**

**FlashAttention 要挡的就是这两个矩阵落到 HBM 上**，做法是分块：一次只取 $ B_r $ 行 query、$ B_c $ 列 key，算 $ B_r \times B_c $ 的小块 $ S_{ij} $、$ P_{ij} $，它们只活在片上（$ B_r $、$ B_c $ 是分块大小，与批大小 $ b $ 无关）。

![Attention 的四个维度：左边标准实现把 b×h×N×N 的 S、P 写进 HBM 再读回；右边 FlashAttention 在单个切片内按 B_r×B_c 分块，Sᵢⱼ、Pᵢⱼ 只留在 SRAM，HBM 只搬 Q,K,V,O](/learning/assets/fa-attn-dims.svg)

> 自绘示意图

![FlashAttention 论文 Figure 1（左）：朴素实现会把 N×N 的注意力矩阵（虚线框，即 S、P）物化到较慢的 HBM；FlashAttention 用分块把它们留在片上 SRAM，右图是 GPT-2 上的加速](/learning/assets/fa-fig1.png)

> 图源：Dao-AILab《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（arXiv:2205.14135）Figure 1

attention 大部分操作是 reduction（softmax、sum），属于 memory-bound，矩阵乘那一点计算量反而被吞没了；序列越长，这个瓶颈越明显。

## GPU 的存储层次

从 SM（streaming multiprocessor）往外看，从寄存器 → 共享内存 → HBM，访存延迟越来越高。

![NVIDIA CUDA 编程指南的存储层次图：寄存器与 local memory（每线程）→ 共享内存（每 block）→ 分布式共享内存（每 cluster）→ 全局内存（整个 grid）](/learning/assets/cuda-memory-hierarchy.png)

> 图源：NVIDIA《CUDA C++ Programming Guide》§2.3 Memory Hierarchy，Figure 6（<https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png>）

- **寄存器（register）**：**线程私有**，延迟最低。A100 每个 SM 的寄存器文件共 256 KB（64K 个 32 位寄存器），但被该 SM 上所有线程瓜分，单线程最多约 255 个寄存器（≈1 KB），**凑不成一整块可共享的缓冲**。
- **共享内存（shared memory / SRAM）**：**block 内共享**、整块可寻址。A100 每个 SM 约 164 KB（L1+共享共 192 KB），带宽约 19 TB/s。**attention 分块要存的 $ S $、$ P $、$ O $ 就放在这里。**
- **HBM（全局内存）**：**grid 级共享**，40–80 GB，1.5–2.0 TB/s，最慢。

register **按线程私有**分配，block 里的其他线程访问不到；真正能当一整块共享缓冲的是 shared memory。

FA 的 kernel 把分块的 $ S $、$ P $、$ O $ 全留在共享内存，只在进出 kernel 时碰一次 HBM。于是瓶颈落在 HBM 带宽上，这就是 attention 慢、memory-bound 的来源。

## Roofline 模型：算力与带宽两种瓶颈

用**算术强度**（arithmetic intensity，$ I = \text{FLOPs} / \text{字节} $）衡量一次操作里算力和内存的比例。一个平台有两个天花板：

- **峰值算力** $ \pi $（compute ceiling）：每秒最多做多少 FLOPs。
- **峰值带宽** $ \beta $（memory ceiling）：每秒最多读写多少字节。

任何 kernel 都可能被这两者中较小的那个卡住：

$$
\text{performance} = \min\left(\pi,\; \beta \times I\right)
$$

$ I $ 大时被计算峰值卡住（compute-bound），$ I $ 小时被带宽卡住（memory-bound）。两条线的交点叫 **ridge point**：
$$ 
I_{\text{ridge}} = \pi / \beta 
$$
A100（80GB SXM）上 BF16 张量核的稠密峰值约 $ \pi = 312 $ TFLOPS、HBM 峰值约 $ \beta = 2.0 $ TB/s，所以 $ I_{\text{ridge}} = 312 / 2.0 \approx 156 $ FLOPs/byte。

attention 落在左侧。head dim $ d $ 不大（几十），标准实现又把 $ N \times N $ 的 $ S $、$ P $ 写进 HBM 再读出来：一行 query 花 $ 4Nd $ FLOPs（两次 $ N \times d $ 的 GEMM），却要搬 $ 8N $ 字节（$ S $、$ P $ 各写一次读一次，fp16），算术强度只有 $ \approx d/2 $（$ d = 64 $ 时约 32），远低于 ridge。所以它是 memory-bound：**瓶颈是 HBM 带宽，不是算力。**

FlashAttention 把 $ S $、$ P $ 留在 SRAM，HBM 只搬 $ Q,K,V,O $，但流量**并没有降到 $ \Theta(Nd) $**：分块之后，每个 $ Q $ 行块还是得把整条 $ K,V $ 扫一遍。下面的账都在**单个 $ (b, h) $ 切片**内算；$ b $、$ h $ 只是把它复制 $ b \cdot h $ 份，FLOPs 和访存同比放大，所以算术强度 $ I $ 与 $ b $、$ h $ 无关（总显存和总时间才要乘上 $ b \cdot h $）。把账拆成三问：

- **块能开多大？** 片上要同时容下 $ Q_i $（$ B_r \times d $）、$ K_j $ 与 $ V_j $（各 $ B_c \times d $）、以及输出累加器 $ O_i $（$ B_r \times d $），合计约 $ 4Bd $ 个元素。所以 $ B \approx M/(4d) $，其中 $ M $ 是片上 SRAM 能放的元素数。
- **块对有多少个？** 外层循环切 $ Q $、内层循环切 $ K,V $，块对总数 $ T_r T_c = (N/B)^2 $。
- **每个块对搬多少？** 内层每前进一格都要读进 $ K_j $ 和 $ V_j $ 两个块，即 $ 2Bd $ 个元素。

三者相乘

$$
T_r T_c \cdot 2Bd = \left(\frac{N}{B}\right)^2 \cdot 2Bd = \frac{2N^2 d}{B} = \frac{8N^2 d^2}{M}
$$

就是论文 Theorem 2 给出的 $ \Theta(N^2 d^2 M^{-1}) $ 次访问（Theorem 2 是按**元素个数**计数的）。除此之外还有 $ \Theta(Nd) $ 的输入输出项。

按这个阶算算术强度：分子是 forward 的 $ 4N^2 d $ FLOPs（$ QK^\top $ 与 $ PV $ 各 $ 2N^2 d $），分母是 $ 2N^2 d^2 M^{-1} $ 字节（每个 fp16 元素 2 字节），相除得

$$
I_\text{asym} = \frac{4N^2 d}{2N^2 d^2 M^{-1}} = \frac{2M}{d}
$$

$ N $ 被约掉了：**这个上限只由 SRAM 大小 $ M $ 和 head dim $ d $ 决定，与序列长度无关。** A100 上 $ M \approx 10^5 $ 个 fp16 元素（192 KB）、$ d = 64 $，代入得三千多 FLOPs/byte，远在 ridge（156）右侧。

但这个阶要 $ N $ 足够大才成立。两项之比约 $ M/(Nd) $，$ N $ 越大它才越小；在常用规模上它还是个 $ O(1) $ 的数。代入 $ N = 1024 $、$ d = 64 $：输入输出项 $ 4Nd \approx 2.6\times10^5 $ 个元素，分块重读项 $ N^2d^2M^{-1} \approx 4.4\times10^4 $ 个元素——**同量级，IO 项反而更大**，所以 $ 2M/d $ 这个上限此时根本用不上。

论文 Figure 2（左）在 GPT-2 medium（$ N = 1024 $、$ d = 64 $、16 head、batch 64）上量到的正是这个结果，而且是 **forward + backward 端到端**的平均：

| 实现 | FLOPs | HBM 流量 | $ I $ |
| --- | --- | --- | --- |
| 标准实现 | 66.6 GFLOPs | 40.3 GB | 1.65 |
| FlashAttention | 75.2 GFLOPs | 4.4 GB | 17 |

注意两处口径差异，别把上面的数字和它们直接比：

- **FLOPs**：FA 反而更高（75.2 > 66.6），因为反向要重算 $ S $、$ P $。换来的收益全在流量上，压掉约 9 倍。
- **口径**：$ I_\text{asym} $ 是渐近上限、单看 forward、且不计 $ \Theta(Nd) $ 项；表里两行是端到端实测，反向还要多搬 $ dO $、$ dS $、$ dP $ 好几个 $ N\times N $ 张量。

所以实测的 $ I \approx 17 $ 比理论上限的三千多差两个数量级。它相对标准实现的 $ 1.65 $ 抬了一个数量级，方向正确，但**仍在 ridge 左边**：瓶颈还在带宽侧，FA1 只做到该点理论峰值的 25–40%。

**消除 attention 的瓶颈，关键是减少 HBM 读写、提高算术强度，而不是减少 FLOPs。**

![Roofline：标准实现与 FlashAttention 的实测强度都落在 ridge 左侧的 memory-bound 区，只有 2M/d 的渐近上限越过 ridge](/learning/assets/roofline.svg)

> 自绘示意图（两个实测点取自 FA1 论文 Figure 2 左：66.6 GFLOPs / 40.3 GB 与 75.2 GFLOPs / 4.4 GB）

## FlashAttention 的三件套

FA1 用了三个技巧，关键是**不让 $ N\times N $ 的 $ S $、$ P $ 落回 HBM**。结果是：

- 显存从 $ O(bhN^2) $ 降到 $ O(bhN) $；
- HBM 读写的**阶**仍是 $ O(N^2) $，只是**常数**从约 $ 4 $（标准做法写、读 $ S,P $ 各一次）降到约 $ d^2/M $（$ M $ 为片上 SRAM 大小，$ d^2/M\ll1 $）：理论上少约 $ 4M/d^2 $ 倍，GPT-2 medium 实测 40.3 GB → 4.4 GB（约 9×）。

三个技巧是：

1. **分块（tiling）**：把 $ Q, K, V $ 切成 block，一次只把一块搬进 SRAM。
2. **online softmax**：softmax 是逐行做的，分母 $ \ell $ 和最大值 $ m $ 可以一边算一边更新，不用等整行算完。
3. **重计算（recomputation）**：反向传播不存 $ S, P $ 这两个 $ N \times N $ 矩阵，而是存输出 $ O $ 和统计量 $ m, \ell $（FA2 起合并成一个 $ L $，见 [[learning/flash-attention/05-backward-kernel|反向内核]]），反向时在 SRAM 里重新算出 $ S, P $。

这些想法都不是 FA1 提出的：分块是 GEMM 的常规操作，online softmax 来自 Milakov、Rabe 等人，重计算就是梯度检查点。FA1 的特殊之处是把它们揉进一个 kernel，让**整个 forward / backward 各跑一个 CUDA kernel，中间尽量不落 HBM**（除了存 $ O $ 和统计量 $ m, \ell $）。

![FlashAttention 论文 Figure 1（左）：K/V 按块拷进 SRAM（Copy Block to SRAM），在片上 Compute Block on SRAM 里算，输出回 HBM](/learning/assets/fa-fig1.png)

> 图源：Dao-AILab《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（arXiv:2205.14135）Figure 1

## 系列地图

这套技术从 Ampere（FA1/FA2）到 Hopper（FA3），再到 Blackwell（FA4），每篇侧重不同。作者本人才疏学浅，且大量使用了 AI 辅助，如果出现了问题，您可以在仓库提出 [issue](https://github.com/luoyueyuguang/luoyueyuguang.github.io/issues) 或 [email to me](mailto:luoyuelight@outlook.com)。

| 文章 | 内容 |
| --- | --- |
| [[learning/flash-attention/02-online-softmax|online softmax 与分块]] | FA1所使用的算法：online softmax 的数学、分块、IO aware |
| [[learning/flash-attention/03-triton-reference|Triton 参考实现]] | `flash_attn_triton_og.py`：只用 `tl.dot`/`tl.max`/`tl.exp` 的 67 行前向主体（核心主循环 33 行），附 FA1 vs FA2 方案对比 |
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
| $ b $ | batch 大小 |
| $ h $ | head 数 |
| $ N $ | 序列长度 |
| $ d $ | head 维度（head dim） |
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

## 从 FA1 到 FA4：速度是怎么涨上去的

FA1 先把 $ S, P $ 的显存从 $ O(bhN^2) $ 降到 $ O(bhN) $（只多存 $ O $ 和统计量，FA2 起合并成一个 $ L $），长序列这才跑得起来。之后三代追的是同一件事：把 attention 的实际效率推到接近 GEMM。三篇论文各留了一张实测图，连起来看就是这条曲线。

### A100：FA1 → FA2

![A100 上 forward+backward 的实测速度：四个子图分别是 causal / 非 causal 与 head dim 64 / 128 的组合，橙色 FlashAttention 是 FA1，紫色 FlashAttention-2 在每个子图里都最高](/learning/assets/fa2-a100-fwd-bwd-speed.png)

> 图源：Tri Dao《FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning》（arXiv:2307.08691）Figure 4（四个子图合为一张）

纵轴是 TFLOPs/s，橙色 FlashAttention 是 FA1、紫色是 FA2。FA1 的成绩：A100 上训练 GPT-2（seq 1K）比 HuggingFace 实现快最多 3×、比 Megatron-LM 快最多 1.7×，BERT-large（512）端到端比 MLPerf 1.1 的训练速度记录快 15%（8×A100，10 次平均）。但 FA2 论文同时给出了一组更说明问题的数字：FA1 的 forward 只到理论峰值的 30–50%、backward 只有 25–35%，而优化良好的 GEMM 能到 80–90%。差距不在内存带宽，在 work partitioning——低占用率、以及多余的共享内存读写。FA2 就是冲着这个去的：forward 提到最高 73%、backward 最高 63%，端到端训练 GPT 式模型到 225 TFLOPs/s（72% 模型 FLOPs 利用率）。

### H100：FA3

![H100 上 FP16/BF16 forward 的实测速度：六个子图是 head dim 64/128/256 与 causal / 非 causal 的组合，逐代对比 FlashAttention-2、cuDNN 与 FlashAttention-3](/learning/assets/fa3-h100-fwd-speed.png)

> 图源：Jay Shah 等《FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision》（arXiv:2407.08608）Figure 5（六个子图合为一张）

换到 Hopper，参照物里多了 cuDNN。同样是 FP16/BF16 的 forward，FA3 把 H100 推到 740 TFLOPs/s（75% 理论峰值），而 FA2 那一代只有约 35%；FP8 接近 1.2 PFLOPs。这一代的收益来自异步（warp specialization + pingpong）和对 FP8 的正确使用。

### B200：FA4

![B200 上 FP16/BF16 forward 的实测 TFLOPS：上为非 causal、下为 causal，head dim 128，参照物是 cuDNN 9.13/9.19、Triton 3.6、Gluon 3.6 与 FA2](/learning/assets/fa4-b200-fwd-tflops.png)

> 图源：Ted Zadouri 等《FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling》（arXiv:2603.05451）Figure 4（原图左右两半改为上下排列）

到 Blackwell，参照物换成了 cuDNN 和 Triton。B200 上 BF16 到 1613 TFLOPs/s（约 71% 理论峰值），比 cuDNN 9.13 快 1.1–1.3×、比 Triton 快 2.1–2.7×。这一代的 kernel 整个用 CuTe-DSL（Python）写成，单核编译时间比 FA3 的 C++ 模板快 20–30×。

三张图的纵轴口径不同（TFLOPs/s 与 TFLOPS），且是三代不同的卡，**不要跨图比绝对值**——能横向看的只有同一张图里各条柱子的相对高低。

## Reference

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（arXiv:2307.08691）：<https://arxiv.org/abs/2307.08691>
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- Self-attention Does Not Need $O(n^2)$ Memory（arXiv:2112.05682）：<https://arxiv.org/abs/2112.05682>
- 官方代码：<https://github.com/Dao-AILab/flash-attention>
- NVIDIA A100 产品规格页（80GB SXM：HBM2e 2039 GB/s、FP32 19.5 TFLOPS、BF16 Tensor Core 312 TFLOPS 稠密）：<https://www.nvidia.com/en-us/data-center/a100/>
- NVIDIA A100 Tensor Core GPU Architecture 白皮书（每 SM 256 KB 寄存器文件、192 KB L1+共享、共享内存可配到 164 KB）：<https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>
- NVIDIA CUDA C++ Programming Guide §2.3 Memory Hierarchy（本文存储层次图的出处，Figure 6）：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2009（$\pi$、$\beta$、ridge point 这组记号的出处；展开见 [[learning/roofline|Roofline 模型]]）：<https://dl.acm.org/doi/10.1145/1498765.1498785>
- Online normalizer calculation for softmax（Milakov & Gimelshein, 2018）：<https://arxiv.org/abs/1805.02867>
