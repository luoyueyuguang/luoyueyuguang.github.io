FA2 在 A100 上已经把 attention 推到接近 GEMM 的效率，但换到 H100 就露馅：FA2 在 H100 上只达到 35% 利用率。原因是 FA2 是为 Ampere 写的，没用上 Hopper 的新硬件。FA3 的三个技术全都围绕 Hopper 的两个新能力：**TMA**（Tensor Memory Accelerator，异步批量拷贝）和 **WGMMA**（warpgroup 矩阵乘，异步 tensor-core GEMM）。

> **FA3 的核心问题是"重叠"。** 数据搬移（TMA）、矩阵乘（WGMMA）、softmax 的指数函数分别用不同的硬件单元，FA2 让它们串行。FA3 用 warp specialization + 双 warpgroup + 多级 SMEM 缓冲，把这三件事错开，让 tensor core 一直忙着。

想对着代码逐行读，前向见 [[learning/flash-attention/08-flashattention3-kernel|FA3 前向内核逐行读]]，反向见 [[learning/flash-attention/09-flashattention3-bwd-kernel|FA3 反向内核逐行读]]。

## 为什么 H100 上 exp 是瓶颈

先看硬件吞吐的不对称（H100 SXM5）：

| 操作 | 吞吐 |
| --- | --- |
| FP16 matmul | 989 TFLOPS/s |
| 特殊函数（`exp` 等） | 3.9 TFLOPS/s |

3.9 TFLOPS 的来源是：每个 SM 每周期能做 16 次特殊函数运算，乘 132 个 SM、再乘 1830 MHz 的 boost 频率（$ 16 \times 132 \times 1.83\,\mathrm{G} \approx 3.9 $ T）。989/3.9 ≈ 250 倍（论文按 256× 记，它是从"FP16 峰值 = 每 SM 每周期 4096 FLOPs"反推的）。FP16、head dim 128 的 attention，matmul FLOPs 比 exp 操作多约 512 倍，但 exp 吞吐低 256 倍，算下来 **exp 可能占掉一半的周期**。

512 这个数不是拍出来的：一个 $ B_r \times B_c $ 的 score 块里，每个元素要做 1 次 exp，却对应 GEMM0 的 $ 2d $ FLOPs 和 GEMM1 的 $ 2d $ FLOPs（两次乘加各算 2 FLOPs，K 维长度都是 $ d $），所以比值就是 $ 4d = 512 $（$ d = 128 $）。FP8 更糟：matmul 吞吐翻倍，exp 不变。所以"把 exp 藏进 tensor core 干活的时候"是 FA3 最大的收益点。

## 技术 1：warp specialization + circular SMEM buffer

Hopper 允许在一条 CTA 内做 warp specialization：把 CTA 的线程拆成几组，各组只干一类活。FA3 前向的角色划分是（一个 CTA）：

- **producer warpgroup**：只负责发 TMA 加载 $ Q_i, K_j, V_j $ 进共享内存，以及（FP8 时）在共享内存里把 $ V $ 转置。
- **consumer warpgroup**：只负责用 WGMMA 算 $ Q K^\top $、做 softmax、算 $ P V $。

consumer 侧用几个 MMA warpgroup 由 tile 形状定：`hopper/tile_size.h` 的 `tile_size_fwd_sm90` 对 FP16/FP8 都选 `kBlockM = 128`（2 个 MMA warpgroup）当 hdim ≥ 97，hdim ≤ 96 才用 192 行（3 个）；例外是 head dim 64 配 `headdim_v = 256/512` 的 decode 形状（降到 128 行且关掉 intra-warpgroup 流水线 / 降到 64 行）。官方 SM90 调参笔记里的表（2 WG 覆盖 hdim ≤ 128、3 WG 覆盖 129–192）与代码里这份启发式不一致，实际生效的是代码这份。再加上 1 个 producer warpgroup；反过来，如果没有 FP8 的转置活，producer 只需要 1 个 warp（TMA 是单个线程就能发起的指令，`NumProducerThreads = NumThreadsPerWarp`）。

两者用 **circular SMEM buffer（多级流水线）** 连接：SMEM 按 $ s $ 个 stage 划分，producer 一次填一个 stage，consumer 一次消费一个 stage。同步靠两类原语：数据流用 **mbarrier**（CUTLASS 的 pipeline 抽象，producer 用 `producer_acquire` 等 stage 空出来、TMA 完成时由 `producer_get_barrier` 把信号打到 barrier 上，consumer 侧 `consumer_wait`；consumer 消费完 `consumer_release` 让 producer 复用这一 stage）；同一时刻只在某些 warp 内协调的场景（pingpong 调度、$ P $ 缓冲区的读写保护）才用 **named barrier**，也就是 `bar.sync` 的具名版本。

关键点是 **TMA 加载是异步的**（发出指令立刻返回，不阻塞）。所以 producer 可以连续发多个 stage 的加载，边发边算；论文 Algorithm 1 里 producer 的循环就是"等第 $ j \% s $ 个 stage 空出 → 发 $ K_j, V_j $ 的 load → commit"，而且在 buffer 填满的前 $ s $ 轮里根本不用等。

FA3 甚至可以让 producer 用 `setmaxnreg` 动态增减寄存器数：producer 只加载、不需要那么多寄存器，就把寄存器让出来给 consumer 的 softmax / GEMM 用（论文 Algorithm 1 第 3、12 行）。

## 技术 2：pingpong 调度（两个 warpgroup 轮流）

单靠 warp specialization 只是让"搬数据"和"算"重叠。但 consumer 内部：$ QK^\top $（GEMM0）→ softmax → $ PV $（GEMM1），softmax 卡在中间，tensor core 会空转。

FA3 用 **pingpong** 解决：把 consumer 再拆成两个 warpgroup，各管 $ Q_i $ 的一半行（`AtomLayoutQK` 把 M 维切给两个 WG）。但**光拆开没用**——两个 warpgroup 若齐步走，就会在同一时刻一起进 softmax：MUFU（特殊函数单元）挤在一起，tensor core 同时没人喂。所以还要用 `warp_scheduler_barrier_sync/arrive`（源码里是 named barrier，语义就是 `bar.sync`）把两边的 GEMM 相位**错开一个 block**：

```
warpgroup 1:   GEMM0(j)   softmax(j)   GEMM1(j)   GEMM0(j+1) ...
warpgroup 2:   softmax(j) GEMM0(j+1)   GEMM1(j+1) softmax(j+2)
```

这样任一时刻总有一个 warpgroup 在 softmax（走 CUDA core + MUFU），另一个在跑 GEMM（走 tensor core）——**两类执行单元同时在干活**，谁都不空着，这才是"隐藏 softmax 延迟"的具体含义。WGMMA 是异步的（发出后不阻塞后续指令），所以错相位只是指令流上的先后约定，不需要真的等对方算完。

相位只能错一块，不能再多：softmax(j) 必须等 GEMM0(j) 算完 $ S_i^{(j)} $ 才能开始，这个依赖是硬的，所以重叠的窗口就是"一块的 GEMM 时间"。论文实测这套从 570 TFLOPS 提到 620–640 TFLOPS（FP16、hdim 128、seq 8192）。

![producer 用 TMA 持续加载；两个 consumer warpgroup 的 GEMM 与 softmax 交替，互不空转](/learning/assets/fa3-pingpong.svg)
> 自绘示意图

## 技术 3：warpgroup 内的 2 级流水线

pingpong 是"warpgroup 之间"的重叠；2 级流水线是"同一个 warpgroup 内"的重叠。问题在于循环体内的依赖是死的：softmax(j) 读 GEMM0(j) 的结果 $ S $，GEMM1(j) 又读 softmax(j) 的结果 $ P $，论文 Algorithm 1 里第 17、21 行的 wait 把三者串成一条链，一个 warpgroup 内部没有能重叠的东西。

FA3 的解法是**跨迭代**把 GEMM1 挪到前一块去（论文 Algorithm 2）：GEMM1 用的是**上一块**的 $ P $（记作 $ \tilde{P}_{\text{cur}} $）和上一块的 $ V $（$ V_{j-1} $），所以它跟本轮的 GEMM0/softmax 没有依赖。每一轮变成：

```text
发 GEMM0：S_next = Q K_j^T      （commit，不等）
发 GEMM1：O += P_cur · V_{j-1}  （commit，不等）
等 GEMM0 完成 → 对 S_next 做 softmax 得 P_next   ← 此时 GEMM1 还在 tensor core 上跑
等 GEMM1 完成 → 重缩 O
```

也就是说，**softmax 读 $ S_{\text{next}} $ 算 $ \tilde{P}_{\text{next}} $ 的那段时间，张量核在算 $ \tilde{P}_{\text{cur}} V_{j-1} $**——exp 的延迟被 GEMM1 盖住。反过来，如果不用这个错位，同一个 warpgroup 只能"算完 $ S $ 再算 softmax 再算 $ PV $"，GEMM 和 exp 严格串行。

实现代价是多存一份 $ S_{\text{next}} $：论文说是每个 threadblock $ B_r \times B_c \times 4 $ 字节的额外寄存器，于是 pipelining 深度和大 block size 抢同一份寄存器预算，实际得靠 profiling 定。

两个容易混的点：这里的"2 级"指的是寄存器里多留一份 $ S $，**和 SMEM 的 stage 数不是一回事**——论文脚注明确说流水级数只是被 $ s $ 个 SMEM stage 上界约束，不必相等。另外伪代码只是理想执行序，论文专门提醒 NVCC 会为了优化重排指令、可能破坏手工安排的重叠，并在附录 B.2 用 SASS 反汇编确认生成的重叠代码符合预期；也就是说"照抄 Algorithm 2 的语句顺序"不保证真的重叠，得看 SASS。

FA3 还测过一个 3 级版本（附录 B.3 给了完整算法），但需要更多寄存器，和大 block size（也耗寄存器）冲突，论文说权衡更难平衡。

## FP8：two layout 问题 + block 量化 + 打散离群值

FP8 比 FP16 麻烦在两点：**布局不兼容**和**精度差**。

### 布局：V 要转置

FP8 的 WGMMA 对第二个 GEMM 要求 $ V $ 在共享内存里**沿序列维连续（k-major）**，而模型里 $ Q, K, V $ 通常是**沿 head 维连续（与序列维垂直）**。TMA 拷贝不能改连续维度，所以要么：
1. 在全局内存做一次转置（融合见 rotary embedding 或单独 kernel），或
2. 加载进共享内存后**在 kernel 内转置**。

FA3 选第 2 种：用 `LDSM`（ldmatrix）/ `STSM`（stmatrix）指令，一个 warp 一次搬 128 字节，还能顺带转置。FP8 时把两个 8-bit 元素打包成 16-bit 用 LDSM/STSM，转置版本不能拆打包的 8-bit 元素，中间还要做字节置换（源码 `transpose_V` 里的 `__byte_perm(upper, lower, 0x6420)` / `0x7531`）。这一步由 producer warpgroup 干，并且从第二块 V 开始，可以安排在"上一块 V 参与的两次 WGMMA 正在跑"的阴影里，不占额外时间。

### 布局：累加器和操作数 A 的寄存器排布不同

FP8 的 WGMMA，其 FP32 累加器（`acc_s`）的寄存器归属（每线程拿哪些元素）和"作为下一轮操作数 A（$ P $）"所需的布局不一样。这跟 [[learning/flash-attention/04-forward-kernel|FP16 forward]] 里的 `convert_layout_acc_Aregs` 是同一类问题，但 FP8 更严重，要显式用 byte-permute 把累加器里 `d0 d1 d2 d3 d4 d5 d6 d7` 重排成 `d0 d1 d4 d5 d2 d3 d6 d7`，再配合 V 转置的行置换，让 WGMMA 算出正确的输出。

### 精度：block quantization + incoherent processing

FP8（E4M3）只有 3 位尾数、4 位指数，误差大。而且大模型普遍有离群值（outlier），把 per-tensor 的 scale 撑大，其余数值被压进很粗的格子。FA3 用两招：

- **块量化（block quantization）**：不再 per-tensor，而是每个 $ B_r \times d $（或 $ B_c \times d $）块一个 scale。因为 FA3 本来就在块上操作，每个块的 $ S $ 乘一个块 scale 几乎零成本。这个量化可以融合进 rotary embedding（memory-bound，不额外耗时）。
- **incoherent processing（打散离群值）**：量化前把 $ Q $ 和 $ K $ 各乘一个随机正交矩阵 $ \mathcal{M} $。因为 $ \mathcal{M} \mathcal{M}^\top = I $，所以 $ (Q\mathcal{M})(K\mathcal{M})^\top = Q K^\top $，**不改变 attention 输出**。但 $ Q\mathcal{M} $ 的每个元素都是 $ Q $ 中若干个元素的随机和，单个离群值被"摊平"到整行，量化误差变小。实践中 $ \mathcal{M} $ 取"随机 ±1 对角阵 × Hadamard 矩阵"（论文跟的是 QuIP 与 QuIP# 的做法），可以 $ O(d \log d) $ 而不是 $ O(d^2) $ 地乘出来，还能融合进 rotary embedding。

论文验证这两招把 FP8 attention 的数值误差压低 **2.6×**。对照的是 per-tensor scale 的 FP8 baseline（RMSE 2.4e-2），拆开看：只去掉块量化是 9.3e-3、只去掉 incoherent processing 是 2.4e-2（也就是误差基本全由它负责），两者都在时 9.1e-3——2.4e-2 / 9.1e-3 ≈ 2.6（论文 Table 3）。

## 结果

| 指标 | FA2（H100） | FA3（H100） |
| --- | --- | --- |
| 利用率 | 35% | FP16 约 75% |
| FP16 峰值 | — | 740 TFLOPs/s |
| FP8 峰值 | — | 约 1.2 PFLOPs/s |
| 相对加速 | — | 1.5–2.0× |

对比一下，H100 的 FP8 密集峰值约 1979 TFLOPS（NVIDIA 规格表里 FP8 写 3958 TFLOPS、FP16 写 1979 TFLOPS，那两组都是"带稀疏"的数，稠密要减半；减半后 FP8 1979、FP16 989，正好和上面表里的 989 对上），1.2 PFLOPs/s 已到约 6 成；FP16 的 740 TFLOPs/s 到 989 TFLOPS 峰值的约 75%。FA3 把 attention 从"HBM 拖后腿"拉到了"接近算力上限"。

论文还用固定配置 `{batch=4, seqlen=8448, nheads=16, hdim=128}` 单独做了消融（Table 2），把三招里的两招分别拆掉，看谁贡献大：

| 配置 | 时间 | TFLOPs/s |
| --- | ---: | ---: |
| 三招全开（最优） | 3.538 ms | 661 |
| 只留 warp specialization（去掉 GEMM-softmax 重叠） | 4.021 ms | 582 |
| 只留 GEMM-softmax 重叠（去掉 warp specialization） | 4.105 ms | 570 |

两行对照起来读：全开 661；**拆掉 GEMM-softmax 重叠**（保留 warp specialization）掉到 582（−79 TFLOPs，−12%）；**拆掉 warp specialization**（保留 GEMM-softmax 重叠）掉到 570（−91 TFLOPs，−14%）。两项的收益量级相当，而且互相依赖：pingpong 的相位错开要靠 warp specialization 分出来的 producer/consumer 和寄存器重分配，2 级流水线又要在 pingpong 错开的那两组 warpgroup 里各跑一份。注意论文没给"两项都关"的配置，所以 570 这一行并不是"无重叠基线"（它本身就带着 GEMM-softmax 重叠）；严格说这三行只支持论文那句话——这两项合起来把 570 抬到 661。

> **2026-09 增补：** FA3 这篇（2407.08608）仍是 Hopper 的基准，论文结论没变；但 Blackwell 上"把 exp 藏进 tensor core"这套前提失效了——B200 的 tensor core 吞吐翻倍后，瓶颈换成共享内存流量和指数单元，后续工作见 [[learning/flash-attention/10-flashattention4|FA4]]（arXiv:2603.05451）。

## 一句话

FA3 是"把 attention 里所有能重叠的都重叠"：producer/consumer 分开让 TMA 和 WGMMA 重叠，双 warpgroup pingpong 让 softmax 的 exp 和 GEMM 重叠，2 级流水线让同一个 warpgroup 内的 GEMM 和 softmax 重叠，FP8 用块量化 + Hadamard 打散把低精度误差压回去。它把 attention 从"内存受限"推进到"几乎纯算力受限"。

## Reference

- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling（arXiv:2603.05451，Blackwell 后继）：<https://arxiv.org/abs/2603.05451>
- 官方代码（hopper/ 目录）：<https://github.com/Dao-AILab/flash-attention>
- NVIDIA CUTLASS（TMA / WGMMA / warp specialization 基础）：<https://github.com/NVIDIA/cutlass>
- QuIP / QuIP#（随机正交矩阵打散离群值）：<https://arxiv.org/abs/2307.13304>
- PTX ISA（LDSM / STSM / TMA 指令）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
