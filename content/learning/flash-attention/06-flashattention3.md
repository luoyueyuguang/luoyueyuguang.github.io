FA2 在 A100 上已经把 attention 推到接近 GEMM 的效率，但换到 H100 就露馅：FA2 在 H100 上只达到 35% 利用率。原因是 FA2 是为 Ampere 写的，没用上 Hopper 的新硬件。FA3 的三个技术全都围绕 Hopper 的两个新能力：**TMA**（Tensor Memory Accelerator，异步批量拷贝）和 **WGMMA**（warpgroup 矩阵乘，异步 tensor-core GEMM）。

> **FA3 的核心问题是"重叠"。** 数据搬移（TMA）、矩阵乘（WGMMA）、softmax 的指数函数分别用不同的硬件单元，FA2 让它们串行。FA3 用 warp specialization + 双 warpgroup + 多级 SMEM 缓冲，把这三件事错开，让 tensor core 一直忙着。

想对着代码逐行读，前向见 [[learning/flash-attention/07-flashattention3-kernel|FA3 前向内核逐行读]]，反向见 [[learning/flash-attention/12-flashattention3-bwd-kernel|FA3 反向内核逐行读]]。

## 为什么 H100 上 exp 是瓶颈

先看硬件吞吐的不对称（H100 SXM5）：

| 操作 | 吞吐 |
| --- | --- |
| FP16 matmul | 989 TFLOPS/s |
| 特殊函数（`exp` 等） | 3.9 TFLOPS/s |

差距约 250 倍。FP16、head dim 128 的 attention，matmul FLOPs 比 exp 操作多约 512 倍，但 exp 吞吐低 256 倍，算下来 **exp 可能占掉一半的周期**。FP8 更糟：matmul 吞吐翻倍，exp 不变。所以"把 exp 藏进 tensor core 干活的时候"是 FA3 最大的收益点。

## 技术 1：warp specialization + circular SMEM buffer

Hopper 的 SM 支持把 4 个 warpgroup 拆成不同角色。FA3 的 CTA（一个 thread block）分成两组角色：

- **producer warpgroup**：只负责发 TMA 加载 `$ Q_i, K_j, V_j $` 进共享内存，以及（FP8 时）在共享内存里把 `$ V $` 转置。
- **consumer warpgroup**：只负责用 WGMMA 算 `$ Q K^\top $`、做 softmax、算 `$ P V $`。

两者用 **circular SMEM buffer（多级流水线）** 连接：SMEM 按 `$ s $` 个 stage 划分，producer 一次填一个 stage，consumer 一次消费一个 stage。`bar.sync` 是同步原语：consumer 消费完会"释放"某 stage 让 producer 复用。

关键点是 **TMA 加载是异步的**（发出指令立刻返回，不阻塞）。所以 producer 可以连续发多个 stage 的加载，边发边算。算法里写的是伪代码（`For each stage... wait for consumed... issue load... commit`），真实实现用 CUTLASS 的 `pipeline_k.producer_acquire/commit` 和 `consumer_wait/release` 管理 barrier。

FA3 甚至可以让 producer 用 `setmaxnreg` 动态增减寄存器数：producer 只加载、不需要那么多寄存器，就把寄存器让出来给 consumer 的 softmax / GEMM 用。

## 技术 2：pingpong 调度（两个 warpgroup 轮流）

单靠 warp specialization 只是让"搬数据"和"算"重叠。但 consumer 内部：`$ QK^\top $`（GEMM0）→ softmax → `$ PV $`（GEMM1），softmax 卡在中间，tensor core 会空转。

FA3 用 **pingpong** 解决：把 consumer 再拆成两个 warpgroup。用 `bar.sync` 强制两个 warpgroup 的 GEMM 交错：

```
warpgroup 1:   GEMM0(j)   softmax(j)   GEMM1(j)   GEMM0(j+1) ...
warpgroup 2:   softmax(j) GEMM0(j+1)   GEMM1(j+1) softmax(j+2)
```

warpgroup 1 做 softmax 的时候，warpgroup 2 在做 GEMM；下一轮反过来。**exp 的硬件单元在 tensor core 旁边两套并行走，谁都不空着。** 论文说这套从 570 TFLOPS 提到 620–640 TFLOPS（FP16、hdim 128、seq 8192）。

![producer 用 TMA 持续加载；两个 consumer warpgroup 的 GEMM 与 softmax 交替，互不空转](/learning/assets/fa3-pingpong.svg)

## 技术 3：warpgroup 内的 2 级流水线

pingpong 是"warpgroup 之间"的重叠；2 级流水线是"同一个 warpgroup 内"的重叠。它把这个序列：

```text
GEMM0(j) → softmax(j) → GEMM1(j) → [next iter]
```

改成把下一轮的 GEMM0 提前：

```text
GEMM0(j) → softmax(j)                 （发 GEMM1(j)，不等待）
GEMM1(j) ∥ GEMM0(j+1) → softmax(j+1) ...
```

具体做法是**多存一份 `$ S_{next} $` 在寄存器**：算完 `$ S_j $`（GEMM0）后立刻发起 `$ P_{j-1} V_{j-1} $`（GEMM1），同时用 `$ S_{j+1} $` 算 softmax。代价是 `$ S_{next} $` 要占 `$ B_r \times B_c \times 4 $` 字节的额外寄存器，寄存器压力变大。

FA3 还测过一个 3 级版本，但需要更多寄存器，和大 block size（也耗寄存器）冲突，论文说权衡更难平衡。

## FP8：two layout 问题 + block 量化 + 打散离群值

FP8 比 FP16 麻烦在两点：**布局不兼容**和**精度差**。

### 布局：V 要转置

FP8 的 WGMMA 对第二个 GEMM 要求 `$ V $` 在共享内存里**沿序列维连续（k-major）**，而模型里 `$ Q, K, V $` 通常是**沿 head 维连续（与序列维垂直）**。TMA 拷贝不能改连续维度，所以要么：
1. 在全局内存做一次转置（融合见 rotary embedding 或单独 kernel），或
2. 加载进共享内存后**在 kernel 内转置**。

FA3 选第 2 种：用 `LDSM`（ldmatrix）/ `STSM`（stmatrix）指令，一个 warp 一次搬 128 字节，还能顺带转置。FP8 时把两个 8-bit 元素打包成 16-bit 用 LDSM/STSM，转置版本不能拆打包的 8-bit 元素，中间还要做字节置换（`__byte_perm`）。

### 布局：累加器和操作数 A 的寄存器排布不同

FP8 的 WGMMA，其 FP32 累加器（`acc_s`）的寄存器归属（每线程拿哪些元素）和"作为下一轮操作数 A（`$ P $`）"所需的布局不一样。这跟 [[learning/flash-attention/03-forward-kernel|FP16 forward]] 里的 `convert_layout_acc_Aregs` 是同一类问题，但 FP8 更严重，要显式用 byte-permute 把累加器里 `d0 d1 d2 d3 d4 d5 d6 d7` 重排成 `d0 d1 d4 d5 d2 d3 d6 d7`，再配合 V 转置的行置换，让 WGMMA 算出正确的输出。

### 精度：block quantization + incoherent processing

FP8（E4M3）只有 3 位尾数、4 位指数，误差大。而且大模型普遍有离群值（outlier），把 per-tensor 的 scale 撑大，其余数值被压进很粗的格子。FA3 用两招：

- **块量化（block quantization）**：不再 per-tensor，而是每个 `$ B_r \times d $`（或 `$ B_c \times d $`）块一个 scale。因为 FA3 本来就在块上操作，每个块的 `$ S $` 乘一个块 scale 几乎零成本。这个量化可以融合进 rotary embedding（memory-bound，不额外耗时）。
- **incoherent processing（打散离群值）**：量化前把 `$ Q $` 和 `$ K $` 各乘一个随机正交矩阵 `$ \mathcal{M} $`。因为 `$ \mathcal{M} \mathcal{M}^\top = I $`，所以 `$ (Q\mathcal{M})(K\mathcal{M})^\top = Q K^\top $`，**不改变 attention 输出**。但 `$ Q\mathcal{M} $` 的每个元素是 `$ Q $` 各元素的一个随机线性组合，离群值被"摊平"，量化误差变小。实践中 `$ \mathcal{M} $` 取"随机 ±1 对角阵 × Hadamard 矩阵"，可以 `$ O(d \log d) $` 乘，还能融合进 rotary embedding。

论文验证这两招把 FP8 attention 的数值误差压低 **2.6×**。

## 结果

| 指标 | FA2（H100） | FA3（H100） |
| --- | --- | --- |
| 利用率 | 35% | FP16 约 75% |
| FP16 峰值 | — | 740 TFLOPs/s |
| FP8 峰值 | — | 约 1.2 PFLOPs/s |
| 相对加速 | — | 1.5–2.0× |

对比一下，H100 的 FP8 密集峰值约 1979 TFLOPS，1.2 PFLOPs/s 已到约 6 成，FP16 的 740 TFLOPs/s 到 989 TFLOPS 峰值的约 75%。FA3 把 attention 从"HBM 拖后腿"拉到了"接近算力上限"。

## 一句话

FA3 是"把 attention 里所有能重叠的都重叠"：producer/consumer 分开让 TMA 和 WGMMA 重叠，双 warpgroup pingpong 让 softmax 的 exp 和 GEMM 重叠，2 级流水线让同一个 warpgroup 内的 GEMM 和 softmax 重叠，FP8 用块量化 + Hadamard 打散把低精度误差压回去。它把 attention 从"内存受限"推进到"几乎纯算力受限"。

## Reference

- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（arXiv:2407.08608）：<https://arxiv.org/abs/2407.08608>
- 官方代码（hopper/ 目录）：<https://github.com/Dao-AILab/flash-attention>
- NVIDIA CUTLASS（TMA / WGMMA / warp specialization 基础）：<https://github.com/NVIDIA/cutlass>
- QuIP / QuIP#（随机正交矩阵打散离群值）：<https://arxiv.org/abs/2307.13304>
- PTX ISA（LDSM / STSM / TMA 指令）：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
