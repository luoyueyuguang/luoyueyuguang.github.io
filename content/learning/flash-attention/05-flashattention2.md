FA1 解决的问题是"HBM 读写太多"，FA2 解决的问题是"算力利用率太低"。同一块 A100，用 GEMM 能做到 80–90% 的理论峰值，FA1 的 forward 只能到 30–50%，backward 只有 25–35%。FA2 没有改算法，只改了**怎么在 GPU 上切分工作**。

> **FA2 的三个改进都可以归为"把时间花在 matmul 上，别花在非 matmul 和共享内存搬移上"。先用一个数字感受成本不对称：A100 的 FP16 matmul 峰值是 312 TFLOPs/s，而非 matmul 的 FP32 只有 19.5 TFLOPs/s。也就是说一个非 matmul FLOP 的成本约等于 16 个 matmul FLOP。**

[[learning/flash-attention/01-flash-attention|系列总览]] 里说 FA2 是"纯速度优化"，具体就这三招：

1. 两个算法微调，把非 matmul 的 FLOPs 砍掉。
2. forward/backward 都沿序列维并行，提高占用率。
3. 一个 thread block 内的 warp 分工改成"把 Q 切给 warp"，去掉共享内存通信。

## 算法微调 1：不用每步都除 ℓ

FA1 的 forward 每合并一个块，都对 $ O $ 做一次除法（$ \mathrm{diag}(\ell)^{-1} $）再乘回旧 $ \ell $（见 [[learning/flash-attention/02-online-softmax|算法]] 第 13 行）。FA2 发现这一步可以省：**直接维护"未归一化"的 $ \widetilde{O} $，只在循环结束时除一次 $ \ell^{(\mathrm{last})} $。**

FA1 的更新（每一步都归一化）：

$$
O^{(2)} = \mathrm{diag}\big(\ell^{(1)}/\ell^{(2)}\big)^{-1} O^{(1)} + \mathrm{diag}\big(\ell^{(2)}\big)^{-1} e^{S^{(2)} - m^{(2)}} V^{(2)}
$$

FA2 维护"未归一化"的 $ \widetilde{O} $，每步只按 max 变化重缩（$ e^{m^{old}-m^{new}} $，注意不是它的逆；新块 max 变大时旧项要**缩小**到同一指数基准）：

$$
\widetilde{O}^{(j)} = e^{m^{(j-1)} - m^{(j)}} \widetilde{O}^{(j-1)} + e^{S^{(j)} - m^{(j)}} V^{(j)}
$$

结尾才做一次归一化：

$$
O = \mathrm{diag}\big(\ell^{(\mathrm{last})}\big)^{-1} \widetilde{O}^{(\mathrm{last})}
$$

> **论文那行 $ \widetilde{O}^{(2)} = \mathrm{diag}(\ell^{(1)})^{-1} O^{(1)} + \dots $ 容易读错。** 如果 $ O^{(1)} $ 是 FA1 那种已归一化输出，还原到未归一化要**乘** $ \ell^{(1)} $（写成 $ \mathrm{diag}(\ell^{(1)}) $），而且合并因子是 $ e^{m^{(1)} - m^{(2)}} $ 而非其逆。拿它在文中的 2-block 例子里 $ e^{s^{(1)}-m}V^{(1)} + e^{s^{(2)}-m}V^{(2)} $ 反推就能对上。论文这里的 $ \diag(\cdot)^{-1} $ 是符号笔误；正确递推见 [[learning/flash-attention/02-online-softmax|在线 softmax 与分块]] 的合并式。

代码上对应 [[learning/flash-attention/03-forward-kernel|forward kernel]] 里的 `softmax_rescale_o`：它只乘 `scores_scale`（$ e^{m_{old} - m_{new}} $），不除 $ \ell $。省掉的是每个 block 一次的对角矩阵乘（$ d $ 个元素乘 $ \ell $）+ 一次除法。循环里 $ T_c $ 次，累计下来可观。

## 算法微调 2：只存 logsumexp

FA1 反向需要 $ m $ 和 $ \ell $ 两个统计量。FA2 发现只需要一个：

$$
L = m + \log \ell
$$

因为 $ P = e^{S - L} $（推导见 [[learning/flash-attention/04-backward-kernel|反向内核]]）。于是反向只需读 $ L $，不用读两个向量。这既是省显存（$ O(N) $ → 还是 $ O(N) $，但少一个），也是省一次 HBM 读取。

## 序列维并行

FA1 的并行方式："1 个 thread block 处理 1 个 (batch, head) 的整个 sequence"，总共 $ \text{batch} \times \text{head} $ 个 block。A100 有 108 个 SM，这个数大于 80 时能填满。但**长序列通常 batch 小**（比如用 1M context 训一个模型，batch 可能只有一两条），这个时候 $ \text{batch} \times \text{head} $ 远小于 SM 数，大量 SM 空置。

举个例：一个 batch 64、16 heads 的 GPT 式配置，$ 64 \times 16 = 1024 $ 个 block，108 个 SM 能吃饱；但换成 batch=1、16 heads 的长上下文，就只剩 $ 1 \times 16 = 16 $ 个 block，绝大多数 SM 全程闲置。

FA2 的思路：**再沿序列维切一刀。**

- **forward**：outer loop 已经是"遍历行块 $ i $"，行块之间完全独立（每个行块只用自己的 $ Q_i $ 和全部 $ K, V $）。FA2 把不同行块调度到不同 thread block，不通信。这是"embarrassingly parallel"。
- **backward**：反向的共享计算是 $ dQ $（要跨列块累加）。FA2 让每个 thread block 管一段**列块**（$ K_j, V_j $），$ dK, dV $ 各自独立累加，$ dQ $ 用 **atomic add** 合并（见 [[learning/flash-attention/04-backward-kernel|反向内核]] 的 `atomicAdd` 那段）。

这个把 sequence 维也并行化的想法，最早来自 Phil Tillet 的 Triton 实现，FA2 把它搬进了 CUDA。

## warp 分工：split-K 换成 split-Q

一个 thread block 通常 4 或 8 个 warp。**算 $ S = QK^\top $ 和 $ O = PV $ 时，$ Q, K, V $ 怎么分给 warp，决定了要不要通信。**

**FA1 的做法（split-K）**：把 $ K, V $ 沿 $ k $ 维切给 4 个 warp，$ Q $ 让所有 warp 都能读。每个 warp 算自己那一列块 $ Q K_{warp}^\top $。问题在第二个 GEMM：每个 warp 算出的 $ P_{warp} V_{warp} $ 是** $ O $ 的偏和**，必须写回共享内存、`__syncthreads`、再对 4 个偏和求和。这一步 shared memory 读写 + 同步拖慢成 proc。

**FA2 的做法（split-Q）**：把 $ Q $ 沿 $ m $ 维切给 4 个 warp，$ K, V $ 全部可读。每个 warp 先算自己那段 $ Q K^\top $ 的 $ S $，再乘 $ V $ 得自己那段 $ O $。**$ Q $ 的行块之间完全独立，warp 之间零通信。**

FA2 正向 warp 分工：

| | FA1（split-K） | FA2（split-Q） |
| --- | --- | --- |
| warp 处理 | K/V 的列块 | Q 的行块 |
| Q 的可见性 | 全 warp 共享 | 每个 warp 一份 |
| P·V 结果 | 需要 warp 间求偏和 | 每 warp 独立成块 |
| 共享内存读写 | 多次（写偏和 + 同步 + 求和） | 无 |
| 通信 | `__syncthreads` | 不需要 |

![FA1 split-K 与 FA2 split-Q 的 warp 分工对比](/learning/assets/fa2-warp-partition.svg)

代码层面，FA2 让 $ Q $ 留在寄存器（`Is_Q_in_regs`），每个 warp 有自己那行块 $ Q $ 的片段 `tSrQ`。这就是 [[learning/flash-attention/03-forward-kernel|forward kernel]] 里那个 `if (Is_Q_in_regs) ... tSrQ_copy_view` 的来历。

**代价**：$ Q $ 切给 warp 意味着每个 warp 要能独立算完整 attention（包括 softmax 的行 max / 行和），所以 $ B_r $ 不能太大，否则寄存器不够。FA2 实测块大小常取 $ \{64, 128\} \times \{64, 128\} $，按 head dim 和设备共享内存调。

**backward 同理**：FA1 反向也用 split-K；FA2 反向避开 split-K，因为 $ dQ, dK, dV $ 之间的依赖比 forward 复杂，但代价是反向仍要一些同步（`__syncthreads` 也存在于 `flash_bwd_kernel.h` 里）。即便如此，避免 split-K 还是省了大量共享内存读写。

## MQA / GQA

FA2 原生支持 MQA（multi-query attention）和 GQA（grouped-query attention）：多个 query head 共享一组 KV head。实现上不复制 $ K, V $，而是用 `h_h_k_ratio = H_q / H_{kv}` 去索引 head。`params.h_h_k_ratio` 就是 GQA 的 group 大小。反向时把 $ dK, dV $ 在共享了同一组 KV 的多个 query head 之间求和即可。

## 结果

几个 key 数字：

| 指标 | FA1 | FA2 |
| --- | --- | --- |
| forward 算力利用率（A100） | 30–50% | 最高 73% |
| backward 算力利用率（A100） | 25–35% | 最高 63% |
| forward-only 峰值 | — | 230 TFLOPs/s |
| GPT 式训练吞吐（每 A100） | — | 225 TFLOPs/s（72% 模型 FLOPs） |

对比 FA1 约快 `2×`；长序列（seq 2k/8k）训练端到端比 FA1 快 `1.3×`，比不带 FA 的 baseline 快 `2.8×`。它已经把 attention 推到接近 GEMM 的效率水平。

## Reference

- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（arXiv:2307.08691）：<https://arxiv.org/abs/2307.08691>
- 官方代码：<https://github.com/Dao-AILab/flash-attention>
- Triton 版 fused attention 教程（Phil Tillet 最先给出 split-Q / 序列并行）：<https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py>
- 前作 FlashAttention（arXiv:2205.14135）：<https://arxiv.org/abs/2205.14135>
