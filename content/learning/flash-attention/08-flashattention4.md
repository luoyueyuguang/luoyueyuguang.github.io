FA3 是为 Hopper 写的，落到 Blackwell（B200/GB200）就失效：FA3 的根本假设是"tensor core 是瓶颈"，但在 B200 上 tensor core 吞吐翻倍了，**瓶颈换人了**。FA4 不再把 GPU 当成一块均匀的算力，而是先做 roofline 找出真正的瓶颈，再"算法 + kernel"一起改。

> **FA4 的核心是"非对称硬件缩放"。** B200 的 tensor core 是 H100 的两倍（FP16/BF16 约 2.25 PFLOPS vs 1 PFLOPS），但共享内存带宽、指数单元、整数/浮点 ALU 涨得慢或没涨。结果 attention 的主导成本不再是 MMA，而是**共享内存流量和指数函数**。FA3 那套"把 exp 藏进 tensor core 干活"在这里不够用了，因为 exp 单元根本没涨。

## roofline 先定性

FA4 对 forward 用三种资源算了个 roofline（tile $ M = N = d = 128 $ / $ 256 \times 128^2 $）：

| 资源 | $ 128^3 $ | $ 256 \times 128^2 $ |
| --- | ---: | ---: |
| MMA 计算 | **1024** | **2048** |
| 共享内存 | 768 | 1536 |
| 指数单元 | **1024** | **2048** |

**MMA 和指数单元并列瓶颈。** 这不是巧合：tensor core 8192 FLOP/cycle，exp 只有 16 op/cycle，差 512 倍；而一个 attention 块里 MMA FLOPs 和 exp 次数也差不多那个量级。所以"多出来的"就卡在 exp 上。

backward 更夸张：$ M=N=d=128 $ 时共享内存 3328 cycle，比 MMA 2560 还多 30%。**反向的瓶颈是共享内存带宽。**

结论：FA4 的优化方向不是多塞 MMA，而是 (1) 把 MMA 和 softmax 更好地重叠，(2) 提高 exp 吞吐，(3) 减少共享内存流量。

## 技术 1：为 Blackwell 重排的流水线

FA3 的 pingpong 是"两个 warpgroup 轮流做 softmax 和 GEMM"。FA4 继承了它，但因为有 **TMEM**（tensor memory，每 SM 256 KB，专门存张量核中间 tile），分配方式变了：

- Blackwell 的 tensor core 把累加器放在 **TMEM**（不是寄存器）。一个累加器 tile 是 **128×128**（Hopper 是 64×128）。
- 于是用**两个 128 线程的 softmax warpgroup**，每个线程处理整行（128 个元素）。这样**不需要跨 warp shuffle 来归约行 max**，每个线程只要一个统计寄存器。
- $ P $ 通过 TMEM（不是寄存器文件）传给下一个 MMA，可以**把输出重缩放单独拆给一个"correction" warpgroup**，让它退出关键路径。

实际的 warp 分工（`flash_fwd_sm100.py` 里的 `FlashAttentionForwardSm100`）：

| warp 组 | warp 号 | 角色 |
| --- | --- | --- |
| softmax | 0–7 | 两个 warpgroup，各 128 线程，做整行 softmax |
| correction | 8–11 | 只在必要时做 $ O $ 重缩放 |
| mma / TMA | 12–15 | 驱动 tensor core 和 TMA 加载 |

TMEM 分配（`tmem_s_offset`、`tmem_o_offset`）：两个 $ S $ tile 在列 0、128，两个 $ O $ tile 在列 256、384。因为 Blackwell tile 大，一个 CTA 就占两个输出 tile（高/低两个 Q tile），一半 TMEM 留给 $ S, P $。**选"两个 $ S $ tile 和 $ P $ 重叠"这一种分配**，这样流水线一启动就能立刻算两个 $ S $，还能留 TMEM 给 correction warpgroup 传 rescale 统计量。

代价是寄存器：一个线程要hold 整行 128 个元素（BF16 下输入 128 寄存器、输出可能 64），很容易 spill。FA4 的招是**把 $ P $ 分成四份存**：前四分之三存一次、立刻触发对应 MMA，最后四分之一单独存，降低峰值寄存器压力。

## 技术 2：用多项式模拟指数

exp 单元 H100/B200 都是 16 op/cycle/SM，tensor core 是 8192，差 512 倍。FA4 把一部分 $ 2^x $ 挪到 **FMA 单元**上算（FMA 可以和 MUFU 并行），绕过 exp 单元的瓶颈。

做法是经典的范围缩减（Cody-Waite）+ 多项式逼近：

$$
2^x = 2^{\lfloor x \rfloor}\, 2^{x - \lfloor x \rfloor}
$$

- **整数部分** $ 2^{\lfloor x \rfloor} $：IEEE 754 的指数字段本身就是 2 的幂，移位 + 加指数位即可，用整数 ALU。
- **小数部分** $ 2^{x - \lfloor x \rfloor} \in [0,1) $：用一个多项式 $ \sum_i p_i x_{\text{frac}}^i $ 逼近，Horner 法 + FMA。系数用 Sollya 算出来，$ p_0 = 1 $。
- 先 clamp $ x \ge -127 $ 防下溢；$ \lfloor x \rfloor $ 用"加 $ 2^{23}+2^{22} $ 再减回来（round-down）"的位技巧。

**精度：** 论文测了不同多项式阶数（4M 个随机输入，对照 FP64）：

| 方法 | FP32 最大相对误差 | BF16 最大相对误差 |
| --- | ---: | ---: |
| 硬件 `MUFU.EX2` | $ 1.41\times10^{-7} $ | $ 3.89\times10^{-3} $ |
| 3 阶多项式 | $ 8.77\times10^{-5} $ | $ 3.90\times10^{-3} $ |

3 阶多项式在 FP32 下误差是硬件的约 600 倍，**但 round 到 BF16 后和硬件几乎无差别**（BF16 本身的量化误差 $ 3.9\times10^{-3} $ 主导）。所以 FA4 默认用 3 阶，代价只要每评估一次多几条 FMA。

**关键不是全换，是"部分模拟"。** 多项式模拟要多寄存器、多寄存器带宽、更长延迟。全换会 spill，得不偿失。FA4 只模拟每行 10–25% 的元素，其余用硬件 `MUFU.EX2`，比例按 tile 配置经验调。代码里用 `ex2_emu_freq` 控制：越大模拟越多；`sm_103`（B300）有原生快 exp2，直接 `freq=0`。

## 技术 3：跳过不必要的 softmax 重缩放

FA2 起，FA 用"未归一化的 $ \widetilde{O} $ + 结尾除 $ \ell $"省掉每步除法。但 online softmax 里每合并一个块还是有理器运算：$ e^{m_{j-1} - m_j} \widetilde{O}_{j-1} $。FA4 观察到一个事实：

**只有当 $ m_j > m_{j-1} $ 出现过大的新值时才需要重缩放。** 而且可以容忍一些"松弛"：只有当 $ m_j - m_{j-1} > \tau $（$ \tau $ 典型取 $ \log_2 256 = 8.0 $，对应缩放因子 256）才重缩，否则跳过并继续用 $ m_{j-1} $：

$$
O_j = \begin{cases}
e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j & m_j - m_{j-1} > \tau \\
O_{j-1} + e^{S_j - m_{j-1}} V_j & \text{否则}
\end{cases}
$$

**为什么正确：** 结尾反正要用最终的 $ m_{\text{final}} $ 和 $ \ell_{\text{final}} $ 归一化，中间跳过的小偏差会被最后一步修正。**工程实现**：为避免 warp 发散，只要 warp 里任一线程需要重缩，整个 warp 就重缩。

## 技术 4：反向用 TMEM + 2-CTA MMA 压共享内存

反向的 roofline 显示共享内存是瓶颈（3328 vs MMA 2560 cycle）。两个改动：

**① TMEM 存更多中间量。** TMEM 只能放 4 个 128×128 累加器 tile。FA4 的分配：$ S $ 和 $ P $ 共用一个 TMEM 块（offset 0），$ dP, dS, dQ $ 共用另一个。这比 FA3（全部累加器在寄存器）允许更多的 MMA 与非 MMA 重叠：FA4 用**上一轮的 $ dQ, dK $ MMA 和当前轮的 softmax** 重叠，因为 Blackwell 要至少两个 MMA 并发。

**② 2-CTA MMA 模式。** Blackwell 的 2-CTA MMA 把输出累加器按 M 维切分。用 $ M=256, N=K=128 $ 的 tile，两个 CTA 当一个更大的 tile：每个 CTA 只加载/暂存**一半**的 operand B，只保留自己那部分累加器。这**粗略减半 operand B 的共享内存流量**。

问题在 $ dQ $ 的归约轴。$ dQ $ 的归约方向是 $ N $，天然被 CTA pair 切分；但每个 CTA 仍要对自己那几行做完整归约。FA4 用 **DSMEM（distributed shared memory）** 交换一半 $ dS $ tile，让每个 CTA 组成一个 $ (M/2 \times 2N) $ 的 operand，再去跑 CTA-pair UMMA（加倍的归约）。**附带好处：每个 CTA 只写一半 $ dQ $，全局 atomic 归约次数减半。**

![反向 dQ 的 2-CTA MMA：两个 CTA 用 DSMEM 交换 dS，凑成加倍归约](/learning/assets/fa4-2cta.svg)

## 确定性反向

反向的全局归约（$ dQ $，GQA 时还有 $ dK/dV $）是**非确定**的（谁先到谁加）。FA4 提供一个确定性模式：用**信号量锁**把全局归约串行化，每个 CTA 按预定顺序拿锁、归约、放锁（signal 计数器 +1）。代价是内存栅栏（保证信号量写全局可见）+ 等锁的停顿。

为了少停顿，做 **CTA swizzle**（head/batch 维度打乱），并针对因果 mask 用 SPT（shortest-processing-time-first）顺序：KV 块按降序、query 块从对角往上、$ dQ $ 归约按 query 块索引降序，保证没有 CTA 在第一次 $ dQ $ 写时被卡住。

## 全部用 CuTe-DSL（Python）写

FA4 的一个亮点是**没有任何 CUDA C++**：整个 kernel 用 **CuTe-DSL**（CuTe 的 Python 嵌入）写，编译器下沉到 PTX，再过 `ptxas` 出 SASS。编程模型和 CUTLASS C++ 同构，保留完全表达能力，还有自定义 PTX 逃生口。想对照着看它怎么落到 CuTe-DSL 代码，前向读 [[learning/flash-attention/09-flashattention4-kernel|FA4 前向内核逐行读]]，反向读 [[learning/flash-attention/10-flashattention4-bwd-kernel|FA4 反向内核逐行读]]。

**编译时间**是关键收益（传统 C++ 模板在 FA2/FA3 要预编译几百个变体）：

| | forward | backward |
| --- | ---: | ---: |
| FA3（C++） | 55 s | 45 s |
| FA4（CuTe-DSL） | 2.5 s | 1.4 s |
| 加速 | 22× | 32× |

代码在仓库 `flash_attn/cute/`：`flash_fwd_sm100.py`（Blackwell forward）、`flash_bwd_sm100.py`（backward）、`flash_fwd_sm120.py`（sm120）等。这套 CuTe-DSL 实现现在已经打包成可 `pip install flash-attn-4` 的独立发行版（`flash-attn-4==4.0.0b29`），接口照常从 `flash_attn.cute` 导入 `flash_attn_func` / `flash_attn_varlen_func`；CUDA 13 时用 `pip install "flash-attn-4[cu13]"` 拿到最优性能。注意它是"Hopper + Blackwell"都覆盖（sm80/sm90/sm100/sm120 都在 `interface.py` 里分派），论文针对 Blackwell 的提速是其中 sm100/sm120 路径，不是包只支持 Blackwell。
```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
out = flash_attn_func(q, k, v, causal=True)
```

## 结果

B200（FP16/BF16）：

| 指标 | FA4 |
| --- | --- |
| 峰值 | 1613 TFLOPs/s（约 71% 理论峰值） |
| vs cuDNN 9.13 | 1.1–1.3×（4k+ 序列稳定领先） |
| vs Triton | 2.1–2.7× |
| 备注 | FA3 在 B200 上跑不了 |

FA4 还测了 DeepSeek V3 用的 `head dim (192, 128)` 配置，也是长序列占优。

顺带一句：这篇论文后来入选了 **MLSys 2026 的 oral**（oral 页面：<https://mlsys.org/virtual/2026/oral/3759>），是 FlashAttention 系列在顶会上的正式亮相。上面这些数字（1613 TFLOPs/s、1.3× vs cuDNN、2.7× vs Triton、20–30× 编译提速）在 oral 页面和 arXiv 正文里一致。

## Reference

- FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling（arXiv:2603.05451，MLSys 2026 oral）：<https://arxiv.org/abs/2603.05451>
- MLSys 2026 oral 页面：<https://mlsys.org/virtual/2026/oral/3759>
- `flash-attn-4` PyPI 发行版（CuTe-DSL，Hopper + Blackwell）：<https://pypi.org/project/flash-attn-4/>
- 官方代码（flash_attn/cute，CuTe-DSL 实现）：<https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute>
- CuTe-DSL：<https://github.com/NVIDIA/cutlass/tree/main/python/cutlass/cute_dsl>
- Blackwell tensor memory（TMEM）架构与 2-CTA MMA：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
- Cody-Waite 范围缩减 / 多项式逼近（Handbook of Floating-Point Arithmetic）：<https://hal.science/hal-00292005>
