聊 4-bit 浮点，绕不开 **NVFP4**——NVIDIA Blackwell 上用于极大模型推理的"最小浮点"。它只有 4 个 bit（1 sign + 2 exponent + 1 mantissa），即**E2M1**，正数能表示的档位只有 8 个。可它恰恰是 Blackwell tensor core 吞吐的顶点：位数砍到 4，换来的是**精度粗到只剩"几档"**，必须靠 block scaling 把这几档"换算"到实际数值附近，才勉强喂得动大模型。想先看清楚整个精度家族，见 [[learning/precision/01-overview|精度总览]]。

> tips：FP4 整个浮点值域很小——0、0.5、1、1.5、2、3、4、6。NVFP4 在 Blackwell 上给每一小块数据配一个高精度缩放因子，把这些值"就地放大缩小"，从而 4-bit 的粗颗粒也能保得住大模型的精度。

### 位布局

| 字段 | 位数 |
| --- | --- |
| sign | 1 |
| exponent | 2（偏置 1） |
| fraction | 1 |
| 总计 | 4 |

正规数（normal）的表示与 fp8/fp16 同构，只是字段全部缩到最小：

$$
x = (-1)^s \cdot 2^{E-1} \cdot (1.f)_2, \qquad E \in \{1,2,3\}
$$

指数**偏置为 1**。$E=0$ 时落入次正规（subnormal），此时指数因子是 $2^{1-1}=2^0=1$：

$$
x = (-1)^s \cdot (0.f)_2
$$

由于尾数只有 1 位，正规数的每个 binade 里只有两个档（$1.0_2$ 和 $1.1_2$），次正规也只有 0 和 0.5。把全部编码列出来：

| exponent (e) | mantissa (m) | 值 | 说明 |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 零点（次正规） |
| 0 | 1 | 0.5 | 最小正数（次正规） |
| 1 | 0 | 1.0 | 最小正规数 |
| 1 | 1 | 1.5 | |
| 2 | 0 | 2.0 | |
| 2 | 1 | 3.0 | |
| 3 | 0 | 4.0 | |
| 3 | 1 | 6.0 | 最大 |

加上符号位，负数就是同样的这一列。所以 FP4 的**正数只有 8 档**（0 和 7 个非零档），任意实数都会被归一到离它最近的那一档——这正是它"粗"的根源。它也没有 inf / NaN 编码，16 个 bit pattern 全是有限数。

### 关键数值

| 项目 | 值 | 补充 |
| --- | --- | --- |
| 有效精度 | 2 bit | 1 位显式尾数 + 隐式前导 1 |
| 单位舍入 $u$ | $2^{-2} = 0.25$ | 相对误差下限 |
| 十进制有效位 | $\approx 0.6$ | $2\log_{10}2$ |
| 最大 | 6 | $2^2 \cdot (1.1)_2 = 6$ |
| 最小正规数 | 1.0 | $2^0 \cdot 1.0$ |
| 最小正数 | 0.5 | $E=0, f=1$（次正规） |

一个常见误区：**别把"位宽 4"当成有效精度**。有效精度是"尾数位数 + 1"，FP4 只有 2 bit；若用总位宽 4 直接算 $4\log_{10}2 \approx 1.2$，会高估一倍。真正能可靠区分"两个不同值"的相对能力只有约 0.6 位十进制——连一个完整的小数位都不到。

### 和相邻格式比


| 格式 | 位宽 | 指数位 | 尾数（有效） | 有效精度 | 单位舍入 | 值域 | 缩放策略 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP4 (E2M1) | 4 | 2 | 1(+1) | 2 bit | $2^{-2}=0.25$ | $\approx[-6,6]$ | 仅软件缩放因子 |
| MXFP4 | 4 | 2 | 1(+1) | 2 bit | $2^{-2}=0.25$ | $\approx[-6,6]\times 2^k$ | 每 32 元素共 1 个 E8M0（$2^n$）scale |
| **NVFP4** | 4 | 2 | 1(+1) | 2 bit | $2^{-2}=0.25$ | $\approx[-6,6]\times$ 每块 scale | 每 16 元素共 1 个 E4M3 scale + 每 tensor 1 个 FP32 二级 scale |
| FP8 (E4M3) | 8 | 4 | 3(+1) | 4 bit | $2^{-4}\approx0.0625$ | $\approx\pm448$ | 无 |

三者共同点是**基础元素都是 E2M1，有效精度和单位舍入完全相同**——差异不在"单个数能多准"，而在"这一档能放到多大/多细的位置上"。MXFP4 用 32 个元素共享一个 $2^n$ 粗缩放；[[learning/precision/03-mxfp4|NVFP4 的兄弟 MXFP4]] 的缩放是"跳档"的 2 的幂。NVFP4 把块缩到 16 个元素，且缩放开到 FP8 的 E4M3（支持非 2 的幂、更细的分数刻度），再配一个 per-tensor 的 FP32 二级缩放——所以同样 4-bit，NVFP4 比 MXFP4 更贴数据的动态范围。上一档的 [[learning/precision/05-fp8|FP8 E4M3]] 位宽翻倍、有效精度翻倍到 4 bit、$u$ 降到 0.0625，精度和值域都好得多，但显存和带宽成本也翻倍。

### 需要FP4的理由

需要（推理显存/带宽/吞吐是瓶颈时）：

1. **超大 LLM 的推理**。权重堆在 HBM 上，走内存带宽的钱比算得快还贵。NVFP4 把权重压到约 4.5 bit/值（4-bit 值 + 每 16 个值共享 1 个 FP8 scale + 每 tensor 1 个 FP32），比 [[learning/precision/09-fp16|FP16]] 省约 3.5× 显存、比 [[learning/precision/05-fp8|FP8]] 省约 1.8×，直接换来更高的 token 吞吐和更低的每 token 能耗（Blackwell Ultra 相对 H100 号称每 token 能耗降约 50×）。
2. **对单值精度不敏感的大模型**。用 PTQ / QAT 配合 per-block 缩放，像 DeepSeek-R1 这类模型从 FP8 降到 NVFP4，关键评测的精度损失能压到 1% 以内——这正是 NVFP4 相对裸 FP4 的价值：靠更密的缩放把粗档位的误差分摊开。
3. **想要 Blackwell 的峰值算力**。FP4 是 Blackwell 上最小的数据类型，也就对应最高的 sparse/dense petaflops；在"算力优先、精度可容忍"的推理场景，它是压榨硬件选择。

需要小心的点

**数据有离群值或动态范围过大的层**。FP4只有 8 档，一个 outlier 会把整块 scale 拉飞，其他小值全被压成 0 或噪声（这正是要 Clustering（聚类）/QQQ 的原因）。此时应改 per-channel 或更高精度。

**精度 vs 范围 vs 成本**：FP4 三个维度全压到极限。有效精度最低（2 bit）、原生值域最窄（0.5→6 只有约 4 个倍频，比 fp16 的约 30 倍频窄得多），换来的是 4-bit 的最小存储和最省的数据搬移——这是一个用正确性换吞吐的刻意选择。

### 硬件与软件现状

**硬件**：

- **NVIDIA Blackwell（B200 / GB200，以及 Blackwell Ultra 的 B300 / GB300）的第五代 Tensor Core** 原生支持 FP4 与 NVFP4——注意是**第五代**，NVIDIA 把 Ampere / Hopper / Blackwell 依次算作第三代 / 第四代 / 第五代。官方博客明确写"fifth-generation NVIDIA Blackwell Tensor Cores"。tensor core 会自动处理 micro-block 分组、动态缩放和 4-bit 矩阵乘，无需软件逐块缩放。
- **Ampere 和 Hopper 不支持 FP4**——Hopper 最低只能到 FP8 / E4M3。所以"用 NVFP4"的前提是有 Blackwell。
- **区分"裸 FP4"与"NVFP4"**：裸 E2M1（FP4）只有软件缩放因子、无硬件加速缩放；NVFP4 才有硬件加速的 per-block（E4M3）缩放。前者计算简单但精度风险高，后者精度更好但需要对应硬件。

**软件 / 生态**：

- 量化工具：NVIDIA **TensorRT Model Optimizer**（支持 PTQ、QAT）、**LLM Compressor**（vLLM 生态）。可以把模型量化到 NVFP4 并导出统一的 Hugging Face checkpoint。
- 推理部署：**TensorRT-LLM**、**vLLM**（早期支持）、**SGLang**（支持中）。
- 预量化权重：Hugging Face 已有多款 NVFP4 checkpoint，如 `DeepSeek-R1-0528-FP4`、`Llama-3.1-405B-Instruct-FP4`、`FLUX.1-dev`。
- **存储 vs 计算**：4-bit E2M1 是"元素宽度"（每值 4 bit 参与矩阵乘），但 NVFP4 在盘上和 HBM 里实际是约 4.5 bit/值（4-bit 值 + 每 16 个值 1 个 FP8 scale），另加每 tensor 1 个 FP32 二级 scale。别把"4-bit"当成"总存储/值"。

### 代码示例

E2M1 只有 16 个编码，直接枚举最直观。下面用 Python 解码全部档位，并写一个"就近取整到 FP4"的量化辅助函数：

```python
def fp4_decode(s, e, m):
    """把一个 FP4 (E2M1) 4-bit 模式 (s,e,m) 解码为 float。"""
    if e == 0:
        v = m * 0.5                            # 次正规：(0.f)_2
    else:
        v = 2 ** (e - 1) * (1.0 + 0.5 * m)     # 正规：(1.f)_2
    return (-1) ** s * v

def fp4_encode(x):
    """把任意实数 x 就近归到最近的 FP4 档位，返回 (s, e, m, 值)。"""
    best = None
    for s in (0, 1):
        for e in range(4):
            for m in range(2):
                v = fp4_decode(s, e, m)
                err = abs(v - x)
                if best is None or err < best[0]:
                    best = (err, s, e, m, v)
    _, s, e, m, v = best
    return s, e, m, v

# 全部正档位
pos = sorted({fp4_decode(0, e, m) for e in range(4) for m in range(2)})
print("FP4 (E2M1) magnitudes:", pos)

# 就近取整演示
for x in [0.4, 0.9, 1.2, 2.6, 5.8, 7.0, -0.5, -4.1]:
    s, e, m, v = fp4_encode(x)
    print(f"{x:6.2f} -> bits s={s} e={e} m={m}  value={v:6.2f}")
```

输出可以看到 4-bit 的"粗"：`0.4` 被吸到 `0.5`，`1.2` 被吸到 `1.0`，`2.6` 被吸到 `3.0`——相邻档位之间隔着 0.5 或更大的缝。这还没算 NVFP4 的 block scale：真实推理里，先把每 16 个值除以各自的 E4M3 缩放因子，再进这个量化函数，最后在反量化时乘回去。实际工程中，就近取整（round-to-nearest）通常带权重/激活的 per-block 或 per-channel 缩放，并配合 outlier 处理（如 QQQ）来控制误差。

### 一些 tips

1. **别用"4-bit"去估精度**。有效精度是尾数 + 1 = 2 bit，约 0.6 位十进制；拿总位宽 4 去算会高估一倍（1.2），这是一个很容易混淆的坑。
2. **决定用 FP4 前先看数据分布**。有没有离群值？动态范围跨多少倍频？如果 layernorm 输出经常超出 `[-6,6]` 或分布太宽，FP4 会把大量值压成同一档，得先做 scaling / QQQ / QAT。
3. **NVFP4 ≠ 裸 FP4**。NVIDIA 说"用 NVFP4"，指的是带 E4M3 per-16 缩放 + FP32 二级缩放的 Blackwell 方案，精度比裸 E2M1 好；裸 FP4 需要你自己在软件里维护缩放因子。
4. **注意硬件代际**。FP4 需要 Blackwell（第五代 tensor core）；Ampere/Hopper 上跑到最低只能到 FP8。在旧卡上"假装用 FP4"意义不大。
5. **训练别盲目上 4-bit**。4-bit 主要兑现的是推理的显存与带宽收益；训练的正反向与梯度累加通常还是 bf16/fp16，FP4 更多出现在前向激活或离线量化权重。
6. **先量化误差再决定**。FP4 的 $u=0.25$ 意味着单次运算相对误差可到 ~25%，任何累积都会迅速放大——只有在"每步误差可容忍 + 需要极致吞吐"时才值得。

## Reference

- NVFP4 Trains with Precision of 16-bit and Speed and Efficiency of 4-bit（NVIDIA）：<https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/>
- Pretraining Large Language Models with NVFP4（arXiv:2509.25149）：<https://arxiv.org/abs/2509.25149>
- OCP Microscaling Formats (MX) Specification v1.0：<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- Microscaling Data Formats for Deep Learning（arXiv:2310.10537）：<https://arxiv.org/abs/2310.10537>
- NVIDIA Transformer Engine（FP4 / FP8 支持）：<https://github.com/NVIDIA/TransformerEngine>
