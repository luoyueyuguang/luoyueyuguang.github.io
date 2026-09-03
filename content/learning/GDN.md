这篇文章讲 **GDN**，也就是 **Gated DeltaNet / Gated Delta Networks**。

把 GDN 当“另一个 Transformer 变体”看会走偏。它更像把三条线合到一起：

- RNN 的固定状态。
- Attention / fast weight memory 的 key-value 读取。
- Delta rule 和 gating 的可编辑记忆。

从 RNN 一步步到 GDN：

```text
RNN
  -> Attention
  -> Linear Attention
  -> Fast Weight Memory
  -> DeltaNet
  -> Gated DeltaNet
```

> GDN 是一种把长上下文压进固定大小矩阵状态里的序列模型。它每来一个 token，就用门控决定旧记忆忘多少，再用 delta rule 精准改写记忆。

![从 RNN 到 GDN 的演化路线](/learning/assets/gdn-rnn-to-gdn.svg)

本文有两类图：

- **教学图**：为了从零解释概念重新画的简化图。
- **原文图转写**：根据论文 LaTeX 源码/TikZ/pgfplots 数据重绘为本地 SVG，便于博客直接展示。它们不是我凭空画的示意图，但也不是论文 PDF 的截图。

读论文时可以重点对照下面几张图：

| 原文图   | 内容                                                                              | 建议什么时候看                                                  |
| -------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Figure 1 | Gated DeltaNet、Gated DeltaNet-H1/H2 的整体架构，以及 Gated DeltaNet block design | 读完第 7-8 节后看，理解论文里的真实模块结构                     |
| Figure 2 | 六个长上下文 benchmark 上的 length extrapolation 曲线                             | 读第 13 节后看，理解 GDN 在长序列上的实验表现                   |
| Figure 3 | 1.3B 模型在单张 H100 上的训练吞吐对比                                             | 读第 9 节后看，理解为什么论文强调 chunkwise / parallel training |

> 原文图源：Gated Delta Networks: Improving Mamba2 with Delta Rule，arXiv:2412.06464，HTML 版本见 <https://ar5iv.labs.arxiv.org/html/2412.06464>，PDF 版本见 <https://arxiv.org/pdf/2412.06464>。

下面三张是根据论文源码重绘/转写到本地的版本。

![GDN 原论文 Figure 1 架构图转写](/learning/assets/gdn-paper-architecture.svg)

> 图源说明：根据论文 Figure 1 与 arXiv:2412.06464 的 LaTeX/TikZ 源码 `figures/model.tex` 转写。原图展示 Gated DeltaNet-H1、Gated DeltaNet-H2，以及 Gated DeltaNet block design。

![GDN 原论文 Figure 2 长上下文曲线转写](/learning/assets/gdn-paper-length-extrapolation.svg)

> 图源说明：根据论文 Figure 2 与 arXiv:2412.06464 的 pgfplots 坐标数据转写。为了适合博客阅读，这里保留六个 benchmark 的相对趋势和核心模型对比；精确曲线请以论文 PDF/源码为准。

![GDN 原论文 Figure 3 H100 训练吞吐转写](/learning/assets/gdn-paper-throughput.svg)

> 图源说明：根据论文 Figure 3 与 arXiv:2412.06464 的 pgfplots 坐标数据转写。横轴是 sequence length × batch size，纵轴是 thousands token per second。

这篇文章面向没接触过线性注意力和 SSM 的读者，先看一个问题：

> 模型读完很多 token 以后，过去的信息到底存在哪里？

Transformer 的答案是：存在越来越长的 KV cache 里。

RNN / SSM / GDN 这类模型的答案是：存在一个固定大小的状态里。

## 1. 从最朴素的 RNN 开始

RNN，全称 Recurrent Neural Network，循环神经网络。

它处理序列的方式很像人读文章：

```text
读第 1 个 token，更新脑中印象
读第 2 个 token，再更新脑中印象
读第 3 个 token，再更新脑中印象
...
```

数学上写成：

$$
h_t = f(x_t, h_{t-1})
$$

这里：

- `x_t` 是当前 token 的输入。
- `h_{t-1}` 是读到上一个 token 后留下来的状态。
- `h_t` 是读完当前 token 后的新状态。
- `f` 是一个神经网络更新函数。

可以把 `h_t` 想成“读到当前位置时，模型脑子里压缩出来的笔记”。

RNN 的优点：

- 每一步只需要一个状态 `h`。
- 内存不会随着序列长度无限增长。
- 处理第 `t` 个 token 时，不需要重新扫描所有历史 token。

但 RNN 也有一个老问题：

> 一个向量状态太小，历史信息被反复压缩，容易忘。

比如前文 10 万 token 之前出现了一个变量定义，后文才问这个变量的类型。朴素 RNN 要把这个信息一直藏在 `h_t` 里，中间经过很多次更新，很容易被冲掉。

所以 RNN 的核心矛盾是：

```text
状态固定大小，很省；
但状态太压缩，难以精确回忆。
```

## 2. Attention：不要压缩，直接查原文

Transformer 的 attention 换了一种思路：把每个历史 token 都留下来。

- 每个历史 token 产生一个 `Key`。
- 每个历史 token 产生一个 `Value`。
- 当前 token 产生一个 `Query`。

当前 query 会去和所有历史 key 比较，决定应该读哪些 value。

公式是：

$$
o_t =
\sum_{i \le t}
\mathrm{softmax}(q_t k_i^T)_i v_i
$$

```text
当前问题 q_t：
  去历史笔记里逐条查 key
  找到相关的条目
  按相关性读取对应 value
```

这就是 Transformer 强大的原因：它能重新访问完整历史。

但是代价也很直接：

| 模型        | 历史怎么存             | 长上下文代价            |
| ----------- | ---------------------- | ----------------------- |
| RNN         | 一个隐藏状态           | 便宜，但容易忘          |
| Transformer | 每个 token 的 KV cache | 准，但 cache 随长度增长 |

如果上下文有 `N` 个 token，KV cache 也大致随 `N` 增长。prefill 阶段的完整 attention 还会有接近 `N^2` 的交互。

![Attention 和循环状态的区别](/learning/assets/gdn-attention-vs-recurrence.svg)

所以大家自然会问：

> 有没有一种方法，既像 attention 一样能按 key 读取 value，又像 RNN 一样只维护固定大小状态？

Linear attention 就是这个方向。

## 3. Linear Attention：把历史压成一个“可查询矩阵”

普通 softmax attention 难以直接写成固定状态，是因为 softmax 里有所有历史 token 的归一化。

Linear attention 的思路是：把 attention kernel 改写成特征映射的内积。

先不纠结细节，可以把 softmax 相似度：

$$
\exp(q^T k)
$$

换成：

$$
\phi(q)^T \phi(k)
$$

这里 `phi` 是一个特征映射函数。

这样 attention 输出可以近似写成：

$$
o_t =
\frac{
\phi(q_t)^T \sum_{i \le t} \phi(k_i) v_i^T
}{
\phi(q_t)^T \sum_{i \le t} \phi(k_i)
}
$$

中间两项可以递推维护：

$$
S_t = S_{t-1} + \phi(k_t) v_t^T
$$

$$
z_t = z_{t-1} + \phi(k_t)
$$

然后当前输出只需要：

$$
o_t =
\frac{\phi(q_t)^T S_t}{\phi(q_t)^T z_t}
$$

这就是 “Transformers are RNNs” 这类工作的核心观察：某些 attention 可以写成 RNN 风格的状态更新。历史从 `N` 个 KV 条目，变成一个矩阵 `S_t` 和一个向量 `z_t`。

这个矩阵 `S_t` 是：

> 一个从 key 到 value 的可学习映射表。

当前 query 来了以后，查这个压缩后的映射表。

## 4. Fast Weight Memory：状态矩阵像一张临时字典

为了理解 DeltaNet 和 GDN，最好把 `S_t` 看成一张临时字典：

```text
key  -> value
```

每来一个 token，模型做一次写入：

```text
把当前 key 对应到当前 value
```

最简单的线性 attention 更新是加法：

$$
S_t = S_{t-1} + k_t v_t^T
$$

这里为了写得简单，我省略 `phi`，直接用 `k_t`。不同论文里会把状态写成 `S` 或 `W`，也会因为行向量/列向量约定不同，把外积写成 `k_t v_t^T` 或 `v_t k_t^T`。本文的原则是：**状态矩阵要能用 key 读出 value**。

这很像在字典里追加一条记忆。但问题是：如果相似的 key 出现多次，简单相加会把很多 value 混在一起。

比如模型先看到：

```text
key = "apple"
value = "苹果公司"
```

后面又看到：

```text
key = "apple"
value = "水果"
```

如果只是不断加，状态里可能同时混入两个含义。等 query 再问 `apple`，读出来的 value 可能混乱。

这就是普通线性 attention 的一个能力瓶颈：它写入很便宜，但改写不够精准。

DeltaNet 要解决的是这个问题。

## 5. DeltaNet：先看旧答案，再写入修正量

DeltaNet 的核心是 **delta rule**。

不要被名字吓到。这里的 delta 就是“差值”“误差”“残差”。

普通线性 attention 写入的是：

$$
v_t
$$

DeltaNet 写入的是：

$$
v_t - \bar{v}_t
$$

其中：

$$
\bar{v}_t = S_{t-1} k_t
$$

也就是旧状态在当前 key 上读出来的旧答案。

于是更新变成：

$$
S_t =
S_{t-1}
+ \beta_t (v_t - S_{t-1} k_t) k_t^T
$$

为了避免符号混乱，可以把它拆成三步：

$$
\text{old}_t = S_{t-1} k_t
$$

$$
\Delta_t = v_t - \text{old}_t
$$

$$
S_t = S_{t-1} + \beta_t \Delta_t k_t^T
$$

这里的外积 `\Delta_t k_t^T` 表示：沿着当前 key 的方向，把 value 误差写回状态矩阵。若你的实现采用相反的矩阵布局，公式会转置，但“先读旧答案，再写残差”这个核心不变。

这里：

- `S_{t-1}` 是旧记忆。
- `k_t` 是当前 key。
- `v_t` 是当前要写入的 value。
- `S_{t-1} k_t` 是旧记忆对这个 key 的预测。
- `v_t - S_{t-1} k_t` 是旧记忆还差多少。
- `beta_t` 是写入强度，相当于学习率或写门。

![Delta rule 的读、算残差、写回](/learning/assets/gdn-delta-update.svg)

这个更新很像在线学习：

```text
先问旧模型：这个 key 应该对应什么 value？
再看目标 value：真正应该是什么？
最后只把误差写进去。
```

好处是：

- 如果旧状态已经答对，就少写。
- 如果旧状态答错，就写入修正。
- 相比纯加法，更像是在维护一个 key-value 映射。

这也是为什么很多论文会把线性 attention 和 fast weight memory 联系起来：状态矩阵更像一个会在推理过程中不断被输入 token 编程的临时权重矩阵。

## 6. 只用 Delta 还不够：旧记忆什么时候该忘？

DeltaNet 让写入更精准，但它还缺一个重要能力：

> 有些旧记忆应该保留，有些旧记忆应该快速忘掉。

在自然语言里，同一个词或同一个位置的语义会不断变化。

比如一段代码里：

```text
config = load_default_config()
...
config = override_config(config, user_args)
...
config = sanitize(config)
```

模型不应该永远记住最早的 `config`。它应该知道当前上下文里更近、更有效的定义是什么。

这时就需要 **gating**。

门控的想法很简单：

```text
新 token 来了以后，先决定旧状态保留多少，再决定新信息写多少。
```

这和 LSTM / GRU 里的门很像：

- 有些信息要保留。
- 有些信息要忘掉。
- 有些新信息要写入。

GDN 就是把 gating 和 delta rule 放到一起。

## 7. Gated DeltaNet：门控遗忘 + Delta 改写

GDN 的核心更新可以用一个简化公式理解：

$$
\tilde{S}_{t-1} = \alpha_t S_{t-1}
$$

$$
S_t =
\tilde{S}_{t-1}
+ \beta_t
(v_t - \tilde{S}_{t-1} k_t)
k_t^T
$$

这里：

- `alpha_t` 是遗忘门，决定旧状态保留多少。
- `beta_t` 是写入门，决定这次 delta 写入多强。
- `v_t - \tilde{S}_{t-1} k_t` 是在遗忘后的状态上计算出来的残差。

论文里的等价写法更接近：

$$
S_t =
S_{t-1}\left(\alpha_t(I-\beta_t k_t k_t^T)\right)
+ \beta_t v_t k_t^T
$$

这个式子看起来更复杂，但含义和上面的三步一致：先按 `alpha_t` 控制旧状态，再用 delta rule 对当前 key 的映射做定向编辑。本文使用简化式，是为了让第一次接触的读者先抓住记忆更新逻辑。

如果 `alpha_t` 接近 1：

```text
旧记忆大多保留。
```

如果 `alpha_t` 接近 0：

```text
旧记忆被快速擦掉。
```

如果 `beta_t` 大：

```text
当前 token 对状态改写很强。
```

如果 `beta_t` 小：

```text
当前 token 只轻微影响状态。
```

![Gated DeltaNet 的遗忘门和 delta 写入](/learning/assets/gdn-gated-update.svg)

> 对照原文图：论文 Figure 1 画的是 Gated DeltaNet 的真实 block design。它显示 query/key/value 路径分别经过 linear projection、short convolution、SiLU，其中 query/key 还会做 L2 normalization；`alpha`、`beta` 由 linear projection 产生；输出侧还包含 normalization、output gate 和 output projection。本文上面的图只抽象出“遗忘门 + delta 写入”的核心逻辑，方便先理解公式。

GDN 的核心是：

> Gating 负责控制记忆寿命，delta rule 负责精准修改 key-value 映射。

GDN 论文的核心观察也正是：gating 和 delta rule 是互补的。gating 擅长快速擦除无用记忆，delta rule 擅长对已有映射做有目标的更新。

## 8. 为什么 GDN 和 Mamba2 有关系？

GDN 的论文标题是 **Gated Delta Networks: Improving Mamba2 with Delta Rule**。

这里的背景是 Mamba2 / SSD。

Mamba2 的重要思想是：很多 SSM、linear attention、attention-like 模型可以放到一个统一框架里理解。它们都可以看成某种结构化的序列混合，只是状态更新和矩阵结构不同。

从实现角度看，Mamba2 这类模型重视两件事：

- **线性复杂度**：处理长度为 `N` 的序列，计算量尽量随 `N` 线性增长。
- **硬件友好训练**：虽然递推看起来是一步一步的，但训练时要能 chunk-wise / parallel scan，否则 GPU 利用率会很差。

GDN 沿着这条路继续走：

- 它保留固定大小状态和线性序列混合的优势。
- 它把 DeltaNet 的 delta rule 引入 gated recurrence。
- 它设计了适合现代硬件的并行训练算法。

所以 GDN 比老式 RNN 多了几样东西：

```text
RNN 的固定状态思想
+ Attention 的 key-value 读取思想
+ Delta rule 的精准改写
+ Gating 的自适应遗忘
+ 现代 GPU 友好的并行训练
```

论文 Figure 1 还展示了两个 hybrid 版本：

- **Gated DeltaNet-H1**：Gated DeltaNet 层和 sliding window attention 层交替/组合。
- **Gated DeltaNet-H2**：把 Mamba2、Gated DeltaNet 和 sliding window attention 放进同一个混合架构。

这说明作者并没有把 GDN 当成“完全消灭 attention”的方案。更现实的做法是：用 GDN 处理大部分线性序列混合，再用局部 attention 补足局部比较、位移和精细检索能力。

这也是 Figure 1 里 H1/H2 存在的原因：纯 recurrent/linear 模型更擅长固定状态和长序列吞吐，sliding window attention 更擅长短距离精细对齐。混合架构把不同模块放在各自擅长的位置。

## 9. 训练时为什么不能直接一步步循环？

GDN 推理时很自然：

```text
for token in sequence:
    update state
    produce output
```

但训练时如果也逐 token 串行跑，就会很慢。

Transformer 训练快的一个重要原因是：整段序列可以并行算 attention。RNN 类模型如果完全串行，会吃亏。

所以 GDN 论文强调 parallel training algorithm。它会把长序列切成 chunks：

```text
chunk 1: token 1-256
chunk 2: token 257-512
chunk 3: token 513-768
...
```

每个 chunk 内部用更适合矩阵乘法的形式计算，chunk 之间再用 scan / recurrence 把状态接起来。

这类算法改变计算组织方式，不改变数学结果：

| 方式                       | 问题                             |
| -------------------------- | -------------------------------- |
| 逐 token Python 循环       | 太慢，GPU 利用率差               |
| chunk-wise / parallel scan | 更容易用大矩阵乘法和并行前缀计算 |

这也是现代 RNN-like 模型和 1990 年代 RNN 的一大区别：它们既提出递推公式，也提出能在 GPU 上训练的算法。

> 对照原文图：论文 Figure 3 给了 1.3B 模型在单张 H100 GPU 上的训练吞吐对比。论文正文说明，GDN 的 gating 项主要是 elementwise multiplication，不破坏 matmul 结构，因此 Gated DeltaNet 的速度和 DeltaNet 接近，只比 Mamba2 有小幅差距。这张图适合用来理解“为什么 chunkwise 形式和 tensor core 友好很重要”。

你可以回看本文开头的 Figure 3 转写图：Transformer++ 在短上下文里很快，但随着序列长度变长吞吐会掉下来；Gated DeltaNet、DeltaNet、Mamba2 这类 recurrent/SSM-like 模型曲线更平。这就是“线性序列混合”在长上下文训练里的工程价值。

## 10. Decode 时 GDN 真的总是更快吗？

不一定。

Transformer decode 时每生成一个 token，需要读取越来越长的 KV cache。GDN 看起来只维护固定大小状态，似乎一定更快。

但实际系统里还要看状态大小和访存。

GDN 的状态通常是每层、每个 head 的矩阵状态。每生成一个 token，都可能要从 HBM 读出状态、更新状态、再写回状态。

所以 batch size 很小、状态矩阵很大时，GDN decode 可能变成 memory-bound：

```text
算术不多，但每步都要搬很多状态矩阵。
```

2026 年有一篇关于 GDN decode 加速的 FPGA 工作就指出，GDN 用固定 recurrent state 替代增长的 KV cache，但 batch-1 GPU decode 会被状态往返 HBM 的访存卡住。它们的思路是把状态常驻在 FPGA 片上 BRAM 里，减少每 token 的状态搬运。

那篇工作把瓶颈讲得很具体：所有 subquadratic 序列模型在 decode 时算术强度都低于 **1 FLOP/B**，也就是每搬一个字节还做不到一次乘加，因此比普通 Transformer 更受限于访存。GDN 的 recurrent state 在它们的配置里约 **2 MB**，每个 token 都要整块读、改、写回。把状态常驻到 FPGA 片上 BRAM 后，它们测得 **63 us / token**，比 H100 PCIe 上的 GPU 参考实现快 **4.5x**，片上功耗约 9.96 W，折算下来每个 token 的能量效率最高提升约 **60x**。

这说明 GDN 的工程判断要更细：

- 长上下文下，它避免了 KV cache 随 `N` 增长。
- 但每步固定状态本身也不小。
- 实际速度取决于硬件、batch size、状态维度、kernel fusion 和数据驻留策略。

所以不要把 GDN 简化成“固定状态，所以一定比 Transformer decode 快”。

> GDN 把随上下文增长的 KV cache 问题，换成了固定大小状态的读写和更新问题。

这通常更适合长上下文，但仍然需要好的实现。

工程上关键的是：状态能不能尽量留在片上，或者至少通过 fused kernel 减少 HBM 往返。如果每个 token 都把大状态矩阵完整读出、更新、写回，batch-1 decode 仍然会被内存带宽限制。

## 11. 和其他模型对比

先放一张表：

| 方法             | 历史怎么存                    | 优点                                 | 主要问题                                 |
| ---------------- | ----------------------------- | ------------------------------------ | ---------------------------------------- |
| RNN              | 一个向量 `h_t`                | 内存固定，推理简单                   | 记忆容量小，长距离回忆弱                 |
| LSTM / GRU       | 带门控的向量状态              | 比朴素 RNN 更会保留/遗忘             | 状态仍是向量，表达受限                   |
| Transformer      | 所有历史 KV cache             | 精确检索，效果强                     | cache 随上下文增长，attention prefill 贵 |
| Linear Attention | 固定矩阵状态                  | 线性复杂度，可写成 recurrence        | 简单加法写入容易混淆记忆                 |
| DeltaNet         | 固定矩阵状态 + delta update   | 能修正已有 key-value 映射            | 还需要更强的遗忘控制                     |
| Mamba2           | SSD / selective SSM           | 统一 SSM 和 attention 视角，硬件友好 | 长距离精确检索仍是挑战                   |
| GDN              | 门控 + delta update           | 同时有快速遗忘和精准改写             | 状态矩阵 decode 访存仍需优化             |
| GDN-2            | 分离 erase gate 和 write gate | 进一步解耦擦除和写入                 | 结构更复杂，仍是较新的方向               |

## 12. GDN vs Transformer

Transformer 的关键优势是精确访问历史。

如果某个信息在 8 万 token 前，Transformer 理论上可以通过 attention 直接看过去。代价是 KV cache 要保存下来，而且 attention 计算和访存都和上下文长度强相关。

GDN 的关键优势是固定状态。

历史信息被压进状态矩阵里。生成第 `t` 个 token 时，不需要保留长度为 `t` 的 KV cache。代价是：历史已经被压缩，不是所有细节都能原样恢复。

```text
Transformer 更像开卷查原文；
GDN 更像维护一张不断更新的压缩索引表。
```

## 13. GDN vs Mamba2

Mamba2 提供了一个重要的统一视角：SSM 和 attention 可以通过结构化矩阵联系起来。

GDN 和 Mamba2 的关系可以这样理解：

- Mamba2 强调 selective SSM / SSD 形式。
- GDN 在这个大框架里强调 gated delta rule。
- GDN 论文报告它在语言建模、常识推理、in-context retrieval、长度外推、长上下文理解等任务上超过 Mamba2 和 DeltaNet。

```text
Mamba2 给出一种高效状态空间序列混合；
GDN 往里面加入更像“可编辑 key-value 记忆”的 delta 更新。
```

> 对照原文图：论文 Figure 2 展示了六个长上下文 benchmark 的 length extrapolation 曲线。作者的结论是，在 RNN 类模型中，Gated DeltaNet 整体 perplexity 更低，长序列上更稳健；hybrid 版本借助 attention 做局部上下文建模，还能进一步改善表现。

这张图别只看某一个点，要看趋势：随着长度从 4K 增到 20K，普通 recurrent/SSM 模型容易出现记忆管理问题，Gated DeltaNet 系列通常更稳。这个结果和前面的机制解释是对得上的：gating 帮助清理旧记忆，delta rule 帮助把当前 key 对应的 value 改写得更准。

## 14. GDN vs DeltaNet

DeltaNet 的核心是：

```text
写入 v - old_read
```

它解决的是“如何精准改写记忆”。

GDN 在 DeltaNet 上加了 gating，解决的是：

```text
旧记忆什么时候该保留，什么时候该快速擦除？
```

所以两者是递进关系：

```text
Linear Attention: 直接加法写入
DeltaNet: 写入残差，减少混淆
GDN: 先门控遗忘，再写入残差
```

## 15. GDN-2 又改了什么？

截至 2026 年 6 月，GDN 已经有后续工作 **Gated DeltaNet-2**。

GDN-2 指出一个问题：原来的 active edit 用一个标量 gate 同时控制两件事：

- key 侧擦除多少旧内容。
- value 侧提交多少新内容。

但擦除和写入不一定应该绑定。

比如：

```text
旧内容明显过时，需要强擦除；
但新内容不一定可靠，只应该弱写入。
```

或者：

```text
旧内容还不错，不该大幅擦除；
但当前 token 很关键，应该强写入补充。
```

GDN-2 的思路是把两个 gate 拆开：

- erase gate：控制擦除。
- write gate：控制写入。

这让记忆编辑更灵活。代价是模型和实现更复杂。

这类后续工作也说明：GDN 的核心是一条开放的方向。它把线性 attention 的状态矩阵当作“可编辑记忆”，后续又在此基础上研究擦除、写入、归一化和曲率预条件这些机制。

## 16. 一个完整例子：模型读代码时发生什么

假设模型正在读一段代码：

```python
timeout = 30
...
timeout = config.request_timeout
...
client = Client(timeout=timeout)
```

我们用 GDN 的视角看。

读到第一行：

```python
timeout = 30
```

模型把类似这样的映射写进状态：

```text
key: timeout
value: 30
```

读到第二次赋值：

```python
timeout = config.request_timeout
```

如果状态里已经有 `timeout -> 30`，DeltaNet / GDN 会先读旧答案：

```text
old_read = 30
target = config.request_timeout
delta = target - old_read
```

然后写入修正量。

如果门控判断旧定义已经过时，`alpha_t` 会让旧状态快速衰减。这样后面读到：

```python
client = Client(timeout=timeout)
```

模型更可能把 `timeout` 理解成新的 `config.request_timeout`，而不是最早的 `30`。

当然真实模型存的是高维向量。GDN 在维护一张会随上下文不断改写的压缩记忆表。

## 17. 这篇文章最值得带走的点

GDN 可以从 RNN 一步步理解：

1. **RNN**：把历史压进一个固定状态，但状态容量有限。
2. **Attention**：保留完整 KV cache，能精确查历史，但长上下文成本高。
3. **Linear Attention**：把历史压成矩阵状态，让 attention 变成 recurrence。
4. **Fast Weight Memory**：矩阵状态可以看成 key 到 value 的临时映射。
5. **DeltaNet**：写入前先读旧答案，再写入残差，让记忆修改更精准。
6. **GDN**：加入门控，让模型既能快速忘掉旧记忆，也能用 delta rule 精准写入新记忆。
7. **工程上**：GDN 避免 KV cache 随上下文增长，但状态矩阵的读写仍然是 decode 性能关键。

> GDN 是一种 RNN-like 的线性序列模型：它用固定大小矩阵状态代替不断增长的 KV cache，用 gating 控制遗忘，用 delta rule 控制精准改写，从而在长上下文里取得比普通线性注意力更强的记忆更新能力。

## Reference

核心论文：

- Gated Delta Networks: Improving Mamba2 with Delta Rule: <https://arxiv.org/abs/2412.06464>
- Gated Delta Networks HTML with figures: <https://ar5iv.labs.arxiv.org/html/2412.06464>
- Gated Delta Networks PDF: <https://arxiv.org/pdf/2412.06464>
- Gated Delta Networks arXiv source used for figure redraws: <https://arxiv.org/e-print/2412.06464>
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length: <https://arxiv.org/abs/2406.06484>
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention: <https://arxiv.org/abs/2006.16236>
- Linear Transformers Are Secretly Fast Weight Programmers: <https://arxiv.org/abs/2102.11174>
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality: <https://arxiv.org/abs/2405.21060>

后续和工程相关：

- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention: <https://arxiv.org/abs/2605.22791>
- A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA: <https://arxiv.org/abs/2603.05931>
- Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences: <https://arxiv.org/abs/2604.21100>
