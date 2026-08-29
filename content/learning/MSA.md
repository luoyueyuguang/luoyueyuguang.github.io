Coauthor with codex 5.5

这篇文章讲 MiniMax Sparse Attention，也就是论文里简称的 **MSA**。

> MSA 先用一个很便宜的小分支给上下文分块打分，只挑出少量最可能有用的块，再让主注意力分支认真阅读这些块。

如果把大模型读长文档比作考试：

- 普通 full attention 像每做一道题都从第一页翻到最后一页。
- Sliding window attention 像只看题目附近几页。
- MSA 像先用目录和索引快速判断哪些章节相关，然后只精读这些章节。

论文的核心目标是：在百万 token 级别长上下文里，保留 softmax attention 的表达能力，同时把注意力计算从“看全部上下文”变成“看少量动态选中的上下文块”。

![MSA 整体流程](/learning/assets/msa-overview.svg)

原论文也给了一张总览图，左边是 Index Branch，右边是 Main Branch。博客里的示意图是为了新手理解重新画的；下面这张是论文原图：

![MiniMax Sparse Attention 原论文架构图](/learning/assets/msa-paper-architecture.png)

> 图源：MiniMax Sparse Attention 论文 Figure 1，原始图片来自 <https://arxiv.org/html/2606.13392v2>。

论文里的主要数字：

- 在 1M context 下，MSA 的每 token attention FLOPs 比 GQA 少 **28.4x**。
- 结合专门设计的 kernel，在 H800 上实现了 **14.2x prefill** 和 **7.6x decoding** 的端到端 attention 加速。
- 论文在一个 **109B 参数、每 token 激活 6B 参数** 的 MoE 多模态模型上做实验，MSA 效果和 full attention GQA 基本相当。
- 论文公开了 inference kernel：<https://github.com/MiniMax-AI/MSA>。
- 使用 MSA 的公开模型是 MiniMax-M3：<https://huggingface.co/MiniMaxAI/MiniMax-M3>。

想快速抓住全文，记住这条线：

```text
长上下文 full attention 太贵
  -> GQA 先减少 KV head，但仍然看全部 token
  -> MSA 再把 KV tokens 切成 blocks
  -> Index Branch 给 blocks 打分并选 Top-K
  -> Main Branch 只在选中的 blocks 上做标准 softmax attention
  -> 专门的 TopK / KV-outer / combine kernels 把理论稀疏变成真实加速
```

这篇文章很长，可以按不同目的读：

| 你想知道什么 | 建议读法 |
|---|---|
| 只想理解 MSA 是什么 | 读 1-9 节，再读 21 节总结 |
| 想理解训练为什么能稳定 | 读 10-13 节 |
| 想理解为什么能省计算 | 读 14-15.10 节 |
| 想对着公开仓库看 kernel 源码 | 重点读 15.11-15.19 节 |
| 想知道它和其他长上下文方法的区别 | 读 18 节 |

先放一个符号表，后面会反复出现：

| 符号 | 含义 | 直觉 |
|---|---|---|
| `N` | 序列长度 / 上下文 token 数 | 一篇长文有多少个 token |
| `H_q` | query head 数 | 有多少个“提问视角” |
| `H_kv` | key/value head 数 | 有多少组可被检索的信息 |
| `G = H_q / H_kv` | 每个 KV head 对应多少个 query heads | GQA group 大小 |
| `d_h` | 每个 head 的维度 | 每个 head 的向量宽度 |
| `B_k` | KV block size | 一个 block 里放多少个 KV tokens |
| `k` | 每个 query/group 选多少个 KV blocks | MSA 的稀疏预算 |
| `q2k` | query 到 KV blocks 的索引 | “这个 query 要看哪些 blocks” |
| `k2q` | KV block 到 queries 的反向索引 | “这个 block 被哪些 queries 看中” |
| `LSE` | log-sum-exp | softmax 分母的稳定表示 |

还有两个部署词也提前解释一下：

| 词 | 含义 |
|---|---|
| Prefill | 模型第一次读入整段 prompt，把 prompt 里的 token 全部处理一遍。长上下文场景最容易在这里爆成本 |
| Decoding | 模型已经读完 prompt 后，一个 token 一个 token 往外生成。这里通常受 KV cache 读取、batch size、生成长度等因素影响 |

下面从零开始讲。

## 1. 为什么长上下文会让大模型变慢

大模型生成一个 token 时，不是只看最后几个字。它要回头看前面已经出现过的 token，判断哪些信息对当前 token 有用。

比如上下文很短时：

```text
用户：请总结这段话：量化可以降低显存占用。
模型：这段话的意思是……
```

模型只需要看几十个 token。

但是长上下文场景会变成：

```text
前面有 100 万个 token 的代码仓库、文档、聊天记录、工具调用日志……
现在问：这个 bug 是哪一行引入的？
```

如果模型每一步都对 100 万个 token 做完整注意力，成本会非常高。

注意力最麻烦的地方是：每个 query token 都要和很多 key token 比较。序列长度记作 `N`，完整 causal attention 的计算量大致随 `N^2` 增长。

可以用一个小表感受平方增长：

| 上下文长度 | 完整注意力大概要比较多少对 token |
|---:|---:|
| 1K | 约 100 万 |
| 10K | 约 1 亿 |
| 100K | 约 100 亿 |
| 1M | 约 1 万亿 |

长度增加 10 倍，注意力比较约增加 100 倍。

这就是长上下文部署难的根本原因。

## 2. 注意力到底在做什么

先不要看公式，把注意力想成一个信息检索过程。

每个 token 会产生三类向量：

- **Query**：我现在想找什么信息。
- **Key**：我这里有什么信息，可以被别人检索。
- **Value**：如果别人觉得我重要，就实际取走什么内容。

例如模型正在生成一句话：

```text
张三把钥匙放进抽屉。过了一会儿，他想开门，于是去找……
```

当模型要生成“钥匙”相关内容时，当前 query 会去和前文每个 key 做匹配。匹配分数高的位置，value 就会被更多读入。

用公式写就是：

$$
\text{score}_{i,j} =
\frac{Q_i K_j^T}{\sqrt{d_h}}
$$

这里：

- `i` 是当前 query token 的位置。
- `j` 是前文 key token 的位置。
- `d_h` 是一个 attention head 的维度。
- 分母 `sqrt(d_h)` 是缩放，防止分数过大。

算完所有分数后，模型会做 softmax：

$$
\alpha_{i,j} =
\frac{\exp(\text{score}_{i,j})}
{\sum_{u \le i}\exp(\text{score}_{i,u})}
$$

然后按权重加权 value：

$$
O_i = \sum_{j \le i} \alpha_{i,j} V_j
$$

1. 当前 token 拿着 query 去问：前面每个位置和我有多相关？
2. softmax 把相关性分数变成比例。
3. 模型按比例读取对应 value。

![完整注意力需要看所有历史 token](/learning/assets/msa-full-attention.svg)

完整 attention 的好处是信息最充分。坏处是太贵：上下文越长，每个 token 要看的历史位置越多。

## 3. GQA：先把 KV head 变少

MSA 建在 **GQA** 上，所以先解释 GQA。

传统 multi-head attention 里，每个 query head 通常有自己的 key/value head。GQA 的做法是：多个 query head 共享一组 key/value head。

假设：

- Query heads 有 `H_q = 64` 个。
- KV heads 有 `H_kv = 4` 个。

那么每个 KV head 服务：

$$
G = H_q / H_{kv} = 64 / 4 = 16
$$

也就是 16 个 query head 共享同一个 KV head。论文里把这叫一个 **GQA group**。

![GQA 把多个 query head 分到同一个 KV group](/learning/assets/msa-gqa.svg)

GQA 的意义是减少 KV cache 和 KV 读写。对长上下文推理来说，KV cache 非常大，减少 KV head 数量很有价值。

但是注意：GQA 主要减少的是 KV 的存储和读取规模，并没有从根本上改变“每个 query 仍然要看长上下文”的问题。MSA 要做的是再往前走一步：**每个 GQA group 动态选择少量 KV blocks。**

## 4. 稀疏注意力：不看全部，只看一部分

完整 attention 是：

```text
当前 query -> 看所有历史 key/value
```

稀疏 attention 是：

```text
当前 query -> 先选出一小部分历史 key/value -> 只对这些位置做 attention
```

论文把稀疏 attention 拆成两个阶段：

$$
\mathcal{I}_i = \mathrm{Index}_{\phi}(q_i, K_{\le i})
$$

$$
o_i = \mathrm{Attn}(q_i, K[\mathcal{I}_i], V[\mathcal{I}_i])
$$

- `Index` 阶段：为当前 query 选出要看的位置集合 `I_i`。
- `Attn` 阶段：只在这些被选中的位置上做标准 attention。

稀疏 attention 的难点不在“少看”，而在“少看哪些”。

如果选错了重要位置，模型质量会掉。如果选择过程本身太复杂，省下来的 attention 成本又会被选择成本吃掉。

MSA 的设计就是围绕这两个问题展开：

- 选择要足够准。
- 选择和后续 attention kernel 都要足够快。

## 5. 为什么 MSA 按“块”选，而不是按 token 选

最细粒度的做法是给每个历史 token 单独打分，然后选 Top-K token。

问题是 GPU 不喜欢这种零散访问。

GPU 擅长的是规则的大矩阵计算。一个 token 一个 token 地跳着读，会带来很多不连续内存访问和调度开销。即使理论 FLOPs 少，实际 wall-clock 也可能不快。

MSA 选择的是 **KV block**。

假设 block size 是 `B_k = 128`，上下文被切成：

```text
Block 1: token 1   - 128
Block 2: token 129 - 256
Block 3: token 257 - 384
...
```

当前 query 不再选择单个 token，而是选择若干个 block。论文主实验用：

```text
B_k = 128
k = 16
```

也就是每个 query、每个 GQA group 选择 16 个 KV block。最多阅读：

$$
kB_k = 16 \times 128 = 2048
$$

个 token。

这句话很关键：

> 即使上下文有 100 万 token，MSA 的主 attention 分支每个 query 仍然只读约 2048 个 token。

这就是 MSA 的计算量能随着长上下文拉开差距的原因。

![MSA 按 KV block 选择，而不是逐 token 选择](/learning/assets/msa-block-selection.svg)

按块选择有一个权衡：

- 优点：内存更连续，GPU 更容易跑快，Top-K 候选数量也变少。
- 缺点：粒度更粗，可能一个 block 里只有几个 token 真有用，但整个 block 都会被读。

论文的实验说明，在它们的设置里 `B_k = 128` 是一个比较实用的选择。附录里的 block size 消融也显示，在若干测试中把 block 从 32 增到 64 或 128，对质量影响有限，但更大的 block 更利于 kernel 效率。

## 6. MSA 的两个分支

MSA 每层 attention 有两个逻辑分支：

- **Index Branch**：轻量索引分支，只负责给 KV blocks 打分并选 Top-K。
- **Main Branch**：主 attention 分支，对选中的 blocks 做正常 softmax attention。

注意：Index Branch 很轻，论文说它只在标准 GQA 上额外加入两个投影矩阵：

$$
Q^{idx} = X W_q^{idx}
$$

$$
K^{idx} = X W_k^{idx}
$$

其中：

- `X` 是当前层输入 hidden states。
- `Q_idx` 是索引用 query。
- `K_idx` 是索引用 key。
- `W_q_idx` 和 `W_k_idx` 是新增参数。

MSA 的 Index Branch 有两个重要形状设计：

$$
Q^{idx} \in \mathbb{R}^{N \times H_{kv} \times d_{idx}}
$$

$$
K^{idx} \in \mathbb{R}^{N \times 1 \times d_{idx}}
$$

- 每个 GQA group 有自己的 index query。
- 所有 group 共享一个 index key。
- `d_idx` 通常比主 attention 的规模更轻。

这样 Index Branch 可以做到“每个 GQA group 单独选块”，但仍然保持轻量。

![Index Branch 和 Main Branch 的分工](/learning/assets/msa-two-branches.svg)

## 7. Index Branch 怎么给 block 打分

对当前 query token `i` 和 GQA group `r`，Index Branch 先算 token 级别分数：

$$
S^{idx,(r)}_{i,j}
=
\frac{(Q^{idx})^{(r)}_i (K^{idx})^T_j}{\sqrt{d_{idx}}}
$$

这里 `j` 是历史 token 位置，并且要满足 causal mask：

$$
j \le i
$$

也就是当前位置不能偷看未来。

然后它把 token 分数聚合到 block 分数。MSA 用的是 **max pooling**：

$$
M^{idx,(r)}_{i,b}
=
\max_{j \in \mathcal{B}_b, j \le i}
S^{idx,(r)}_{i,j}
$$

意思是：

- 对第 `b` 个 block 里的所有可见 token 计算分数。
- 取最大值当作这个 block 的分数。

为什么用最大值？

因为一个 block 里只要有一个 token 和当前 query 很相关，这个 block 就值得被读。用平均值可能会把少数重要 token 淹没掉。

最后做 Top-K：

$$
\mathcal{I}^{(r)}_i =
\mathrm{TopK}(M^{idx,(r)}_{i,\cdot}, k)
$$

得到当前 query、当前 GQA group 要看的 `k` 个 block。

可以用伪代码理解：

```text
for each query token i:
  for each GQA group r:
    for each visible KV block b:
      block_score[b] = max(score(i, token_j) for token_j in block b and j <= i)

    selected_blocks = top_k(block_score, k)
    selected_blocks must include local block
    output = normal_attention(query_i, K/V inside selected_blocks)
```

还有一个稳定性设计：**local block 一定会被选中**。

local block 就是包含当前 query token 的那个 block。这样可以防止索引分支早期乱选时，把当前位置附近的直接上下文漏掉。

## 8. Main Branch 仍然是标准 softmax attention

MSA 容易被误解成“近似 attention”。更准确地说：

> MSA 近似的是 attention 的候选范围；一旦 blocks 被选中，Main Branch 在这些 blocks 内仍然做标准 softmax attention。

对某个 query head `h`，如果它属于 GQA group `r`，Main Branch 做：

$$
O^{(h)}_i =
\mathrm{softmax}
\left(
\frac{Q^{(h)}_i (K^{(r)}[\mathcal{I}^{(r)}_i])^T}{\sqrt{d_h}}
\right)
V^{(r)}[\mathcal{I}^{(r)}_i]
$$

这个公式看起来长，其实意思很简单：

- `I_i^(r)` 是 Index Branch 选出的 blocks。
- 从这些 blocks 里取出对应的 K/V。
- 用普通 attention 算输出。

所以 MSA 不是把 softmax 换成线性 attention，也不是状态空间模型。它保留了 softmax attention，只是把 full context 换成 selected blocks。

## 9. 为什么 Top-K 可以不做 softmax

论文 kernel 里有一个小但重要的点：**选择 Top-K 时不需要先做 softmax**。

原因是 softmax 不改变排序。

如果：

$$
s_i \le s_j
$$

那么：

$$
\mathrm{softmax}(s)_i \le \mathrm{softmax}(s)_j
$$

也就是说，原始分数最大的那些位置，softmax 后仍然最大。

所以如果只是为了选 Top-K blocks，可以直接对 raw score 排序，不用做：

```text
max -> exp -> sum -> divide
```

这就是论文里说的 **exp-free Top-K selection**。它跳过了指数运算和归一化，减少索引阶段开销。

## 10. 训练难点：Top-K 不可导

MSA 最大的训练难点是：Top-K 是离散选择。

模型选了这些 block：

```text
[3, 9, 12, 28, ...]
```

这个选择本身不是一个平滑函数。普通反向传播很难直接告诉 Index Branch：

```text
你刚才应该多选 block 7，少选 block 12。
```

如果只靠语言模型 loss，Index Branch 收到的训练信号会很弱。论文附录也说明：只有 LM Loss 时，短上下文能力还行，但长上下文检索表现不好，因为 indexer 没有直接压力去学会选相关 block。

MSA 的做法是加一个辅助监督：**KL loss**。

## 11. KL Loss：让小索引分支模仿主分支

KL loss 就是"分布对齐损失"。

在 MSA 里：

- Main Branch 对选中的 token 算出 attention 分布。
- Index Branch 对同一批 token 也有自己的分数分布。
- KL loss 让 Index Branch 的分布接近 Main Branch。

也就是说，主分支像老师，索引分支像学生。

老师告诉学生：

```text
在你选中的这些 token 里，我真正关注的是这些位置。
下次你选 blocks 时，要更像我的关注模式。
```

公式写成：

$$
\mathcal{L}_{KL}
=
\frac{1}{NH_{kv}}
\sum_{i=1}^{N}
\sum_{r=1}^{H_{kv}}
D_{KL}
(
\mathrm{stopgrad}(P^{(r)}_{i,\cdot})
\parallel
P^{idx,(r)}_{i,\cdot}
)
$$

这里：

- `P` 是 Main Branch 的注意力分布，作为 teacher。
- `P_idx` 是 Index Branch 的分布，作为 student。
- `stopgrad(P)` 表示不要让 KL loss 更新 teacher，只更新 student。

![KL loss 让索引分支学习主分支关注模式](/learning/assets/msa-kl-training.svg)

## 12. 为什么要 stop-gradient

这篇论文里 stop-gradient 很关键。

如果不做限制，KL loss 不只会训练 Index Branch，还可能反向影响 backbone 和 Main Branch。这样会出现一个问题：模型为了降低 KL loss，可能让 Main Branch 的注意力分布变简单，而不是让 Index Branch 变聪明。

这有点像学生答不对题时，把老师的标准答案改得更简单。

论文附录里提到，不 detach 时会观察到两类问题：

- KL 梯度可能传进 backbone，导致 gradient norm spike 和 LM loss 发散。
- 即使训练不发散，也可能让短上下文 benchmark 退化。

所以 MSA 把 KL loss 限制成局部监督：

$$
Q^{idx} = \mathrm{stopgrad}(X) W_q^{idx}
$$

$$
K^{idx} = \mathrm{stopgrad}(X) W_k^{idx}
$$

这样 KL loss 更新的主要是：

- `W_q_idx`
- `W_k_idx`

而不是通过 `X` 影响整个模型主干。

> KL loss 只训练“索引器怎么找书页”，不要改变“模型怎么理解内容”。

## 13. 为什么需要 indexer warmup

如果从第 0 步就让一个还没学会的 Index Branch 控制路由，会很危险。

训练早期，Main Branch 的 attention 分布变化很快。论文附录里说，注意力熵会在很早阶段从平滑分布快速变尖锐。此时 Index Branch 还近似随机，如果立刻用 Top-K sparse selection，就可能把主分支带去错误的 blocks。

MSA 用两阶段训练：

1. **Warmup 阶段**：Main Branch 仍然跑 full attention，Index Branch 用 KL loss 学习。
2. **Sparse 阶段**：启用 Top-K sparse attention，继续用 KL loss 对齐。

在主实验里：

- MSA-PT 从头训练，先做 **40B tokens indexer warmup**，再进入 sparse pretraining。
- MSA-CPT 从 2.6T tokens 的 full-attention checkpoint 开始，替换成 MSA 后继续训练 400B tokens，其中前 **40B tokens** 是 indexer warmup。

这相当于先让索引器看一段时间标准答案，再让它正式上岗。

## 14. MSA 的计算量为什么会下降

普通 GQA 的 attention FLOPs 大致是：

$$
F_{GQA}(N) = 2H_q d_h N^2
$$

这里核心是 `N^2`。

MSA 的 FLOPs 分成两部分：

$$
F_{MSA}(N)
=
\underbrace{H_{kv}d_{idx}N^2}_{Index\ Branch}
+
\underbrace{4H_qd_hNkB_k}_{Main\ Branch}
$$

第一项还是 `N^2`，但很轻，因为：

- `H_kv` 远小于 `H_q`。
- `d_idx` 是轻量索引维度。

第二项是主 attention 分支，它不再是 `N^2`，而是：

$$
N \times kB_k
$$

当 `kB_k << N` 时，节省会很明显。

代入论文主设置：

```text
N = 1,000,000
k = 16
B_k = 128
kB_k = 2048
```

full attention 的主分支要看近百万历史 token，而 MSA 的 Main Branch 只看约 2048 个 token。随着 `N` 越大，这个差距越大。

![GQA 和 MSA 的计算量增长方式](/learning/assets/msa-complexity.svg)

当然，这不是说实际速度能达到百万除以 2048 那么夸张。因为 MSA 还要做索引、Top-K、query gather、反向索引、load balancing 等额外工作。论文也强调：实际 wall-clock speedup 会小于理论 FLOPs reduction。

但在长上下文下，dense attention 的成本持续随完整上下文增长，而 MSA 的主 attention 预算固定，所以实际加速会随上下文变长而增加。

## 15. Kernel 实现：理论稀疏怎么变成真实加速

到这里，算法层面已经清楚了：MSA 每个 query 先选 `k` 个 KV blocks。可是 GPU 上有一个很现实的问题：

> 少算 FLOPs 不等于一定跑得快。  
> 如果访存很乱、线程负载不均、矩阵形状太小，GPU 仍然可能很慢。

所以论文第 4 节专门讲 kernel design。这里我把它拆成从入门到实现的版本。

先解释几个会反复出现的 GPU 词：

| 词 | 可以先这样理解 |
|---|---|
| HBM / global memory | GPU 显存，容量大，但访问比片上存储慢 |
| shared memory | 一个 CUDA thread block 内共享的片上高速缓存 |
| register | 每个线程自己的最快存储，但数量有限 |
| warp | NVIDIA GPU 上 32 个线程一起执行的一组线程 |
| CTA | CUDA thread block，论文里说 CTA 时基本相当于一个线程块 |
| tensor core / MMA | 专门做矩阵乘法的小硬件单元，形状合适时非常快 |
| TMA | Tensor Memory Accelerator，Hopper/Blackwell 这类 GPU 上用于高效搬运多维 tensor tile 的机制 |
| LSE | log-sum-exp，FlashAttention 类 softmax kernel 用它稳定地合并 softmax 分母 |
| atomic | 多个线程同时更新同一个地址时用的同步写操作，能保证正确，但热点多时会很贵 |

MSA kernel 要解决三件事：

1. **TopK 怎么快**：Index Branch 给每个 query/group 打出很多 block score，必须快速选出 Top-K blocks。
2. **selected KV blocks 怎么算 attention**：不能让 K/V 读取得太碎，也不能让 tensor core 空转。
3. **训练时 KL loss 怎么便宜**：KL 需要一些 softmax 的 LSE 信息，最好不要额外再跑一遍 forward。

### 15.1 TopK kernel：先把“选块”做快

Index Branch 会产生每个 query 对每个 block 的分数。记住，MSA 选的是 KV block。

例如：

```text
query i, GQA group r:
  block 0 score = 1.2
  block 1 score = -0.4
  block 2 score = 3.7
  ...
  block B score = 0.8
```

现在要从几千个 block 里选出最大的 `k=16` 个。

最直接的写法可能是：

```python
prob = softmax(block_scores)
top_blocks = topk(prob, k=16)
```

但论文指出这里可以省掉 softmax：softmax 不会改变大小顺序。

如果原始分数里：

```text
score A > score B
```

那么 softmax 后仍然有：

```text
softmax(score) A > softmax(score) B
```

所以 TopK 只需要看 raw score，不需要先做 `max -> exp -> sum -> divide`。这叫 **exp-free selection**。

更具体地，论文的 TopK kernel 做了这些事：

1. 采用论文主设置 `B_k=128`、`k=16`。
2. 一个 warp 有 32 个 lanes，每个 lane 扫描一行 block scores 的 `1/32`。
3. 每个 lane 维护一个大小为 `k` 的局部 min-heap。
4. heap root 缓存在 register 里，减少 shared memory 访问。
5. 插入 heap 时延迟写回，进一步减少 shared memory 写入。
6. 最后用 `k` 轮 warp shuffle，把 32 个 lane 的局部 TopK 合并成整行 TopK。
7. shared memory 布局让每个 lane 尽量固定落在自己的 bank，避免 bank conflict。

![MSA TopK kernel 示意](/learning/assets/msa-topk-kernel.svg)

这个设计针对的是“小 k、大量行”的场景。通用排序或通用 TopK 可能更灵活，但会为很多 MSA 不需要的情况付开销。MSA 只要 unsorted top-k indices，也就是只要“哪 16 个块最大”，不要求它们内部排好序。

论文 Table 1 里给了 TopK kernel benchmark。下面是原表的关键信息，单位是 `us`，测试硬件是 H800，输入是 fp32，结果取 warmup 后 50 次的 median：

| Seq Len | Blocks | k | torch.topk | TileLang | MSA kernel | vs torch | vs TileLang |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 1024 | 16 | 3970 | 2864 | 779 | 5.1x | 3.7x |
| 128K | 2048 | 32 | 5378 | 3630 | 1991 | 2.7x | 1.8x |
| 512K | 4096 | 16 | 33810 | 17779 | 7880 | 4.3x | 2.3x |
| 512K | 8192 | 32 | 57659 | 26100 | 21326 | 2.7x | 1.2x |

> 数据源：MiniMax Sparse Attention 论文 Table 1，见 <https://arxiv.org/html/2606.13392v2>。

TopK 这一步看起来不像 attention 本体，但它很关键。因为如果 indexer 的选择本身很慢，后面 sparse attention 省下来的时间就会被吃掉。

论文正文描述的是 H800 实验里的 TopK kernel 设计；当前公开仓库 `MiniMax-AI/MSA` 里的 `sparse_topk_select` 是另一个工程化实现，代码注释显示它采用的是 `transpose + indexerTopK histogram/insertion-sort + warp bitonic sort` 的 pipeline。下面第 15.11 节开始会按公开源码来讲。

### 15.2 为什么 Q-outer 不够好

拿到 TopK blocks 后，最自然的 sparse attention 写法是 **Q-outer**：

```text
for each query:
  blocks = TopK(query)
  for each block in blocks:
    load K/V block
    compute attention
```

它很直观，因为 attention 本来就是“每个 query 看哪些 key/value”。

问题在于 GPU 喜欢连续、规则、可复用的数据访问。Q-outer 在 MSA 里会有几个麻烦：

- query A 选了 blocks `[3, 8, 20, ...]`。
- query B 选了 blocks `[0, 8, 41, ...]`。
- query C 选了 blocks `[0, 7, 8, ...]`。

block 8 被很多 query 选中。如果按 query 逐个处理，block 8 的 K/V 可能会被重复加载很多次。K/V 读显存是很贵的，重复读会拖慢整体。

论文用算术强度解释这个问题。算术强度可以粗略理解为：

```text
每搬一次数据，能做多少计算
```

如果一个 kernel 搬了很多数据但计算很少，它就容易被显存带宽卡住。如果搬一次数据能喂给 tensor core 做很多矩阵乘法，它就更容易跑满。

对 Q-outer 来说，MSA sparse attention 的 FLOPs 约为：

$$
\mathrm{FLOPs}
=
4H_qNd_hkB_k
$$

但 K/V 会按 query 选中的 blocks 反复读取，论文估算 Q-outer 的算术强度大约是：

$$
\mathrm{FLOPs}/\mathrm{IO} \approx G
$$

这里 `G = H_q / H_kv`，也就是一个 KV head 对应多少个 query heads。论文主设置是：

```text
H_q = 64
H_kv = 4
G = 16
```

算术强度大约是 16，不算差，但还不够好。

### 15.3 KV-outer：把循环顺序反过来

MSA 论文选择的是 **KV-outer sparse attention**。它把循环反过来：

```text
for each KV block:
  找出哪些 query 选择了这个 KV block
  把这些 query 聚在一起
  load 这个 KV block
  一起算 attention
```

这样做的核心好处是：同一个 K/V block 可以被一批 query 复用。

![KV-outer kernel 把选择同一 KV block 的 query 聚起来](/learning/assets/msa-kv-outer.svg)

论文对 KV-outer 的算术强度估算是：

$$
\mathrm{FLOPs}/\mathrm{IO}
\approx
\frac{2}{3}B_k
$$

代入 `B_k=128`：

$$
\frac{2}{3}B_k \approx 85
$$

这比 Q-outer 的 `G=16` 高很多。每次把一个 128-token 的 KV block 搬上来，尽量让更多 query 使用它，而不是搬上来只服务一个 query。

这也是为什么 MSA 要选 block：block 粒度会损失一点选择精度，但能换来更连续的 K/V 读取和更适合 GPU 的矩阵形状。

### 15.4 反向索引：从 q2k 变成 k2q

TopK 的自然输出是 `q2k`：

```text
query 0 -> block 2, block 7, block 9, ...
query 1 -> block 7, block 8, block 19, ...
query 2 -> block 2, block 8, block 13, ...
```

但 KV-outer kernel 需要的是 `k2q`：

```text
block 2 -> query 0, query 2, ...
block 7 -> query 0, query 1, ...
block 8 -> query 1, query 2, ...
```

这一步就是把 TopK indices 反过来组织。

公开仓库的 CuTe-DSL README 里也能看到这层接口：sparse attention 推荐传入 CSR 风格的 metadata，包括：

- `k2q_row_ptr`
- `k2q_q_indices`
- `schedule`

CSR 是一种压缩存储“每个 block 对应哪些 query”的格式。

例如有 3 个 KV blocks：

```text
block 0 -> [query 1, query 4]
block 1 -> []
block 2 -> [query 0, query 2, query 3]
```

可以存成：

```text
k2q_q_indices = [1, 4, 0, 2, 3]
k2q_row_ptr   = [0, 2, 2, 5]
```

解释一下 `row_ptr`：

- block 0 的 query 在 `q_indices[0:2]`，也就是 `[1, 4]`。
- block 1 的 query 在 `q_indices[2:2]`，为空。
- block 2 的 query 在 `q_indices[2:5]`，也就是 `[0, 2, 3]`。

这就是 KV-outer 能启动的关键数据结构。

### 15.5 attention kernel 里到底在做什么

有了 `k2q` 之后，sparse attention kernel 可以按 `(kv_block, kv_head)` 分 tile 执行。

一个 tile 大致对应：

```text
某一个 KV head
某一个 KV block
选择了这个 block 的一批 query positions
```

每个 tile 内部做的事大概是：

1. 从 `k2q_row_ptr` / `k2q_q_indices` 找到这批 query。
2. 把这些 query 的 Q 向量 gather 出来。
3. 用 TMA 或类似高效搬运方式把 Q/K/V tiles 放进 shared memory。
4. 用 tensor core 做 `QK^T`，得到 attention logits。
5. 对当前 KV block 内部做局部 softmax。
6. 再做 `softmax(QK^T)V`，得到这个 block 对 query 的 partial output。
7. 把 partial output 和对应 LSE 写到全局 buffer。

KV-outer 下，每个 KV tile 通常只对应几个到几十个 query positions。如果一个 query position 只有 `G=16` 个 query heads，那么单独处理时，MMA 的 M 维只有 16，tensor core 吃不饱。

MSA 的做法是 **query concatenation**。

在论文主设置里：

```text
G = 16
目标 MMA M 维 = 128
ceil(128 / G) = 8
```

也就是说，kernel 会把 8 个 query positions 和它们各自的 16 个 query heads 拼起来：

```text
8 query positions x 16 heads = 128 rows
```

这样就能形成更饱满的 `128 x 128` score MMA。

这一步只有 KV-outer 容易做。因为同一个 tile 下的这些 query 都在看同一个 KV block，所以它们共享 K/V operands。Q-outer 下，不同 query 通常选了不同 KV blocks，就不容易这样拼。

### 15.6 热门 block：为什么需要 pre-scheduled tile chunking

KV-outer 的新问题是负载不均。

有些 KV block 会非常热门。最典型的是序列开头的 attention sink block，很多 query 都会选它。如果一个热门 block 由一个 CTA 负责，它就会处理海量 query；另一些冷门 block 几乎没人选，对应 CTA 很快结束。结果就是：

```text
大部分 GPU 线程已经闲了
少数热点 CTA 还在慢慢干活
```

论文的处理方式叫 **pre-scheduled tile chunking**。

做法是：调度阶段先看每个 KV tile 收到了多少 query。如果某个 tile 太热门，就沿着 query 维度切成多个 chunks。论文里提到每个 chunk 最多大约：

$$
\sim 2kB_k
$$

在主设置里：

```text
k = 16
B_k = 128
2kB_k = 4096 queries
```

如果某个 KV block 被 100K 个 query 选择，就不会只给一个 CTA，而是拆成很多 chunks，让多个 CTAs 并行处理。

还有一个设计是 **避免 atomic 写 output**。

因为一个 query 选了 `k` 个 blocks，所以它会产生最多 `k` 个 partial outputs。MSA 让 scheduler 预先给每个 `(query, chunk)` 分配一个 slot：

```text
s in [0, k)
```

然后把 `query index` 和 `slot id` 打包成一个 32-bit handle。attention kernel 写结果时，直接写到预分配位置：

```text
O_buf[slot, query, head, dim]
LSE_buf[slot, query, head]
```

这样就不需要多个 CTA 抢同一个输出地址，也就避免了热点 atomic。

### 15.7 two-phase forward：为什么 partial output 不能直接相加

KV-outer 把一个 query 的 `k` 个 selected blocks 拆给不同 CTAs 计算。每个 CTA 只看到其中一个 KV block，所以它只能得到一个 **局部 softmax** 的结果。

但真正的 attention softmax 应该是在所有 selected tokens 上一起归一化。

举个简单例子。假设 query 选了两个 blocks：

```text
block A logits: [10, 9]
block B logits: [1, 0]
```

如果在 block A 内部单独 softmax，block A 会得到一个归一化分布。如果在 block B 内部单独 softmax，block B 也会得到一个归一化分布。可全局看，block A 的 logits 远大于 block B，最终输出应该主要来自 block A。

所以 partial outputs 必须知道每个 block 的 softmax 分母有多大，才能正确合并。

这就是 LSE 的作用。

对每个 partial，attention kernel 写：

- `O_buf[s, i, h]`：第 `s` 个 selected block 对 query `i`、head `h` 的局部输出。
- `LSE_buf[s, i, h]`：这个 block 内 logits 的 log-sum-exp。

然后 combine kernel 对同一个 query/head 的多个 slots 做稳定合并：

$$
a = \max_s \mathrm{LSE}_s
$$

$$
\mathrm{LSE}[i,h]
=
a + \log \sum_s \exp(\mathrm{LSE}_s - a)
$$

$$
w_s
=
\exp(\mathrm{LSE}_s - \mathrm{LSE}[i,h])
$$

$$
O[i,h]
=
\sum_s w_s O_{buf}[s,i,h]
$$

通俗说，每个 block 先报告：

```text
我这个 block 内部的输出是什么
我这个 block 的 softmax 总权重规模有多大
```

combine kernel 再按全局权重把它们合成真正的 attention 输出。

![MSA two-phase forward 和 LSE 合并](/learning/assets/msa-two-phase-forward.svg)

论文还提到两个 kernel 之间用 Programmatic Dependent Launch 来隐藏 kernel launch latency：第二个 combine kernel 依赖第一个 attention kernel 的结果，但调度上尽量减少“等 kernel 启动”的额外开销。

### 15.8 Sparse KL loss kernel：训练时不要多跑一遍 forward

前面讲过，MSA 用 KL loss 让 Index Branch 学 Main Branch 的注意力分布。KL loss 需要一些 softmax 相关的量，尤其是 LSE。

最朴素的实现可能是：

```text
正常 forward 跑一遍
为了 KL loss 再跑一个专门 forward kernel 算 LSE_main 和 LSE_idx
backward 再用这些量算梯度
```

这样太浪费。

论文的优化是 **LSE fusion**：

- Main Branch 正常 sparse attention forward 时，顺手把 `LSE_main` 写出来。
- Index Branch 算 block scores 时，保存 per-block LSE。
- 对 Top-K blocks 做 reduce，得到 `LSE_idx`。
- KL backward 时直接读这些标量，不再额外跑 KL forward kernel。

这类优化看起来不改变数学公式，但对训练吞吐很重要。长上下文训练时，每多一次大规模 attention-like pass 都会很贵。

Sparse KL backward 还有一个负载均衡问题：不同 tile 的 query 数量差别可能非常大。论文采用 persistent grid，让 CTAs 通过一个 global atomic counter 动态领取工作。也就是说：

```text
CTA 干完一个 tile/sub-tile 后
再去全局计数器领下一个任务
```

这样比静态地“一人分一块”更能适应数据相关的 sparse pattern。

### 15.9 公开仓库当前实现和论文实现的关系

论文里报告的主要端到端加速是在 H800 上测的。论文同时公开了 inference kernel 仓库：<https://github.com/MiniMax-AI/MSA>。

我查了当前公开 README。它的包名是 `fmha_sm100`，当前公开实现主要面向 NVIDIA **SM100**，并把实现分成几层：

| 公开仓库模块 | 作用 |
|---|---|
| `csrc JIT` | dense FMHA、paged KV、`sparse_topk_select` indexer |
| `CuTe-DSL` | block-sparse attention forward、paged FP8 decode、BF16/FP8/NVFP4/FP4 路径 |
| `Bridge` | 把 `fmha_sm100` API 接到 sparse backend |

CuTe-DSL README 里还写了当前 sparse attention 的一些约束：

- head dimension 目前文档化支持 `D=128`。
- sparse attention forward 支持 `qhead_per_kv` 为 `{1, 2, 4, 8, 16}`。
- CSR builder 支持 `topK` 为 `{4, 8, 16, 32}`。
- `blk_kv=128` 是公开支持路径里的关键块大小。
- 推荐流程是先构造 `q2k_indices`，再通过 `build_k2q_csr(..., return_schedule=True)` 同时构造 CSR metadata 和 schedule，最后调用 `sparse_atten_func`。

这和论文主设置高度一致：`D=128`、`B_k=128`、`k=16`、`G=16` 都是核心配置。但要注意，公开仓库 README 描述的是当前开源代码的支持边界，论文实验的 H800 kernel 和仓库当前 SM100 代码不应该被混成完全同一个二进制实现。

### 15.10 把 kernel pipeline 串起来

现在可以从实现视角把 MSA 一层串起来：

```text
1. Index Branch
   Q_idx, K_idx -> block scores

2. TopK kernel
   raw block scores -> q2k_indices

3. CSR / scheduler
   q2k_indices -> k2q_row_ptr, k2q_q_indices, schedule

4. KV-outer sparse attention kernel
   for each scheduled (kv_block, kv_head, query_chunk):
     gather Q
     load K/V block
     concatenate query positions
     run QK and PV MMA
     write O_buf and LSE_buf

5. Combine kernel
   read O_buf and LSE_buf
   merge partial outputs with global softmax weights
   write final O and final LSE

6. Sparse KL backward
   reuse fused LSE_main / LSE_idx
   update indexer without extra KL forward
```

从这个 pipeline 看，MSA 的速度来自几层工程设计叠加：

- TopK 不做 softmax，直接按 raw score 选。
- KV-outer 提高 K/V 复用。
- query concatenation 让 tensor core 的 MMA 形状更饱满。
- pre-scheduled chunking 处理 attention sink 等热点 blocks。
- 预分配 `O_buf` slot，避免 attention partial 写回时使用 atomic。
- two-phase forward 用 LSE 正确合并 partial softmax。
- KL loss 的 LSE 信息融合进已有 forward，减少训练时额外 kernel。

这也是 MSA 论文比较强调 kernel co-design 的原因：没有这些实现细节，算法上的 sparse FLOPs 很容易停留在纸面上。

### 15.11 公开源码地图：先知道每个文件管什么

下面开始按公开仓库的源代码实现讲。源码仓库是：

<https://github.com/MiniMax-AI/MSA>

公开源码里，MSA kernel 主要分成两条路径：

| 路径 | 代码位置 | 主要作用 |
|---|---|---|
| `csrc JIT` | `python/fmha_sm100/csrc/` | dense FMHA、paged KV、`sparse_topk_select` |
| `CuTe-DSL` | `python/fmha_sm100/cute/` | SM100 block-sparse attention forward、combine、paged decode、FP4/FP8/NVFP4 路径 |

和 MSA sparse attention 关系最直接的是这些文件：

| 文件 | 读这个文件看什么 |
|---|---|
| [`python/fmha_sm100/api.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/api.py) | Python 侧公开 API，包括 `sparse_topk_select` |
| [`python/fmha_sm100/csrc/sparse_topk_select.cu`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/csrc/sparse_topk_select.cu) | TopK CUDA wrapper，检查参数后调用真正的 TopK dispatcher |
| [`python/fmha_sm100/csrc/include/sparse_topk_select.cuh`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/csrc/include/sparse_topk_select.cuh) | TopK CUDA kernel 主体 |
| [`python/fmha_sm100/cute/sparse_index_utils.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/sparse_index_utils.py) | 把 `q2k_indices` 转成 KV-outer 需要的 `k2q` CSR |
| [`python/fmha_sm100/cute/src/sm100/prepare_scheduler.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/src/sm100/prepare_scheduler.py) | 把不均匀的 CSR row 切成 worklist，并预分配 split slot |
| [`python/fmha_sm100/cute/interface.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/interface.py) | `sparse_atten_func`，分配 partial buffers，调用 forward kernel 和 combine kernel |
| [`python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py) | SM100 sparse attention forward kernel |
| [`python/fmha_sm100/cute/src/sm100/fwd/combine.py`](https://github.com/MiniMax-AI/MSA/blob/main/python/fmha_sm100/cute/src/sm100/fwd/combine.py) | 合并 `O_partial` / `LSE_partial` 的 combine kernel |

如果只想看 prefill sparse attention 的核心调用链，可以按这个顺序读：

```text
api.py / interface.py
  -> sparse_index_utils.py
  -> prepare_scheduler.py
  -> fwd/atten_fwd.py
  -> fwd/combine.py
```

如果只想看 TopK：

```text
api.py::sparse_topk_select
  -> csrc/sparse_topk_select.cu
  -> csrc/include/sparse_topk_select.cuh
```

### 15.12 TopK 源码实现：公开版不是论文里的 min-heap 版本

先看 Python 侧入口。公开 API 里的 `sparse_topk_select` 接受的 `max_score` 形状是：

```text
(num_qo_heads, max_k_tiles, total_qo_len)
```

输出形状是：

```text
(total_qo_len, num_qo_heads, topk)
```

也就是把每个 query token、每个 head 的 Top-K KV tile indices 选出来。

源码逻辑可以简化成：

```python
def sparse_topk_select(max_score, topk, num_valid_pages=None, output=None, ...):
    assert max_score.dtype == torch.float32
    assert max_score.dim() == 3
    assert topk == 16

    num_qo_heads, max_k_tiles, total_qo_len = max_score.shape
    assert max_k_tiles < 12288

    workspace_buffer = alloc_workspace(num_qo_heads * max_k_tiles * total_qo_len)
    output_indices = torch.empty(total_qo_len, num_qo_heads, topk, dtype=torch.int32)

    module = get_sparse_topk_module()
    module.sparse_topk_select(
        max_score,
        output_indices,
        workspace_buffer,
        topk,
        num_valid_pages,
        force_begin_blocks,
        force_end_blocks,
        current_cuda_stream,
    )
    return output_indices
```

这段对应 `python/fmha_sm100/api.py` 里的 `sparse_topk_select`：

- 当前公开 API 强制 `topk == 16`。
- `max_k_tiles < 12288`，因为当前公开 TopK 只启用了 insertion-sort 相关路径，还没有 radix-sort 大 K 路径。
- `num_valid_pages` 用来把 padding tile 过滤掉。超过真实 KV page 范围的 index 会被 kernel 写成 `-1` 并放到 tail。
- `force_begin_blocks` 和 `force_end_blocks` 可以强制包含开头 blocks 或结尾 blocks，分别对应 sink 和 local-window 类需求。

进入 C++/CUDA wrapper 后，`sparse_topk_select.cu` 做的事情很薄：

```cpp
void sparse_topk_select(TensorView max_score,
                        TensorView output_indices,
                        TensorView workspace_buffer,
                        int64_t topk,
                        int64_t num_valid_pages,
                        int64_t force_begin_blocks,
                        int64_t force_end_blocks,
                        int64_t stream_ptr) {
  CHECK_INPUT(max_score);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(workspace_buffer);
  CHECK_DIM(3, max_score);
  CHECK_DIM(3, output_indices);

  const int64_t num_qo_heads = max_score.size(0);
  const int64_t max_k_tiles = max_score.size(1);
  const int64_t total_qo_len = max_score.size(2);

  sparse_topk::SparseTopKSelect(
      static_cast<const float*>(max_score.data_ptr()),
      static_cast<int32_t*>(output_indices.data_ptr()),
      static_cast<int32_t*>(workspace_buffer.data_ptr()),
      total_qo_len,
      num_qo_heads,
      max_k_tiles,
      num_valid_pages,
      force_begin_blocks,
      force_end_blocks,
      stream);
}
```

真正的实现都在 `sparse_topk_select.cuh`。

公开源码注释里直接给了 pipeline：

```text
input:  (Hq, K, qo) row-contig fp32
  -> SparseTopKTransposeKernel
workspace: (Hq, qo, K) row-contig fp32
  -> IndexerTopKWithSortKernel<16>
output: (qo, num_qo_heads, topk) int32
```

为什么要先 transpose？

原始 `max_score` 的形状是：

```text
head, k_tile, query
```

但 TopK 是“对一个 query 的所有 k_tile 做选择”。如果 query 不是连续维度，逐行 TopK 会很难做高效连续读取。所以源码先转置成：

```text
head, query, k_tile
```

这样每一行的 `k_tile` 是连续的，后面的 TopK kernel 每个 CTA 处理一行。

转置 kernel 有两个版本：

- `SparseTopKTransposeKernel`：通用 32x32 shared memory tile 转置。
- `SparseTopKTransposeXorF4Kernel`：快路径，用 XOR swizzle 和 `float4` 读写减少 bank conflict 和提升带宽。

快路径触发条件很工程化：

```cpp
const bool xor_path_ok =
    (total_qo_len % 32 == 0) &&
    (max_k_tiles % 32 == 0) &&
    (total_qo_len % 4 == 0) &&
    (max_k_tiles % 4 == 0);
```

如果满足条件，就走 XOR swizzle + `float4` 的快路径；否则走 padded shared-memory tile 的 fallback。

第二个 kernel 是 `IndexerTopKWithSortKernel<16>`。公开源码的思路改自 TensorRT-LLM 的 `indexerTopK`：

```text
1. 每个 CTA 处理一行，也就是一个 (query, head) 的所有 KV tile scores。
2. 用 histogram step 找到 Top-K 分数所在的阈值 bin。
3. 对边界 bin 里的候选做 insertion sort，补齐剩余 Top-K。
4. 用 warp-only bitonic sort 把 Top-K indices 按 index 升序排好。
5. 写到 output_indices[query, head, k]。
```

升序排序是为了后续 sparse attention 读 KV blocks 时更规整。源码注释还说，早期版本用 `cub::BlockRadixSort<uint32_t, 512, 1>` 排序，但真实 TopK 只有很少元素，512 个线程里大量位置是 sentinel，所以浪费。当前版本换成 warp-only bitonic sort：

```cpp
if constexpr (MAX_TOPK <= 32) {
  uint32_t key = ~0u;
  if (lane < topK) {
    const int32_t idx = smemOutput[lane];
    const bool valid = (idx >= 0) && (idx < num_valid_pages);
    key = valid ? uint32_t(idx) : ~0u;
  }
  WarpBitonicSortAsc32(key, lane);
  if (lane < topK) {
    row_out[lane] = (key == ~0u) ? -1 : int32_t(key);
  }
}
```

这里的 `~0u` 是一个很大的 unsigned sentinel。它会在升序排序里跑到末尾，最后再被写成 `-1`。

所以如果你按源码理解当前公开 TopK，可以记成：

> 公开版 TopK = 先转成 query-contiguous 行，再用 histogram/insertion-sort 找 TopK，最后用 warp bitonic sort 把 indices 升序写出。

### 15.13 q2k -> k2q：源码里怎样把 query 视角反过来

TopK 输出天然是 `q2k_indices`：

```text
q2k_indices[head_kv, q, slot] = kv_block_id
```

但 KV-outer forward kernel 需要按 KV block 启动，所以要转成：

```text
k2q_row_ptr[head_kv, row]
k2q_q_indices[head_kv, edge]
```

公开源码里这个入口是 `build_k2q_csr`，位置在 `python/fmha_sm100/cute/sparse_index_utils.py`：

```python
def build_k2q_csr(
    q2k_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_block_size,
    *,
    total_k,
    max_seqlen_k=None,
    max_seqlen_q=None,
    total_rows=None,
    qhead_per_kv=1,
    return_schedule=False,
):
    return _K2Q_CSR_BUILDER(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        total_k=int(total_k),
        blk_kv=int(kv_block_size),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        return_schedule=return_schedule,
    )
```

这里 `cu_seqlens_q` / `cu_seqlens_k` 是变长 batch 常用的 prefix sum。例如 batch 里有两条样本，Q 长度分别是 5 和 7：

```text
cu_seqlens_q = [0, 5, 12]
```

这样 kernel 就知道第 0 条样本在 `[0,5)`，第 1 条样本在 `[5,12)`。

CSR 的含义前面已经讲过。源码注释里写得很直接：

```text
q2k_indices:   [head_kv, total_q, topK]
k2q_row_ptr:   [head_kv, total_rows + 1]
k2q_q_indices: [head_kv, total_q * topK]
```

`return_schedule=True` 还会顺手构造 forward 需要的 schedule。也就是说，实际生产路径通常是：

```python
k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
    q2k_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    blk_kv=128,
    total_k=total_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    qhead_per_kv=16,
    return_schedule=True,
)
```

### 15.14 scheduler 源码：把热点 KV block 切成 worklist

源码里的 schedule 数据结构在 `prepare_scheduler.py`：

```python
@dataclass
class SparseAttentionSchedule:
    enabled: bool
    scheduler_metadata: Optional[torch.Tensor]
    work_count: Optional[torch.Tensor]
    qsplit_indices: Optional[torch.Tensor] = None
    split_counts: Optional[torch.Tensor] = None
    target_q_per_cta: int = 0
```

几个字段的含义是：

| 字段 | 含义 |
|---|---|
| `scheduler_metadata` | 每个 work item 的元数据，形状是 `[capacity, 6]` |
| `work_count` | 实际生成了多少个 work item |
| `qsplit_indices` | 把 `q_idx` 和 split slot 打包后的 query index |
| `split_counts` | 每个 query/head 实际有多少个 split partial |
| `target_q_per_cta` | 每个 CTA 目标处理多少个 query |

`scheduler_metadata` 的 6 列在 `_emit_work` 里写入：

```python
mSchedulerMetadata[work_idx, 0] = head_kv_idx
mSchedulerMetadata[work_idx, 1] = row_linear
mSchedulerMetadata[work_idx, 2] = q_begin
mSchedulerMetadata[work_idx, 3] = q_count
mSchedulerMetadata[work_idx, 4] = batch_idx
mSchedulerMetadata[work_idx, 5] = kv_block_idx
```

这 6 个数就描述了一个 CTA 该干什么：

```text
哪个 KV head
哪个 CSR row
从这个 row 的第几个 query 开始
处理多少个 query
属于 batch 里的哪条样本
对应哪个 KV block
```

`target_q_per_cta` 的计算也对应论文里说的 `~2kB_k` 上限。源码里有这一行：

```python
sink_balance_cap = max(q_tokens_per_group, int(topk) * int(blk_kv) * 2)
target = min(max(occupancy_target, q_tokens_per_group), sink_balance_cap)
```

代入主设置：

```text
topk = 16
blk_kv = 128
topk * blk_kv * 2 = 4096
```

也就是热门 KV block 的一个 chunk 最多大约 4096 个 query，这和论文文字一致。

生成 worklist 的 kernel 是 `SparseAttentionPrepareFlatScheduleSm100.kernel`。简化后是：

```python
row_count = k2q_row_ptr[head_kv, row + 1] - k2q_row_ptr[head_kv, row]
num_chunks = ceil(row_count / target_q_per_cta)

for chunk_idx in chunks_of_this_row:
    work_idx = atomic_add(work_count, 1)
    q_begin = chunk_idx * target_q_per_cta
    q_count = min(target_q_per_cta, row_count - q_begin)
    emit_work(work_idx, head_kv, row, q_begin, q_count, batch, kv_block)
```

这里的 atomic 只发生在 scheduler 阶段，用来往 worklist 里抢一个位置。attention forward 正式写 `O_partial` 时不再需要 atomic。

接下来还有一步 `SparseAttentionPrepareFwdSplitAtomicSm100`，它负责给每个 query/head 分配 split slot：

```python
split_slot = atomic_add(split_counts[q_abs, head_kv_idx])
if split_slot < topk:
    k2q_qsplit_indices[head_kv, edge] = q_idx | ((split_slot & 0xFF) << 24)
```

这行把两个信息塞进一个 int32：

```text
低 24 bits：batch-local q_idx
高 8 bits ：split_slot
```

forward kernel 之后会解出来：

```python
q_idx = qsplit & 0x00FF_FFFF
split_idx = (qsplit >> 24) & 0xFF
```

这就是源码里“避免 attention partial 写 atomic”的核心：slot 已经提前分好了，forward kernel 只需要写：

```text
O_partial[split_idx, q_abs, head, dim]
LSE_partial[split_idx, q_abs, head]
```

### 15.15 sparse_atten_func：公开 API 怎么串起两个 kernel

公开 sparse attention 的入口是 `python/fmha_sm100/cute/interface.py` 里的 `sparse_atten_func`。

它的参数已经很像我们前面讲的数学结构：

```python
def sparse_atten_func(
    q, k, v,
    k2q_row_ptr,
    k2q_q_indices,
    topK,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    blk_kv=128,
    causal=False,
    schedule=None,
    ...
):
    ...
```

这个函数做完参数检查后，会进入 `_sparse_atten_csr_varlen_forward`。在那里，源码先分配 partial buffers：

```python
O_partial = torch.empty(
    topK, total_q, head_q, dim,
    dtype=partial_dtype,
    device=q.device,
)

LSE_partial = torch.empty(
    topK, total_q, head_q,
    dtype=torch.float32,
    device=q.device,
)

O_out = torch.empty(total_q, head_q, dim, dtype=torch.bfloat16, device=q.device)
LSE_out = torch.empty(total_q, head_q, dtype=torch.float32, device=q.device)
```

如果调用方没有传入 schedule，它会临时分配：

```python
k2q_qsplit_indices = torch.empty_like(k2q_q_indices)
split_counts = torch.zeros((total_q, head_kv), dtype=torch.int32, device=q.device)
```

然后调用 sparse forward kernel：

```python
schedule = _call_sparse_forward_sm100_csr_varlen(
    q, k, v,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    split_counts,
    cu_seqlens_q,
    cu_seqlens_k,
    O_partial,
    LSE_partial,
    ...
    schedule=schedule,
)
```

最后调用 combine：

```python
combine(
    O_partial,
    LSE_partial,
    O_out,
    LSE_out,
    cu_seqlens=cu_seqlens_q,
    split_counts=split_counts,
    use_pdl=True,
)
```

这就是源码层面 two-phase forward 的完整形态：

```text
forward kernel: 写 O_partial / LSE_partial
combine kernel: 读 O_partial / LSE_partial，写 O_out / LSE_out
```

### 15.16 forward kernel 源码：一个 CTA 怎么对应一个 KV-outer work item

核心 kernel class 是 `SparseAttentionForwardSm100`，位置在：

```text
python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py
```

构造函数里有几个固定设计：

```python
self.m_block_size = 128
self.n_block_size = 128
self.q_tokens_per_group = m_block_size // qheadperkv
self.mma_tiler_qk = (m_block_size, n_block_size, head_dim)
self.mma_tiler_pv = (m_block_size, head_dim, n_block_size)
self.total_warps = 16
self.threads_per_cta = 32 * self.total_warps  # 512
```

如果 `qheadperkv=16`：

```text
q_tokens_per_group = 128 / 16 = 8
```

这就是前面说的 query concatenation：8 个 query positions，每个 position 16 个 Q heads，拼成 128 行。

源码还把 16 个 warp 分成不同角色：

| warp 角色 | 做什么 |
|---|---|
| softmax0 / softmax1 | 两组 softmax worker group |
| store / q_load | 负责 Q load 和 epilogue/store |
| mma_warp | 发起 QK/PV MMA |
| kv_load_warps | 加载 K/V |
| empty warp | 占位和调度需要 |

kernel 一开始从 `scheduler_metadata` 里读这个 CTA 的任务：

```python
work_idx = block_idx.x
if work_idx < work_count[0]:
    head_kv_idx = scheduler_metadata[work_idx, 0]
    row_linear = scheduler_metadata[work_idx, 1]
    work_q_begin = scheduler_metadata[work_idx, 2]
    work_q_count = scheduler_metadata[work_idx, 3]
    batch_idx = scheduler_metadata[work_idx, 4]
    kv_block_idx = scheduler_metadata[work_idx, 5]
```

这说明公开实现确实是 KV-outer：grid 的每个 CTA 对应“一个 KV block/head 的一个 query chunk”。

然后 kernel 做几类工作：

1. **读 CSR row 范围**：根据 `row_linear` 找到这个 KV block 对应的 query 列表。
2. **读 qsplit**：从 `k2q_qsplit_indices` 得到 `q_idx` 和 `split_idx`。
3. **加载 Q**：如果 `qheadperkv` 是 4/2/1，走 `gather4`；否则走 TMA load。
4. **加载 K/V**：按 `kv_block_idx` 和 `head_kv_idx` 用 TMA 把 K/V tile 搬进 shared memory。
5. **QK MMA**：算当前 query chunk 对当前 KV block 的 logits。
6. **softmax**：只在当前 KV block 内做局部 softmax，并缓存 row max / row sum。
7. **PV MMA**：算局部输出。
8. **epilogue**：根据 `split_idx` 写 `O_partial`，并写 `LSE_partial`。

源码里 epilogue 写 LSE 的关键变量是：

```python
mLSE_partial[split_lse, q_abs_lse, h_abs] = lse_cur
```

这里：

- `split_lse` 是提前分好的 split slot。
- `q_abs_lse` 是全局 query index。
- `h_abs` 是 query head。
- `lse_cur` 是这个 KV block 内部 logits 的 log-sum-exp。

这是 two-phase forward 的第一阶段输出。

### 15.17 combine kernel 源码：LSE 怎么变成最终 softmax 权重

combine kernel 在：

```text
python/fmha_sm100/cute/src/sm100/fwd/combine.py
```

入口函数叫 `combine`，里面会 JIT 编译并调用 `SparseAttentionForwardCombine`。

它的输入就是上一阶段的 partial buffers：

```python
combine(
    o_partial,
    lse_partial,
    o_out,
    lse_out,
    split_counts=split_counts,
    use_pdl=True,
)
```

`use_pdl=True` 对应 Programmatic Dependent Launch。源码里 combine kernel 开始时会等 producer kernel：

```python
if use_pdl:
    cute.arch.griddepcontrol_wait()
```

然后它先读 `LSE_partial`。对无效 split，它填 `-inf`：

```python
if split_id < row_count:
    load LSE_partial[split_id]
else:
    LSE = -inf
```

接着源码里做的就是我们前面写的数学公式。简化后是：

```python
lse_max = max(LSE_partial[s] for s in valid_splits)
lse_sum = sum(exp(LSE_partial[s] - lse_max) for s in valid_splits)
final_lse = log(lse_sum) + lse_max

scale[s] = exp(LSE_partial[s] - final_lse)
```

源码中实际为了快，用的是 `exp2` 和 `LOG2_E`：

```python
scale = exp2(LSE_s * LOG2_E - lse_max * LOG2_E)
```

得到 `scale` 后，再累计 `O_partial`：

```python
O = 0
for s in valid_splits:
    O += scale[s] * O_partial[s]
```

最后写：

```text
O_out[q, head, dim] = O
LSE_out[q, head] = final_lse
```

所以源码里的 combine kernel 是一个 softmax-aware reduce。它必须用 `LSE_partial` 先恢复每个 partial 在全局 softmax 里的权重。

### 15.18 从源码视角重新看整个 kernel pipeline

把公开源码串起来，prefill sparse attention 的执行路径可以写成：

```text
1. 生成 block scores
   dense/proxy FMHA 或 FP4 indexer 产生 max_score

2. sparse_topk_select
   api.py
     -> csrc/sparse_topk_select.cu
     -> csrc/include/sparse_topk_select.cuh
   输出 q2k_indices[total_q, head, topk]

3. build_k2q_csr
   sparse_index_utils.py
     -> SparseK2qCsrBuilderSm100
   输出 k2q_row_ptr, k2q_q_indices

4. prepare_sparse_fwd_schedule_and_split
   prepare_scheduler.py
   输出 scheduler_metadata, work_count, qsplit_indices, split_counts

5. SparseAttentionForwardSm100
   fwd/atten_fwd.py
   每个 CTA 读取一个 scheduler_metadata work item
   写 O_partial 和 LSE_partial

6. combine
   fwd/combine.py
   用 LSE_partial 计算全局 softmax 权重
   合并 O_partial，写 O_out
```

这条路径和论文里的概念是一一对应的：

| 论文概念 | 源码中的落点 |
|---|---|
| Top-K block selection | `sparse_topk_select` |
| KV-outer | `k2q_row_ptr` / `k2q_q_indices` / `scheduler_metadata` |
| Pre-scheduled tile chunking | `prepare_sparse_flat_schedule` |
| 预分配 split slot | `q_idx | (split_slot << 24)` |
| Query concatenation | `q_tokens_per_group = 128 // qheadperkv` |
| Two-phase forward | `O_partial/LSE_partial -> combine` |
| LSE merge | `combine.py` 中的 `final_lse` 和 `scale[s]` |

如果你要自己对着源码看，建议先不要从 `atten_fwd.py` 第一行开始硬读。更好的顺序是：

1. 先看 `interface.py` 里 `sparse_atten_func` 如何分配 `O_partial` / `LSE_partial`。
2. 再看 `prepare_scheduler.py` 里 `scheduler_metadata` 的 6 列含义。
3. 然后回到 `atten_fwd.py` 看 kernel 如何读取 `scheduler_metadata`。
4. 最后看 `combine.py` 如何用 `LSE_partial` 合并 partial outputs。

这样读源码会容易很多，因为你先知道每个 tensor 是为什么存在的，再去看 CuTe-DSL 里复杂的 TMA、MMA、pipeline、barrier。

### 15.19 如果你要自己跟源码，最容易踩的几个实现点

前面讲的是源码的主干逻辑。真正开始读代码或改代码时，更容易出错的地方是 **tensor layout、API 边界和中间 buffer 的生产消费关系**。

先看一张源码追踪表：

| 对象 | 在哪里产生 | 形状或布局 | 被谁消费 | 作用 |
|---|---|---|---|---|
| `max_score` | Index Branch / proxy FMHA 路径 | `api.py` 注释里是 `(num_qo_heads, max_k_tiles, total_qo_len)` | `sparse_topk_select` | 每个 query/head 对每个 KV tile 的最大分数 |
| `output_indices` | `sparse_topk_select` | `(total_qo_len, num_qo_heads, topk)` | 后续 sparse index 构造逻辑 | Top-K KV tile id，公开 TopK kernel 当前要求 `topk == 16` |
| `q2k_indices` | TopK 结果整理后 | `build_k2q_csr` 注释里是 `[head_kv, total_q, topK]` | `build_k2q_csr` | query 视角：每个 query 选了哪些 KV blocks |
| `k2q_row_ptr` | `build_k2q_csr` | `[head_kv, total_rows + 1]` | scheduler / forward kernel | CSR row pointer，描述每个 KV block 对应的 query 列表范围 |
| `k2q_q_indices` | `build_k2q_csr` | `[head_kv, >= total_q * topK]` | scheduler / forward kernel | CSR values，存具体 query index |
| `scheduler_metadata` | `prepare_scheduler.py` | `[work_capacity, 6]` | `SparseAttentionForwardSm100` | 每个 CTA 的 work item 元数据 |
| `qsplit_indices` | `prepare_scheduler.py` | 和 `k2q_q_indices` 类似 | forward kernel | 把 `q_idx` 和 `split_slot` 打包，避免 forward 写回 atomic |
| `split_counts` | `prepare_scheduler.py` | `[total_q, head_kv]` | combine kernel | 每个 query/head 有多少个 partial splits |
| `O_partial` | `interface.py` | `[topK, total_q, head_q, dim]` | combine kernel | 每个 selected KV block 的局部 attention 输出 |
| `LSE_partial` | `interface.py` | `[topK, total_q, head_q]` | combine kernel | 每个局部 softmax 的 log-sum-exp |
| `O_out` / `LSE_out` | combine kernel | `[total_q, head_q, dim]` / `[total_q, head_q]` | attention 后续层 | 最终 attention 输出和最终 LSE |

**不同公开函数注释里的 layout 不完全一样**。`sparse_topk_select` 的输出是 `(total_qo_len, num_qo_heads, topk)`，而 `build_k2q_csr` 要求的输入是 `[head_kv, total_q, topK]`。所以工程里通常会在 TopK 之后做一次 layout 整理，至少要确保传给 `build_k2q_csr` 的张量满足它自己的注释要求。读源码时不要只看变量名猜 shape，要以每个 API 的 docstring 和 assert 为准。

把调用关系写成最小伪代码，大概是这样：

```python
# 1. Index Branch 或 proxy FMHA 产生 block score
max_score = ...  # (head_or_group, max_k_tiles, total_q)

# 2. TopK CUDA kernel 选 KV blocks
topk_indices = sparse_topk_select(
    max_score,
    topk=16,
    num_valid_pages=ceil(kv_len / blk_kv),
    force_begin_blocks=sink_blocks,
    force_end_blocks=local_blocks,
)

# 3. 整理成 build_k2q_csr 需要的 q2k layout
q2k_indices = arrange_to_head_first_layout(topk_indices)

# 4. 反向索引：query -> KV blocks 变成 KV block -> queries
k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
    q2k_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_block_size=128,
    total_k=k.shape[0],
    qhead_per_kv=Hq // Hkv,
    return_schedule=True,
)

# 5. 真正的 block-sparse attention forward
out = sparse_atten_func(
    q, k, v,
    k2q_row_ptr,
    k2q_q_indices,
    topK=16,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    blk_kv=128,
    schedule=schedule,
)
```

这段伪代码把公开 API 的真实职责压缩到最小路径。真正工程里还会有 paged KV、FP8/FP4 cache、workspace 分配、causal mask、temperature LSE 等分支。

再看几个源码实现边界：

1. **TopK 选择 kernel 和 sparse attention kernel 的 `topK` 支持范围不同**。`sparse_topk_select` 当前公开实现强制 `topk == 16`；但 `sparse_atten_func` 的 docstring 写的是 forward kernel 支持 `4, 8, 16, 32`。所以不要看到 forward 支持 32，就以为公开 TopK kernel 也能直接选 32。
2. **`num_valid_pages` 最好显式传**。TopK 输入里常有为了对齐而补出来的 padding tiles。如果不告诉 kernel 真实有效 page 数，后续 sparse attention 可能读到无效 page-table 位置。公开 API 注释里也把这一点写成强烈建议。
3. **`force_begin_blocks` 和 `force_end_blocks` 是工程上的保底机制**。前者常用来保留 attention sink，后者常用来保留当前位置附近的 local window。它们不是 MSA 公式的核心，但对长上下文质量和稳定性很实用。
4. **`qsplit_indices` 是避免 forward 输出 atomic 的关键**。scheduler 阶段可以用 atomic 给每个 query/head 分 split slot；forward kernel 写 `O_partial` 时已经知道自己的 slot，所以不用多个 CTA 抢同一个输出地址。
5. **combine 不能简单相加 `O_partial`**。每个 partial 只在自己的 KV block 内做了 softmax，它的归一化分母不同。必须先用 `LSE_partial` 还原全局 softmax 权重，再加权合并。
6. **`head_dim` 在公开 SM100 sparse forward 里固定为 128**。这和论文主模型设置一致，但如果你拿一个 head dimension 不是 128 的模型直接套这个 kernel，就不是改几个 Python 参数的问题。
7. **`Hq / Hkv` 只能是 `{1, 2, 4, 8, 16}`**。这是 forward kernel 的编译期支持边界，也对应 GQA group 的实现方式。

如果要调试，可以按下面顺序打断点或打印 shape：

```text
sparse_topk_select:
  check max_score dtype/shape/topk/num_valid_pages

build_k2q_csr:
  check q2k_indices layout, trailing -1, cu_seqlens_q/k

prepare_scheduler.py:
  check scheduler_metadata[:, 0..5]
  check split_counts.max() <= topK

atten_fwd.py:
  check work_idx -> head_kv, row_linear, q_begin, q_count, kv_block_idx
  check qsplit 解包出来的 q_idx / split_slot

combine.py:
  check LSE_partial 是否有 -inf invalid splits
  check final_lse 和 O_out 是否正常
```

这样读，kernel 源码就变成一条很清楚的数据流：**score -> TopK -> q2k -> k2q CSR -> schedule -> partial attention -> LSE merge**。

## 16. MSA 学到了什么 sparse pattern

论文附录可视化了 Index Branch 的选择概率。它们观察到几个模式：

1. **Local diagonal**：模型经常选当前位置附近的 blocks。
2. **Sink column**：模型经常选序列开头的 block，也就是 attention sink。
3. **Long-range stripes**：不同 GQA group 会选不同的远距离条带。

可以把它理解为：

- 附近上下文通常重要，所以要看。
- 开头 token 可能承载格式、系统信息、全局锚点，所以常被看。
- 远距离信息不是固定窗口能覆盖的，不同 group 会学会找不同类型的远程线索。

这也是 MSA 相比普通 sliding window 的价值：它按内容相关性动态选 blocks。

![MSA 学到的稀疏模式示意](/learning/assets/msa-patterns.svg)

论文附录里的真实可视化如下。不同 GQA group 的远距离条带不一样，本地对角线和开头 sink column 则比较稳定：

![MiniMax Sparse Attention 原论文稀疏选择可视化](/learning/assets/msa-paper-patterns-layer1.png)

> 图源：MiniMax Sparse Attention 论文 Figure 5(a)，原始图片来自 <https://arxiv.org/html/2606.13392v2>。

论文附录还比较了 MSA 和 FLOP-matched sliding-window baseline。在相同稀疏预算下，固定窗口 baseline 在 agent 相关任务上的 perplexity 更高。这说明至少在这些任务里，内容相关的动态选择比固定位置规则更适合。

## 17. 实验结果怎么读

论文主要做了两条路线：

| 名称 | 含义 |
|---|---|
| Full | 标准 full-attention GQA baseline |
| MSA-PT | 从头训练 MSA 模型 |
| MSA-CPT | 从 full-attention checkpoint 继续训练，转换成 MSA |

模型设置：

- 41 层 MoE backbone。
- 约 109B 总参数。
- 每 token 激活约 6B 参数。
- 200K 词表。
- hidden size 3072。
- 64 query heads。
- 4 KV heads。
- head dimension 128。
- `B_k = 128`。
- `k = 16`。

训练预算：

- 总预算 3T tokens。
- MSA-PT 从头训练，40B tokens warmup 后 sparse training。
- MSA-CPT 从 2.6T tokens full-attention checkpoint 转换，继续训练 400B tokens，其中 40B tokens warmup。

论文报告的主要结论是：

- MSA 的 LM loss 和 gradient norm 与 full attention 非常接近，说明大规模训练稳定。
- 在通用推理、数学、代码、多模态、长上下文等 benchmark 上，MSA-PT 和 MSA-CPT 总体接近 Full baseline。
- MSA-PT 在一些数学、图像、视频和长上下文检索任务上表现更强，可能是因为从头训练时表示可以适应稀疏模式。
- MSA-CPT 更保守，适合已有 full-attention checkpoint 后做转换。
- 长上下文扩展后，MSA-CPT 在 HELMET 和 RULER 上仍接近 Full baseline。

效率方面：

- 1M context 下 theoretical attention FLOPs reduction 是 **28.4x**。
- H800 上 measured prefill speedup 是 **14.2x**。
- H800 上 measured decode speedup 是 **7.6x**。

论文原图里的效率对比如下：

![MiniMax Sparse Attention 原论文效率图](/learning/assets/msa-paper-efficiency.png)

> 图源：MiniMax Sparse Attention 论文 Figure 4，原始图片来自 <https://arxiv.org/html/2606.13392v2>。

这些数字是在论文的模型结构、上下文长度、GPU、kernel 实现和 benchmark 设置下得到的，不应该直接外推到所有模型和硬件。但方向很清楚：MSA 的收益主要来自超长上下文，越长越有意义。

## 18. 和其他长上下文方法对比

为了避免概念混淆，我把相关方法分成几类。这里看的是它们分别在解决哪个问题。

| 方法 | 做法 | 优点 | 代价 |
|---|---|---|---|
| Full attention | 每个 token 看全部历史 | 表达力最完整 | 长上下文太贵 |
| GQA | 减少 KV heads，让多个 query heads 共享 KV | KV cache 和读写更省 | 仍然看完整上下文 |
| Longformer / BigBird | 固定稀疏模式，比如局部窗口、全局 token、随机连接 | 结构简单，复杂度可降到线性 | 稀疏模式大多和当前内容无关 |
| StreamingLLM | 保留 attention sink 和最近窗口 | 适合流式生成，不需要微调 | 更像推理策略，不是内容动态检索 |
| H2O / SnapKV | 根据注意力统计压缩或保留 KV cache | 不需要重新训练模型 | 主要是 inference-time KV 管理，训练仍然是 full attention |
| Quest | 根据当前 query 选择重要 KV pages | query-aware，适合长上下文推理 | 依赖推理时页面级估计，不是原生训练的 attention 结构 |
| MInference / FlexPrefill | 在 prefill 阶段动态构造稀疏 attention pattern | 可直接加速已有长上下文模型 | 主要是推理加速，不改变模型预训练机制 |
| InfLLM | 把远距离上下文组织成可检索 memory units | 不微调也能外推很长上下文 | 更像外部记忆/检索机制，不是每层原生 sparse attention |
| NSA | 原生可训练 sparse attention，结合压缩、选择和局部窗口 | 从训练开始就面向稀疏和硬件 | 结构分支更多，设计更复杂 |
| MoBA | query 只 attend 到少量 KV blocks | 和 MSA 一样关注 block sparse | 路由和 block 设计不同，早期实现对 kernel 也有挑战 |
| DeepSeek Sparse Attention | token 级 indexer 选重要 token | 细粒度更强 | indexer 自己可能成为长上下文瓶颈 |
| MSA | 训练一个轻量 indexer，按 GQA group 动态选 blocks，再做 softmax attention | 保留 softmax 主分支，选择内容相关，GPU block 友好 | 需要训练/继续训练和专门 kernel |

下面把几组最容易混淆的方法展开说。

### 18.1 MSA vs GQA

GQA 的目标是减少 KV heads。

如果原来有 64 个 query heads，也有 64 个 KV heads，那么 KV cache 很大。GQA 可以变成 64 个 query heads 共享 4 个 KV heads。这样 KV cache 和 KV 读写都少很多。

但 GQA 仍然有一个问题：

```text
每个 query head 仍然要看完整上下文。
```

MSA 是在 GQA 之上再做稀疏选择：

```text
每个 GQA group 先选 Top-K KV blocks，再只读这些 blocks。
```

所以 GQA 是“减少 KV head 数量”，MSA 是“减少每个 query 实际读取的 KV token 数量”。两者不是替代关系，MSA 直接建立在 GQA 上。

### 18.2 MSA vs Longformer / BigBird / StreamingLLM

Longformer 和 BigBird 属于比较经典的固定稀疏注意力。

Longformer 用局部窗口加任务相关 global attention。BigBird 用局部窗口、随机 attention 和 global tokens 组合，把复杂度降下来。

这类方法的特点是：**稀疏模式大多提前规定好**。

比如：

```text
每个 token 看附近 512 个 token
再看若干全局 token
```

这很稳定，也容易理解。但问题是，如果真正相关的信息在很远的位置，而固定规则没有覆盖到，就可能漏掉。

StreamingLLM 则发现 attention sink 很重要：只保留最近窗口会坏掉，但保留开头几个 sink token 可以显著改善流式生成稳定性。MSA 也观察到了 sink column，但 MSA 让 Index Branch 学出 sink、本地和远距离模式。

区别是：

> 固定稀疏方法主要按位置规则省计算；MSA 按内容相关性动态选 blocks。

### 18.3 MSA vs H2O / SnapKV / Quest

H2O、SnapKV、Quest 都更偏 inference-time KV 选择或 KV cache 管理。

H2O 的思路是识别 heavy hitter tokens，也就是少数对 attention 结果贡献很大的 token，然后动态保留 heavy hitters 和最近 token。

SnapKV 是 training-free 的 KV cache 压缩方法，它利用模型在生成前已经暴露出的注意力模式来减少 KV cache。

Quest 是 query-aware 的 KV page 选择。它根据当前 query 估计哪些 KV cache pages 重要，只加载 Top-K pages。

这些方法很实用，因为它们通常不要求重新训练模型。代价是：模型训练时仍然是 full attention；推理时的选择策略需要额外设计，并且通常作为已有模型的外挂优化。

MSA 的选择分支是模型结构的一部分，并且在训练中通过 KL loss 学习。它的目标是从训练/继续训练阶段就让模型适应稀疏 block attention。

### 18.4 MSA vs MInference / FlexPrefill

MInference 和 FlexPrefill 更关注 **prefill 加速**。

prefill 是模型一次性处理长 prompt 的阶段。长上下文里，prefill 的 attention matrix 很大，所以这里很容易成为瓶颈。

MInference 观察到长上下文 attention matrix 里常见几类稀疏模式，比如 A-shape、Vertical-Slash、Block-Sparse，然后为每个 head 离线选择合适模式，推理时动态构建 sparse indices。

FlexPrefill 则根据输入和 attention head 动态调整稀疏模式和预算，比如在 query-specific 模式和预定义模式之间切换。

它们和 MSA 的共同点是：都不满足于固定窗口，都想根据内容或 head pattern 做动态稀疏。

区别是：

- MInference / FlexPrefill 更像已有模型上的推理加速策略。
- MSA 是一个训练进模型里的 Index Branch + Main Branch 结构。
- MSA 明确把 GQA group、block selection、KL training、KV-outer kernel 放在一个整体设计里。

### 18.5 MSA vs InfLLM

InfLLM 是另一条路线：它把远距离上下文放进额外 memory units，需要时再检索相关 memory。

这和 MSA 的气质很不一样。

InfLLM 更像：

```text
我有一个外部记忆系统，长文本先进记忆，需要时检索。
```

MSA 更像：

```text
我仍然在 Transformer attention 层里工作，只是每层先选相关 KV blocks。
```

所以 InfLLM 适合从“无限上下文/外部记忆”的角度理解，MSA 适合从“原生 attention 层如何稀疏化”的角度理解。

### 18.6 MSA vs NSA

NSA，也就是 Native Sparse Attention，是和 MSA 最接近的一类：它也是原生可训练、硬件对齐的 sparse attention。

NSA 采用动态层级稀疏策略，把几类信息结合起来：

- coarse-grained token compression，用压缩方式保留全局上下文。
- fine-grained token selection，细粒度选择重要 token。
- local window，保留局部精确信息。

这套设计更“组合式”：压缩、选择、局部窗口都在结构中占一席。

MSA 的设计更克制：

- 只有一个轻量 Index Branch。
- 只选 block，不额外引入压缩 branch。
- 在每个 GQA group 内独立选 Top-K blocks。
- Main Branch 只对 selected blocks 做标准 attention。

可以这样理解：

> NSA 像一个多工具箱：全局压缩、局部窗口、细粒度选择都要用。MSA 更像一个简化版路线：先按块找相关内容，再只对这些块做精读。

MSA 的简单并不意味着它一定比 NSA 更好；它的设计目标更偏“Occam's razor”：保留最少必要组件，让训练和 GPU 部署更直接。

### 18.7 MSA vs MoBA

MoBA，全称 Mixture of Block Attention，也和 MSA 一样关注 block sparse attention。

MoBA 的核心思想也是：每个 query 不看全部上下文，只 attend 到少量 key-value blocks。

它和 MSA 的相似点：

- 都是按 block 选择，而不是只做固定窗口。
- 都希望降低长上下文 attention 的计算量。
- 都关注 GPU 上 block sparse attention 是否真的跑得快。

差异在于：

- MoBA 更强调 mixture/router 式 block attention。
- MSA 明确绑定 GQA group，每个 GQA group 选自己的 Top-K blocks。
- MSA 的 Index Branch 只额外加索引用 Q/K 投影，并用 KL loss 对齐 Main Branch。
- MSA 的论文重点之一是把算法和 KV-outer sparse attention kernel 一起设计。

如果把它们都看成“按块检索上下文”，那么 MoBA 更像一类 block attention router，MSA 更像 GQA-native 的轻量 block indexer。

### 18.8 MSA vs DeepSeek Sparse Attention

DeepSeek-V3.2 论文里提出了 DeepSeek Sparse Attention，也就是 DSA。相关后续工作 HISA、MISA 对 DSA 的描述可以帮助理解它的关键特点：DSA 用轻量 token-wise indexer 为每个 query 扫描前缀 token，然后选最重要的 token 进入主 attention。

这类 token 级选择的优点是粒度细。它可以只选具体 token，而不是整个 block。

代价是 indexer 本身可能很贵：如果每个 query 都要扫描很长前缀，长上下文下 indexer 也可能成为瓶颈。HISA、MISA、IndexCache 等后续工作基本都在围绕这个瓶颈做优化，比如层级索引、减少活跃 indexer heads、跨层复用索引。

MSA 在这里选择了另一种折中：

- block 级最终选择。
- 用 max pooling 让 block 里最相关的 token 能代表这个 block。
- 牺牲一点粒度，换取更连续的 KV 读取和更容易部署的 kernel。

所以：

> DSA 更细，MSA 更块状；DSA 的挑战容易落在 indexer 扫描成本，MSA 的挑战更多落在 block 粒度和 block sparse kernel。

### 18.9 MSA vs FlashAttention

FlashAttention 是 exact attention 的 IO-aware kernel 设计。

它的核心贡献是：完整 attention 仍然精确计算，但通过 tiling 减少 HBM 和 SRAM 之间的数据搬运。

MSA 和 FlashAttention 的关系更像上下游：

- FlashAttention 告诉我们：attention 速度不只看 FLOPs，还要看 IO。
- MSA 继承了这个思路：只减少 FLOPs 不够，还要设计 KV-outer sparse attention kernel，把 block sparse 真正映射到 GPU 上。

所以 MSA 是在 sparse attention 场景里继续做 IO-aware kernel co-design。

### 18.10 总结对比

MSA 的位置可以概括成：

> 它把 softmax attention 做成可训练、GQA 原生、硬件友好的动态 block sparse attention。

各方法：

- GQA：少存一些 KV heads。
- Longformer / BigBird：按固定模式少看一些位置。
- StreamingLLM：保留 sink 和最近窗口，让流式生成稳定。
- H2O / SnapKV：推理时压缩 KV cache。
- Quest：推理时按 query 选择 KV pages。
- MInference / FlexPrefill：推理 prefill 时动态稀疏化。
- InfLLM：把远距离上下文变成可检索记忆。
- NSA：原生训练的多分支 sparse attention。
- MoBA：按 block 路由 query 到少数 KV blocks。
- DSA：用 token-wise indexer 做细粒度 sparse attention。
- FlashAttention：从 IO 角度把 attention kernel 跑快。
- MSA：按 GQA group 用轻量 indexer 选 Top-K KV blocks，再在这些 blocks 上做标准 attention。

## 19. 用一个完整例子串起来

假设有 1M tokens 的代码仓库上下文。当前 query 问：

```text
这个函数为什么会在空指针输入时崩溃？
```

普通 full attention 会尝试让当前 query 和前面所有 token 做比较。

MSA 的一层大概这样工作：

1. 把 1M tokens 切成 blocks，每个 block 128 tokens，所以大约有 7813 个 blocks。
2. 当前 query 在每个 GQA group 里生成一个轻量 `Q_idx`。
3. 所有历史 token 生成轻量 `K_idx`。
4. Index Branch 给每个可见 token 打分。
5. 每个 block 取最大 token score 作为 block score。
6. 选出 Top-16 blocks，并强制包含 local block。
7. Main Branch 只读取这些 blocks 里的 K/V，最多 2048 tokens。
8. 在这 2048 tokens 上做标准 softmax attention。
9. KL loss 在训练时让 Index Branch 更像 Main Branch 的关注分布。

如果某个远处文件里有相关函数定义，Index Branch 有机会把那个 block 选出来。相比固定 sliding window，它不被位置窗口限制。相比 full attention，它又不用每次看完整 1M tokens。

## 20. 几个容易误解的地方

有几个相近概念容易混在一起，单独做一次纠偏。

### 20.1 MSA 不是简单把 KV cache 删掉

KV cache 压缩方法通常是在推理时决定保留哪些历史 KV，丢掉哪些 KV。它们关心的是：

```text
历史 KV cache 太大，能不能少存、少读？
```

MSA 的问题设定更靠近 attention 结构本身：

```text
每一层、每个 GQA group，能不能先动态选 blocks，再只对这些 blocks 做标准 attention？
```

所以 MSA 不只是“删 KV cache”。它引入了训练中的 Index Branch、Top-K block selection、KL alignment，以及专门的 sparse attention kernel。它的稀疏模式是模型结构的一部分。

### 20.2 MSA 不是用近似 softmax 替代标准 attention

Main Branch 在被选中的 KV blocks 上做的仍然是标准 softmax attention：

```text
先选 blocks
再对 selected blocks 做正常 attention
```

近似发生在“看哪些 blocks”这一步，而不是把 softmax 公式换掉。被选中的 blocks 内部，attention 仍然按正常方式算分数、softmax、加权 value。

这也是为什么需要 KL loss：它让 Index Branch 学会选出主 attention 在意的 blocks。

### 20.3 MSA 不等于 FlashAttention

FlashAttention 解决的是：

```text
完整 attention 怎么通过更好的 IO tiling 跑得更快？
```

MSA 解决的是：

```text
长上下文里能不能不要看完整上下文，只看动态选中的 blocks？
```

二者层级不同。FlashAttention 是 exact dense attention kernel 的经典代表；MSA 是 sparse attention 结构和 sparse kernel 的共同设计。MSA 仍然继承了 FlashAttention 的核心工程思想：不能只看 FLOPs，还要看 IO、tiling、LSE 合并和硬件执行形状。

### 20.4 论文 kernel 和公开仓库源码要区分

论文报告的 H800 结果对应论文里的 kernel design。公开仓库 `MiniMax-AI/MSA` 目前给出的是 inference kernel 代码，其中 SM100 sparse attention 使用 CuTe-DSL，TopK 公开实现也和论文里描述的 min-heap TopK 不完全一样。

所以读源码时要分清两层：

| 层 | 应该怎么看 |
|---|---|
| 论文设计 | 理解 MSA 为什么需要 exp-free TopK、KV-outer、two-phase forward |
| 公开源码 | 理解当前仓库怎样用 `sparse_topk_select`、`build_k2q_csr`、`SparseAttentionForwardSm100`、`combine` 实现 sparse attention |

这是论文系统和公开工程实现之间的正常差异。写博客时最容易犯的错误，就是把论文里的 kernel 伪描述和公开仓库里的具体实现混成一个完全相同的东西。

### 20.5 MSA 的收益主要出现在超长上下文

如果上下文只有几千 token，full attention 本来就没那么离谱，MSA 的 Index Branch、TopK、CSR、schedule、combine 这些额外开销可能不划算。

MSA 有优势的场景是：

- 上下文很长，比如几十万到百万 token。
- 模型需要保留远距离检索能力，而不是只靠最近窗口。
- attention 成本在系统里占比足够高。
- 有配套 kernel 能把 block sparse 映射到 GPU 上。

也就是说，MSA 是为超长上下文大模型准备的一套结构和工程方案。

## 21. 这篇论文最值得带走的点

MSA 的关键是几个设计咬合在一起：

1. **GQA group 级别选择**：每个 KV group 有自己的选择结果，比全局一个 sparse pattern 更灵活。
2. **Block sparse**：选择连续 KV blocks，而不是零散 token，方便 GPU 执行。
3. **轻量 Index Branch**：只额外加索引用 Q/K 投影，选择开销可控。
4. **Max-pooling block score**：block 里只要有强相关 token，就能被选中。
5. **Top-K without softmax**：选择时不需要 exp 和归一化。
6. **KL alignment**：给不可导的 Top-K indexer 提供直接训练信号。
7. **Gradient detach**：让 KL loss 只训练 indexer，不扰动 backbone。
8. **Indexer warmup**：避免训练早期随机 sparse routing 破坏学习。
9. **KV-outer kernel**：把理论稀疏转成 GPU 上的速度收益。

> MSA 是一种为超长上下文设计的、GQA 原生的、按块动态选择的 sparse softmax attention：先用轻量索引器找相关 KV blocks，再在这些 blocks 上做标准 attention，并通过 KL loss 和专门 kernel 让它既能训练稳定，又能实际跑快。

## Reference

MSA 原论文和项目：

- Paper PDF: <https://arxiv.org/pdf/2606.13392>
- Paper HTML: <https://arxiv.org/html/2606.13392v2>
- Kernel repo: <https://github.com/MiniMax-AI/MSA>
- Kernel CuTe-DSL README: <https://raw.githubusercontent.com/MiniMax-AI/MSA/main/python/fmha_sm100/cute/README.md>
- MiniMax-M3: <https://huggingface.co/MiniMaxAI/MiniMax-M3>

基础 attention / kernel：

- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints: <https://arxiv.org/abs/2305.13245>
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness: <https://arxiv.org/abs/2205.14135>

固定稀疏和 streaming：

- Longformer: The Long-Document Transformer: <https://arxiv.org/abs/2004.05150>
- Big Bird: Transformers for Longer Sequences: <https://arxiv.org/abs/2007.14062>
- Efficient Streaming Language Models with Attention Sinks: <https://arxiv.org/abs/2309.17453>

推理时 KV 选择 / 稀疏 prefill：

- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models: <https://arxiv.org/abs/2306.14048>
- SnapKV: LLM Knows What You are Looking for Before Generation: <https://arxiv.org/abs/2404.14469>
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference: <https://arxiv.org/abs/2406.10774>
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention: <https://arxiv.org/abs/2407.02490>
- FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference: <https://arxiv.org/abs/2502.20766>
- InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory: <https://arxiv.org/abs/2402.04617>

原生可训练 sparse attention：

- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention: <https://arxiv.org/abs/2502.11089>
- MoBA: Mixture of Block Attention for Long-Context LLMs: <https://arxiv.org/abs/2502.13189>
- Optimizing Mixture of Block Attention: <https://arxiv.org/abs/2511.11571>
- InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation: <https://arxiv.org/abs/2509.24663>
- DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models: <https://arxiv.org/abs/2512.02556>
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention: <https://arxiv.org/abs/2603.28458>
- MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference: <https://arxiv.org/abs/2605.07363>
