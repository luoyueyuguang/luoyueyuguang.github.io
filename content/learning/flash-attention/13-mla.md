前面 12 篇讲的是 FlashAttention 本身的算法与内核。但仓库 `flash_attn/cute/` 里还专门实现了一类**不一样**的注意力：**MLA（Multi-head Latent Attention）**。它不来自 FlashAttention 论文，而是 DeepSeek-V2/V3 用来**压缩 KV cache** 的架构，作者（等）在 FA 仓库里把它写成了 Blackwell 内核。这篇单独讲它：算法是什么、为什么能省显存、FA 仓库的内核怎么实现。

> **MLA 要解决的问题不是"算得快"，是"KV cache 太大"。** 标准 MHA 每个 token 存 `$ (n_h \cdot d_h) \times 2 $` 的 key/value；MLA 把 key/value 压到一个**小的 latent** `$ c^{KV} $` 里，用的时候再展开。深度模型要放上百层、百万 token 的 KV cache，这个压缩是能不能跑长上下文的关键。

## MHA vs MLA 的 KV 占用

标准 MHA 的 key/value per token（单头）是 `$ d_h $`，`$ n_h $` 个头就是 `$ 2 \cdot n_h d_h $`。MLA 改成：

- 先用一个**降维投影** `$ W^{DKV} $` 把 hidden state `$ h_t $` 压成 **latent** `$ c_t^{KV} \in \mathbb{R}^{d_c} $`（`$ d_c \ll 2 n_h d_h $`）。
- key/value 从 latent 展开：

$$
k_t^C = W^{UK} c_t^{KV}, \qquad v_t = W^{UV} c_t^{KV}, \qquad k_t^{R} = \mathrm{rope}(W^{KR} h_t)
$$

- KV cache 只存 `$ c_t^{KV} $`（`$ d_c $` 维）和一个**绕 rope 的小 key** `$ k_t^R $`（`$ n_h \cdot d_R $` 维，`$ d_R $` 通常远小于 `$ d_h $`）。展开后的全量 `$ k^C, v $` 不留。

`$ k^C $`（content key）走 `$ c^{KV} $`，`$ k^R $`（rope key）走 `$ h_t $` 直接算。**content 部分是共享的、可压缩的；rope 部分本身量小，单独存。** 这就是 MLA 的"解耦"设计。

## Query 也压

Query 同样压：

$$
c_t^Q = W^{DQ} h_t, \qquad q_t^C = W^{UQ} c_t^Q, \qquad q_t^R = \mathrm{rope}(W^{QR} c_t^Q)
$$

## 注意力的点积如何分解

每个 head 的 query 是 `$ [q_t^C ; q_t^R] $`（content 拼接 rope），key 是 `$ [k_s^C ; k_s^R] $`。**content 和 rope 不交叉**（roformer 的 block-diagonal），所以：

$$
q_t \cdot k_s = q_t^C \cdot k_s^C + q_t^R \cdot k_s^R
$$

得分是两块之和；`$ v_t $` 只由 `$ c^{KV} $` 展开。于是 attention 是：

$$
O_t = \mathrm{softmax}\big(q_t^C K^{C\top} + q_t^R K^{R\top}\big) V
$$

## Absorbed：不展开 K/V 也能算

直接照上面要先把 `$ K^C, V $` 展开（`$ n_h d_h $` 大矩阵），那就白压缩了。**absorbed MLA** 把投影吸收进计算：

- **content score**：`$ q_t^C \cdot (W^{UK} c_s) = (q_t^C W^{UK\top}) \cdot c_s $`。把 `$ q_t^C W^{UK\top} $` 当"content 侧的基"，得分就是"基"点 latent `$ c_s $`，**不用物化 `$ k_s^C $`**。
- **output**：`$ O_t = \mathrm{softmax}(\cdot)\, (W^{UV} c_s) = \big(\mathrm{softmax}(\cdot)\, W^{UV}\big) \cdot c_s $`。先算 `$ P W^{UV} $`，再点 `$ c_s $`，**不用物化 `$ v $`**。

这样整段注意力只在 latent `$ c^{KV} $` 上做，展开的 `$ K^C, V $` 从不落地。这就是"absorbed"名字的来历。FA 仓库内核里，这块对应 `head_dim_v == 512` 且 `head_dim == 64（或 == head_dim_v）` 的 `is_deepseek_mla_absorbed_shape` 模式。

## FA 仓库的 MLA 内核

`flash_attn/cute/` 里是 CuTe-DSL 的 MLA 实现（反向拆成三个核，见 [[learning/flash-attention/15-mla-bwd-kernel|MLA 反向内核]]）：

- `flash_fwd_mla_sm100.py`（前向，3160 行）、`flash_bwd_mla_sm100.py`、`flash_bwd_mla_dq_dqv_sm100.py`、`flash_bwd_mla_dk_sm100.py`（反向拆成 dQ/dV 和 dK 两块）。
- 它在结构上**复用了 FA4 前向那套**（`softmax_loop`、`softmax_step`、`correction_loop`、`mma`），所以 [[learning/flash-attention/09-flashattention4-kernel|FA4 前向内核逐行读]] 里的 TMEM/流水线/exp2 模拟都适用。

和普通 FA4 前向的差别，看一眼它的调用签名就知道：

```python
mQ:  (b, s_q, h, d)      # content query（QK 用）
mQv: (b, s_q, h, dv)     # 另一个投影的 query（QvV 用）
mK:  (b, s_k, h_k, d)    # content key / latent
mV:  (b, s_k, h_k, dv)   # value
```

- `mQ` / `mQv` 是**两个不同投影的 Q**（一个走 QK content 得分，一个走 Qv）。`tiled_mma_QK`（`$ Q \cdot K^\top $`）和 `tiled_mma_QvV`（`$ Q \cdot V $`，或 absorbed 时的 `$ Q \cdot c^{KV} $`）分开算，正是 content/rope 解耦在 kernel 里的映射。
- `has_qk`：**QK 是否直接算**。MLA 有个"吸收"路径：当 `$ Q $` 和 `$ K $` 都是投影后的 content 时，可以直接 QK；而非 absorbed 时 QK 走 `tiled_mma_QK`，V 部分走 `tiled_mma_PVt`。
- `h_k`（KV head 数）很小，通常 `1`：**所有 query head 共享同一组 latent KV**。这是 MQA 式的做法，因为 `$ c^{KV} $` 本来就是"全 head 共享一份"（注意 `W^{DKV} h_t` 没有 head 维）。
- 三个 MMA：`tiled_mma_QK`、`tiled_mma_QvV`、`tiled_mma_PVt`，对应"content score + rope score + output"。

`flash_attn/cute/interface.py` 里给了两个关键形状模式：

```python
is_deepseek_shape = head_dim == 192 and head_dim_v == 128
is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
```

- `(192, 128)`：DeepSeek V2/V3 的原生形状（content 192 + rope 128，或 192 的 QK 维）。
- `(64, 512)` / `(head_dim, 512)`：**absorbed** 模式，`$ d_h=64 $`、`$ d_c=512 $`（大 latent、小 head），用 `$ P W^{UV} $` 吸收 value 投影。

## 一句话

MLA 不是"更快"，是"更省 KV cache"：把 key/value 压成一个 latent `$ c^{KV} $`，content 部分共享、rope 部分单存；然后靠 absorbed 技巧**把 `$ W^{UK}, W^{UV} $` 吸收进得分和输出，全程只在 latent 上算，从不物化大 `$ K^C, V $`**。FA 仓库用 CuTe-DSL 实现了它，结构复用 FA4 的 softmax/流水线，差别只在"Q 有两个投影、KV head 共享、形状 192/128 或 64/512"。

## Reference

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（MLA 论文）：<https://arxiv.org/abs/2405.04434>
- DeepSeek-V3 技术报告（MLA 应用与蒸馏）：<https://arxiv.org/abs/2412.19437>
- flash-attention 仓库（flash_attn/cute/flash_fwd_mla_sm100.py 等）：<https://github.com/Dao-AILab/flash-attention>
- vLLM 的 MLA "absorbed" 实现（投影吸收原理）：<https://github.com/vllm-project/vllm>
- RoFormer（rotary 的 block-diagonal 性质）：<https://arxiv.org/abs/2104.09864>
