前面 12 篇讲的是 FlashAttention 本身的算法与内核。但仓库 `flash_attn/cute/` 里还专门实现了一类**不一样**的注意力：**MLA（Multi-head Latent Attention）**。它不来自 FlashAttention 论文，而是 DeepSeek-V2/V3 用来**压缩 KV cache** 的架构，仓库里另有一条实现线把它写成了 Blackwell 内核（`*_mla_sm100.py` 几个文件头署名 Colfax International）。这篇单独讲它：算法是什么、为什么能省显存、FA 仓库的内核怎么实现。

> **MLA 要解决的问题不是"算得快"，是"KV cache 太大"。** 标准 MHA 每个 token 存 $ (n_h \cdot d_h) \times 2 $ 的 key/value；MLA 把 key/value 压到一个**小的 latent** $ c^{KV} $ 里，用的时候再展开。深度模型要放上百层、百万 token 的 KV cache，这个压缩是能不能跑长上下文的关键。

## MHA vs MLA 的 KV 占用

标准 MHA 的 key/value per token（单头）是 $ d_h $，$ n_h $ 个头就是 $ 2 \cdot n_h d_h $。MLA 改成：

- 先用一个**降维投影** $ W^{DKV} $ 把 hidden state $ h_t $ 压成 **latent** $ c_t^{KV} \in \mathbb{R}^{d_c} $（$ d_c \ll 2 n_h d_h $）。
- key/value 从 latent 展开：

$$
k_t^C = W^{UK} c_t^{KV}, \qquad v_t = W^{UV} c_t^{KV}, \qquad k_t^{R} = \mathrm{rope}(W^{KR} h_t)
$$

- KV cache 只存 $ c_t^{KV} $（$ d_c $ 维）和一个**带 rope 的小 key** $ k_t^R $（$ d_h^R $ 维，而且**所有 head 共用同一个**——$ W^{KR} \in \mathbb{R}^{d_h^R \times d} $ 没有 head 维，head 维只出现在 query 侧的 $ W^{QR} \in \mathbb{R}^{n_h d_h^R \times d_c'} $ 里）。展开后的全量 $ k^C, v $ 不留。

$ k^C $（content key）走 $ c^{KV} $，$ k^R $（rope key）走 $ h_t $ 直接算。**content 部分是共享的、可压缩的；rope 部分本身量小，单独存。** 这就是 MLA 的"解耦"设计——为什么非解耦不可，见下一节。

拿 DeepSeek-V3 的真实数字算一笔（`config_671B.json`：`n_heads=128`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`）：若把 K、V 完整展开存下来，每个 token 是 $ n_h (d^{\text{QK}} + d^V) = 128\,(192+128) = 40960 $ 个元素；MLA 每个 token 只存 $ d_c + d_h^R = 512 + 64 = 576 $ 个，约 **71×** 的压缩。DeepSeek-V2 的 Table 1 把它记成 $ (d_c + d_h^R) l \approx \frac{9}{2} d_h l $（$ d_h = 128 $ 时正是 576），并指出这相当于"只有 2.25 个 group 的 GQA"的 KV cache。这就是 MLA 用一点计算换来的长上下文能力。

## Query 也压

Query 同样压（V2 明确说这一步**不省 KV cache**，省的是训练时的 activation memory）：

$$
c_t^Q = W^{DQ} h_t, \qquad q_t^C = W^{UQ} c_t^Q, \qquad q_t^R = \mathrm{rope}(W^{QR} c_t^Q)
$$

## 注意力的点积如何分解

每个 head 的 query 是 $ [q_t^C ; q_t^R] $（content 拼接 rope），key 是 $ [k_t^C ; k_t^R] $。两段占的是拼接向量里互不重叠的坐标，所以点积天然拆成两块：

$$
q_t \cdot k_s = q_t^C \cdot k_s^C + q_t^R \cdot k_s^R
$$

**为什么必须解耦？** RoPE 是**按位置**作用在 key 上的旋转。如果直接把 RoPE 加在 $ k_t^C = W^{UK} c_t^{KV} $ 上，$ W^{UK} $ 后面就跟着一个只跟"当前生成到哪个位置"有关的矩阵；吸收式里 $ W^{UQ} $ 和 $ W^{UK} $ 之间夹进它之后，矩阵乘法不可交换，$ W^{UK} $ 就没法预先并进 $ W^{UQ} $——每来一个新 token 都得把整个 prefix 的 key 重算一遍，压缩 KV cache 换来的收益全吐回去。所以 MLA 另开一路只有 $ d_h^R $ 维的小通道专门承载位置信息（key 侧是所有 head 共享的一份），让 content 那一路保持"位置无关、可吸收"。

得分是两块之和；$ v_t $ 只由 $ c^{KV} $ 展开。于是 attention 是（$ i $ 指第 $ i $ 个 head）：

$$
O_{t,i} = \mathrm{softmax}\big(q_{t,i}^{C\top} K^{C\top} + q_{t,i}^{R\top} K^{R\top}\big) V
$$

## Absorbed：不展开 K/V 也能算

直接照上面要先把 $ K^C, V $ 展开（$ n_h d_h $ 大矩阵），那就白压缩了。先写下几个投影的形状（V2 §2.1.2），推导要用：

$$
W^{DKV} \in \mathbb{R}^{d_c \times d}, \qquad W^{UK}, W^{UV} \in \mathbb{R}^{n_h d_h \times d_c}, \qquad W^{KR} \in \mathbb{R}^{d_h^R \times d}
$$

按 head 把 $ W^{UK}, W^{UV} $ 切块（第 $ i $ 块 $ W_i^{UK}, W_i^{UV} \in \mathbb{R}^{d_h \times d_c} $），**absorbed MLA** 就是把这两个投影从 $ c^{KV} $ 那一侧挪到等式另一侧：

- **content score**：$ q_{t,i}^C \cdot (W_i^{UK} c_s) $ 是 $ d_h $ 维点积，但夹在中间的 $ c_s $ 是所有 head 共享的 $ d_c $ 维向量，于是可以把 $ W_i^{UK} $ 吸收进 query：

$$
q_{t,i}^{C\top} W_i^{UK} c_s = \underbrace{\big(W_i^{UK\top} q_{t,i}^C\big)}_{\tilde q_{t,i}^C \in \mathbb{R}^{d_c}} \cdot c_s
$$

  离线把 $ \tilde q_{t,i}^C = W_i^{UK\top} q_{t,i}^C $ 预计算好（等价于把 $ W^{UK\top} $ 并进 $ W^{UQ} $），得分就变成"$ d_c $ 维的 query"点 latent $ c_s $，**不用物化 $ k_s^C $**。

- **output**：同样把 $ W_i^{UV} $ 挪到求和号外面（结合律），先对 latent 加权求和：

$$
\tilde o_{t,i} = \sum_s P_{ts}^{(i)} c_s \in \mathbb{R}^{d_c}, \qquad o_{t,i} = W_i^{UV} \tilde o_{t,i}
$$

  即每头用自己的 $ P^{(i)} $ 去对 $ d_c $ 维 latent 加权（而不是对 $ d_h $ 维的 $ v $ 加权），**不用物化 $ v $**。再往外一层，最终输出投影是 $ u_t = W^O [o_{t,1};\dots;o_{t,n_h}] $，于是

$$
u_t = \underbrace{W^O \cdot \mathrm{blkdiag}(W_1^{UV},\dots,W_{n_h}^{UV})}_{\text{离线合并}}\;[\tilde o_{t,1};\dots;\tilde o_{t,n_h}]
$$

  也就是 $ W^{UV} $ 整体也能并进 $ W^O $。

这就是 V2 里那句"$ W^{UK} $ 可以吸收进 $ W^Q $、$ W^{UV} $ 可以吸收进 $ W^O $，于是推理时连 key/value 都不需要算出来"的完整含义。整段注意力只在 latent $ c^{KV} $ 上做：得分的两侧都是 $ d_c $ 维、输出的中间量也是 $ d_c $ 维，展开的 $ K^C, V $ 从不落地——这就是"absorbed"名字的来历。代价是内容分和 rope 分要分别算再相加，以及下标重排（$ W_i^{UK\top}W_i^{UQ} $、$ W^O\cdot\mathrm{blkdiag}(W^{UV}) $ 都能离线算掉）。FA 仓库内核里，这块对应 `head_dim_v == 512` 且 `head_dim == 64（或 == head_dim_v）` 的 `is_deepseek_mla_absorbed_shape` 模式。

## FA 仓库的 MLA 内核

`flash_attn/cute/` 里是 CuTe-DSL 的 MLA 实现（反向拆成主核 + 两个 GEMM 核，见 [[learning/flash-attention/15-mla-bwd-kernel|MLA 反向内核]]）：

- `flash_fwd_mla_sm100.py`（前向，3176 行）、`flash_bwd_mla_sm100.py`（`FlashAttentionSparseMLABackwardSm100`）、`flash_bwd_mla_dq_dqv_sm100.py`（`dQdQvGemmKernel`）、`flash_bwd_mla_dk_sm100.py`（`dKGemmKernel`）。
- 它在结构上**复用了 FA4 前向那套**（`softmax_loop`、`softmax_step`、`correction_loop`、`mma`），所以 [[learning/flash-attention/11-flashattention4-kernel|FA4 前向内核逐行读]] 里的 TMEM/流水线/exp2 模拟都适用。

和普通 FA4 前向的差别，看一眼它的调用签名就知道：

```python
mQ:  (b, s_q, h, d)      # rope 那一路的 query，d = 64 = qk_rope_head_dim
mQv: (b, s_q, h, dv)     # 吸收后的 content query，dv = 512 = kv_lora_rank
mK:  (b, s_k, h_k, d)    # rope key（所有 head 共用，h_k = 1）
mV:  (b, s_k, h_k, dv)   # latent c^{KV}
```

- `mQ` / `mQv` 是**两个不同投影的 Q**：`mQ` 带 decoupled rope，`mQv` 是 absorbed 后的 content query（$ \tilde q = W^{UK\top} q^C $，$ d_c = 512 $ 维）。`mma` 的源码注释写得很直白——"Computes Q @ K^T, Qv @ V^T, and P @ V"，而且第一个 gemm 带 `zero_init=True`、第二个不带，两者**累加到同一块 TMEM 的 S 上**：

$$
S = \underbrace{Q K^\top}_{\text{rope 分}} + \underbrace{Q_v V^\top}_{\text{content 分（对 latent）}}
$$

  这正是 content/rope 解耦在 kernel 里的样子：`mK` 是那个 64 维的共享 rope key，`mV` 是 512 维 latent。
- **`mV` 一份数据当两个 operand 用**：$ Q_v V^\top $ 里它是"K"（被 query 点），$ P V $ 里它是"V"（被 softmax 权重点），后者的输出就是 $ \tilde o_{t,i}\in\mathbb{R}^{d_c} $（`mO` 的最后一维是 `dv`），$ W^{UV} $ 留在 attention 外面并进 $ W^O $。
- `has_qk`（`interface.py` 里就是 `q is not None`）：调用方**有没有**传 `q`/`k` 这一对。传了就是上面那套（`head_dim=64` 的 rope key 单独走 `tiled_mma_QK`）；不传（只有 `qv`、`v`，且 `head_dim == head_dim_v == 512`）就没有单独的 rope 项，S 只由 $ Q_v V^\top $ 一项给出，同一个 512 维 cache 既当 key 又当 value。
- `h_k`（KV head 数）很小，通常 `1`：**所有 query head 共享同一组 latent KV**。这是 MQA 式的做法，因为 $ c^{KV} $ 本来就是"全 head 共享一份"（$ W^{DKV} h_t $ 没有 head 维）；DSA 那条 top-k gather 路径干脆断言 `qhead_per_kvhead == 128`。
- 三个 MMA：`tiled_mma_QK`（rope 得分）、`tiled_mma_QvV`（content 得分，累加）、`tiled_mma_PVt`（输出），对应"S 的两项 + O"。

`flash_attn/cute/interface.py` 的 `_validate_head_dims` 里给了两个 SM100 允许的形状模式：

```python
is_deepseek_shape = head_dim == 192 and head_dim_v == 128
is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
```

- `(192, 128)`：DeepSeek V2/V3 的**非吸收**形状，`head_dim=192` 是 QK 维 = content（nope）128 + rope 64（DeepSeek-V3 `qk_nope_head_dim=128`、`qk_rope_head_dim=64`），`head_dim_v=128` 是 V 维。注意它走的是**普通 FA4 前向核**（论文里 (192, 128) 的 benchmark 用的就是普通核），`is_deepseek_shape` 只是 SM100 允许列表里的一项，`flash_fwd_mla_sm100.py` 碰不到它。
- `(64, 512)` / `(512, 512)`：**absorbed** 形状，`head_dim_v=512` 就是 latent 维 $ d_c $。`head_dim=64` 是 decoupled rope 维 $ d_h^R $（不是 $ d_h $）；`head_dim == head_dim_v == 512` 时没有单独的 rope 那一对，`has_qk=False`、score 只剩 $ Q_v V^\top $ 一项。只有调用方传了 `qv` 才会进这个核——`interface.py` 里对 qv 那条路径的断言就是 `head_dim_v == 512` 且 `q is None or head_dim == 64`。

## MLA 被替换了吗？

没有——**MLA 仍是 DeepSeek 的稠密注意力底座**。2025-09 发布的 **DeepSeek-V3.2-Exp** 在 MLA 之上**叠加**了 DeepSeek Sparse Attention（DSA）：用一个小的 lightning indexer 把 token 级稀疏当作"哪块 KV 要看"的选择器，主注意力仍走 MLA 的 latent。所以本文讲的 latent 压缩与 absorbed 技巧在 V3.2 里原样成立，只是外面套了一层稀疏。反过来的例子是 DeepSeek 自己的 **FlashMLA**（和 FA 仓库的 MLA 是两套实现）：它有稠密 MLA decode 核（H800 上 memory-bound 到 3000 GB/s、compute-bound 到 660 TFLOPS）和 DSA 稀疏核（FP8 KV cache、bf16 算，H800 上 410 TFLOPS，B200 上到 700 TFLOPS），稀疏 prefill 则是 640 TFLOPS（H800）/1450 TFLOPS（B200）。这些数字是 DeepSeek 的核，不是 FA 仓库 `flash_attn/cute` 里那套；两者别混。稀疏之外，MLA 本身没被替换——FlashMLA 的 README 里最新的条目已经是 DeepSeek-V4/V4.1 的稀疏核（prefill / decoding，FP8 或 FP4 KV cache），其 MLA 模式依然是 MQA（`head_dim_k`=512、`head_dim_v`=512），说明从 V3.2-Exp 到 V4.1，稀疏始终是**叠加**在 MLA 上的一层，MLA 这门"压缩 KV cache"的技术底座没被换掉。

## 一句话

MLA 不是"更快"，是"更省 KV cache"：把 key/value 压成一个 latent $ c^{KV} $，content 部分共享、rope 部分单存（一份 $ d_h^R $ 维的小 key，所有 head 共用）；然后靠 absorbed 技巧**把 $ W^{UK}, W^{UV} $ 吸收进得分和输出，全程只在 latent 上算，从不物化大 $ K^C, V $**。FA 仓库用 CuTe-DSL 实现了它，结构复用 FA4 的 softmax/流水线，差别只在"Q 有两个投影、KV head 共享一份、latent 同时当 key 和 value、形状是 (64,512)/(512,512) 这批 absorbed 形状（(192,128) 那套非吸收形状走的是普通 FA4 核）"。

## Reference

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（MLA 论文）：<https://arxiv.org/abs/2405.04434>
- DeepSeek-V3 技术报告（MLA 应用与蒸馏）：<https://arxiv.org/abs/2412.19437>
- DeepSeek-V3 MLA 超参（config_671B.json：kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim / v_head_dim / n_heads）：<https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json>
- DeepSeek-V3.2-Exp（DSA 叠加在 MLA 之上）：<https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
- FlashMLA（DeepSeek 自带 MLA/DSA 核，与 FA 仓库实现不同）：<https://github.com/deepseek-ai/FlashMLA>
- flash-attention 仓库（flash_attn/cute/flash_fwd_mla_sm100.py 等）：<https://github.com/Dao-AILab/flash-attention>
- vLLM 的 MLA absorbed 实现（文件头的推导注释正是 $ W^{UK}/W^{UV} $ 的吸收：`ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)`、`o = einsum(...)` 对 $ W^{UV} $）：<https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py>
- RoFormer（rotary 的 block-diagonal 性质）：<https://arxiv.org/abs/2104.09864>
