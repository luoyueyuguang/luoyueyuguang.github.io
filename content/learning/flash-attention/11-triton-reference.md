[[learning/flash-attention/03-forward-kernel|FA1/FA2 的 CUDA 内核]] 用了大量 cuTe/CUTLASS 抽象（`partition_fragment_A`、`SmemLayoutQ`、cp.async），对没读过 CUTLASS 的人几乎是天书。仓库里另有一版 **Triton 实现** `flash_attn/flash_attn_triton_og.py`，只用 `tl.dot` / `tl.max` / `tl.exp` 这几个原语，把同一个算法写得清清楚楚。这一篇逐行读它，你就能真的看懂 [[learning/flash-attention/02-online-softmax|在线 softmax]] 怎么落到代码。

> **Triton 的价值不是快，是"可读"。** 它把"一个程序块负责一段 query 行、内层循环扫 key/value 块"这个结构直接写了出来。FA2 论文里"沿序列维并行、把 Q 切给 warp"的思路，最初就来自 Tillet 的 Triton 教程。

## 网格与偏移

```python
@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, TMP, L, M, Out, ...,
                Z, H, N_CTX,
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)          # M 块索引（sequence 维）
    off_hz = tl.program_id(1)           # (batch, head) 平面索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 这一块的 query 行号
    offs_n = tl.arange(0, BLOCK_N)                        # 这一块的 key 列号
    offs_d = tl.arange(0, BLOCK_DMODEL)                   # head 维度
```

- 网格是 `(T_M, Z*H)`：`program_id(0)` 是 **行块**（哪种 query 行），`program_id(1)` 是 (batch, head)。
- **每个程序块负责一个 `$ B_r \times d $` 的 query 块，内层扫它需要的全部 key/value 块。** 这就是"沿序列维并行"：不同行块之间完全独立，不需要通信。FA2 的 [[learning/flash-attention/05-flashattention2|序列维并行]] 就是这个结构（Tillet 最先提出，FA2 搬进 CUDA）。

```python
off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
```

`offs_m[:, None] * stride_qm` 是"第几行 × 行步长"，`offs_d[None, :] * stride_qk` 是"第几列 × 列步长"，`[:, None]` / `[None, :]` 把一维索引广播成二维。三个 tile 的全局地址就是这样算出来的。

## 运行状态

```python
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # 行最大（-inf 起）
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # 指数和（0 起）
acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # 归一化的输出累计
q = tl.load(q_ptrs)     # Q 只加载一次，全程留在 SRAM/寄存器
```

`acc` 在这版 Triton 里是**每一步都归一化的**（见下文 `p_scale`），和 [[learning/flash-attention/02-online-softmax|FA1]] 的 `O` 一致。`Q` 加载一次是它的关键：整个循环里 `$ Q $` 不动，只有 K/V 在换。

## 主循环：S = QK^T

```python
for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)     # 提示编译器对齐，方便向量化
    k = tl.load(k_ptrs + start_n * stride_kn)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k, trans_b=True)               # S = Q K^T（tensor core）
    qk *= sm_scale                                  # × 1/sqrt(d)
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))   # causal mask
```

- 循环上界 `(start_m + 1) * BLOCK_M`：**因果限制**。这块 query 的最后一行是 `start_m*BLOCK_M + BLOCK_M - 1`，它只能看 `≤` 它自己的 key，所以 key 列最多扫到 `(start_m+1)*BLOCK_M`。后面那些 key 块整块跳过（约一半的块）。
- `tl.dot(q, k, trans_b=True)`：`$ q k^\top $`，`trans_b` 表示 K 是转置乘。tensor core 指令。
- `qk *= sm_scale`：缩放 `$ 1/\sqrt{d} $`。
- `tl.where(...)`：因果掩码。`offs_m[:,None] >= start_n + offs_n[None,:]`（query 行 ≥ key 列）为真留 `0`，否则 `-inf`，即"只看之前的位置"。

## 在线 softmax

```python
m_ij = tl.max(qk, 1)                     # 这一块的 row max
p = tl.exp(qk - m_ij[:, None])           # P_ij = exp(S - m_ij)
l_ij = tl.sum(p, 1)                      # 这一块的指数和

m_i_new = tl.maximum(m_i, m_ij)          # 新的全局行最大
alpha = tl.exp(m_i - m_i_new)            # e^{m_old - m_new}
beta  = tl.exp(m_ij - m_i_new)           # e^{m_ij - m_new}
l_i_new = alpha * l_i + beta * l_ij      # 合并指数和
```

这就是 [[learning/flash-attention/02-online-softmax|合并公式]] 的逐行翻译。`alpha`、`beta` 是两块之间的重缩放因子。

## 更新输出：FA1 版归一化方案

```python
p_scale = beta / l_i_new                 # 把 P 归一化到"新的总 l"
p = p * p_scale[:, None]

acc_scale = l_i / l_i_new * alpha        # 把旧 acc 从"旧 l 归一化"搬到"新 l 归一化"
tl.store(t_ptrs, acc_scale)              # BUG: 必须先 store 再 load
acc_scale = tl.load(t_ptrs)
acc = acc * acc_scale[:, None]

v = tl.load(v_ptrs + start_n * stride_vk)
p = p.to(v.dtype)
acc += tl.dot(p, v)                      # O += P · V
l_i = l_i_new
m_i = m_i_new
```

- `acc` 一直保持**按当前 `$ \ell $` 归一化**。所以合并时旧 `acc` 要乘 `acc_scale = l_i/l_i_new * alpha`（旧到新 `$ \ell $` 的转换 + 新旧 max 的 `$ e^{m_{old}-m_{new}} $`），新块 `$ P $` 要乘 `p_scale = beta/l_i_new`。
- `tl.store/tl.load(t_ptrs)` 是一处 **Triton 编译器 bug 的 workaround**：注释写着"have to store and immediately load"。这属于历史遗留，不必深究。
- `p = p.to(v.dtype)`：`$ P $` 从 fp32 转成 v 的精度（fp16/bf16）喂 tensor core。

## 写回

```python
l_ptrs = L + off_hz * N_CTX + offs_m
m_ptrs = M + off_hz * N_CTX + offs_m
tl.store(l_ptrs, l_i)
tl.store(m_ptrs, m_i)
...
tl.store(out_ptrs, acc)     # O
```

写出 `$ \ell, m $`（给反向）和 `$ O $`。注意这版存的是分离的 `$ m, \ell $`，和 FA2 只存 `$ L = m + \log \ell $` 不同。

## 新旧 Triton 的差别：FA1 vs FA2

上面是 `flash_attn_triton_og.py`（og=original，FA1 方案）。新版 `flash_attn_triton.py` 换成了 FA2 方案，核心差异三行：

```python
# 旧（FA1）：acc 每步都除 l（保持归一化）
p_scale = beta / l_i_new
acc = acc * acc_scale[:, None]
# 新（FA2）：acc_o 攒"未除 l"的 Õ，每步只按 max 变化重缩，结尾才除 l
m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)   # 用上一轮 lse_i 当 max 下界
acc_o_scale = tl.exp(m_i - m_ij)                          # 每步按新旧 max 重缩（不除 l）
acc_o = acc_o * acc_o_scale[:, None]
acc_o += tl.dot(p, v)                                     # 累加未归一化 Õ
l_i_new = tl.exp(lse_i - m_ij) + l_ij
lse_i = m_ij + tl.log(l_i_new)                            # 只维护 L = m + log l
o_scale = tl.exp(m_i - lse_i)                             # 结尾 × 1/l
tl.store(lse_ptrs, lse_i)                                 # 只存 L
```

- 新版**每步仍会按 max 变化重缩** `acc_o`（`acc_o_scale`），但**不再除 `$ \ell $`**：`acc_o` 攒的是未归一化的 `$ \widetilde{O} $`，结尾乘 `o_scale = e^{m_i - \text{lse}_i} = 1/\ell` 得到结果。这就是 [[learning/flash-attention/05-flashattention2|FA2 微调 1]]。
- 只存 logsumexp `L`，不存 `m` 和 `l` 两个。这是 [[learning/flash-attention/05-flashattention2|FA2 微调 2]]。
- 注意 `m_ij = tl.maximum(..., lse_i)` 用的是**上一轮的 lse_i** 当作 max 的下界（因为 `L = m + \log \ell \ge m`），多加了一层数值保护。

## 一句话

Triton 版把 FA 的核心结构压缩到了 30 多行：**一个程序块一段 query 行，`tl.dot` 算 QK^T，`tl.max/tl.exp/tl.sum` 做在线 softmax，`tl.dot` 算 P·V，`m/l` 两个统计量从头攒到尾。** 它比 cuTe 版少了几层抽象，但算法分毫不差。想读懂任何一个版本的 flash attention，先读懂这 30 行。

## Reference

- flash-attention 仓库（flash_attn/flash_attn_triton_og.py、flash_attn_triton.py）：<https://github.com/Dao-AILab/flash-attention>
- Triton fused attention 教程（Phil Tillet）：<https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py>
- FlashAttention 论文（forward）；FlashAttention-2 论文（算法微调）：<https://arxiv.org/abs/2205.14135>
