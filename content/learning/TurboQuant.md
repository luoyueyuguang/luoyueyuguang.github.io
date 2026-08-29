Coauthor with codex 5.5

这篇文章按三个层次来讲 TurboQuant：

1. **直觉**：为什么 AI 里有很多向量，为什么要量化，为什么 KV cache 特别占显存。
2. **数学**：TurboQuant 到底在优化什么，随机旋转为什么有用，MSE 量化和内积量化为什么不是一回事。
3. **代码**：用 NumPy 写一个教学版 TurboQuant，看到每行代码对应哪条公式。最后**再落到 kernel**：看 packed cache、CUDA block/thread 映射、融合解码和 attention 打分该怎么想。

> TurboQuant 是有损压缩。它承认压缩会丢信息，但让丢掉的信息尽量不影响 AI 在乎的计算，尤其是距离、相似度和内积。

![TurboQuant整体流程](/learning/assets/turboquant-pipeline.svg)

## 1. AI 里的“向量”到底是什么

大模型不会直接拿中文词语做计算。它会把词、token、句子、中间状态都变成一串数字。

比如简化地看：

```text
北京 = [0.12, -0.84, 0.33, 1.27]
天气 = [0.77,  0.05, -0.61, 0.18]
```

这串数字就叫向量。

向量就是“用很多数字描述一个东西”。一个人可以用身高、体重、年龄、收入这些数字描述；AI 也用很多数字描述一个词或一段上下文。

现实里的大模型向量不会只有 4 个数字。它们可能有 128、256、1024、4096 个数字。我们把维度记作 `d`。

数学上写成：

$$
\mathbf{x} = [x_1, x_2, \ldots, x_d]
$$

这里：

- `x` 是一个向量。
- `x_1` 是第 1 个数字。
- `x_d` 是第 d 个数字。
- `d` 是向量长度，也叫维度。

## 2. 为什么要压缩这些向量

大模型推理时，慢的不一定只是“算不动”，很多时候是“数据太多，搬不动”。

可以把 GPU 想成厨房：

- 算力像厨师炒菜的速度。
- 显存像厨房仓库。
- 显存带宽像从仓库拿食材到灶台的速度。

如果食材太多，厨师再快也要等搬运。大模型也是这样：大量权重、激活值、KV cache 都要从显存读出来。数据越大，越慢，也越贵。

TurboQuant 特别关心的一类数据是 **KV cache**。

## 3. KV cache 是什么

大模型生成文字时，是一个 token 一个 token 生成的。

例如要生成：

```text
量化可以节省显存。
```

它可能先生成“量”，再生成“化”，再生成“可”。

每生成一个新 token，模型都要回头看前面已经生成的 token。为了不每次从头重新计算，它会把前面 token 的一些中间结果缓存下来。这里最重要的一类缓存叫 Key 和 Value，合起来叫 **KV cache**。

普通理解：

> KV cache 就是模型读过上下文后留下的“笔记”。

上下文越长，笔记越多。长文本聊天、长文档问答、多轮对话都会让 KV cache 变大。

所以，如果我们能把 KV cache 的向量压小，同时不让模型明显变差，就能：

- 降低显存占用。
- 支持更长上下文。
- 同时服务更多请求。
- 减少显存读写，让推理更快。

这就是量化的现实动机。

## 4. 量化是什么

量化就是把精细数字变成粗略数字。

比如你的身高真实是：

```text
172.348 cm
```

填表时写：

```text
172 cm
```

这就是量化。它丢掉了一点精度，但节省了表达成本。

计算机里也类似。原来一个小数可能用 16 bit 或 32 bit 存。量化后可能只用 8 bit、4 bit、2 bit，甚至 1 bit 存。

bit 就是开关：

```text
1 bit -> 2 种状态
2 bit -> 4 种状态
4 bit -> 16 种状态
8 bit -> 256 种状态
```

如果一个数字原来用 16 bit，现在只用 2 bit，理论上单个数字的存储变成原来的 `1/8`。当然，压缩不是免费午餐，代价是误差。

## 5. 点积：TurboQuant 真正在乎的计算

AI 里经常要问：

> 这两个向量像不像？

一种常见做法叫点积。

两个二维向量：

```text
A = [2, 3]
B = [4, 5]
```

点积就是对应位置相乘再相加：

```text
A · B = 2 * 4 + 3 * 5 = 23
```

一般写成：

$$
\langle \mathbf{x}, \mathbf{y} \rangle
= x_1y_1 + x_2y_2 + \cdots + x_dy_d
$$

大模型注意力里就大量用到类似计算。当前 Query 会和历史 Key 做点积，模型靠这个分数判断“前文哪个位置更重要”。

所以 TurboQuant 在乎的是：

> 压缩后，用这个向量算点积，结果还准不准？

能不能恢复出原向量本身，是次要的。

这是后面 MSE 量化和内积量化分开的原因。

## 6. 两种误差：MSE 和内积误差

假设原始向量是：

$$
\mathbf{x}
$$

压缩再解压后的近似向量是：

$$
\tilde{\mathbf{x}}
$$

第一个目标叫 **MSE**，也就是重建误差：

$$
D_{\text{mse}} =
\mathbb{E}\left[\|\mathbf{x} - \tilde{\mathbf{x}}\|_2^2\right]
$$

普通解释：

- `x` 是原图。
- `x_hat` 是压缩后恢复出来的图。
- MSE 问的是：两张图逐个像素差多少。

第二个目标叫 **inner-product error**，也就是内积误差：

$$
D_{\text{prod}} =
\mathbb{E}\left[
\left|
\langle \mathbf{y}, \mathbf{x} \rangle
-
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
\right|^2
\right]
$$

普通解释：

- `y` 是查询向量，比如当前 Query。
- 原来应该算 `<y, x>`。
- 压缩后只能算 `<y, x_hat>`。
- 内积误差问的是：最终分数差多少。

这两个目标不一样。

一个压缩方法可能很会恢复向量本身，但用它算内积会有系统性偏差。TurboQuant 论文的一个重点就是：**MSE 最优不等于内积最优**。

## 7. 先归一化：把向量放到单位球上

论文里经常假设：

$$
\|\mathbf{x}\|_2 = 1
$$

这句话意思是：向量长度等于 1。

二维里，如果：

```text
x = [3, 4]
```

它的长度是：

$$
\sqrt{3^2 + 4^2} = 5
$$

归一化后变成：

```text
[3/5, 4/5] = [0.6, 0.8]
```

长度就变成 1。

这不是很强的限制。真实系统里可以把长度单独存下来，先量化方向，解码后再乘回长度。

## 8. 普通逐坐标量化为什么不够好

最直接的量化是逐坐标做四舍五入。

例如只允许 4 个格子：

```text
-1.0   -0.33   0.33   1.0
```

真实数字 `0.28` 就存成 `0.33`。

问题是，高维向量可能有异常值：

```text
[0.02, 0.01, 0.03, 9.80]
```

如果量化范围为了照顾 `9.80` 拉得很大，前面那些小数会被压得很粗糙。很多小误差加起来，方向和点积就可能变歪。

TurboQuant 的第一步就是：先把向量随机旋转，让信息不要集中在少数坐标上。

## 9. 随机旋转：先把能量摊平

随机旋转用一个矩阵表示，记作：

$$
\mathbf{\Pi}
$$

旋转后的向量是：

$$
\mathbf{z} = \mathbf{\Pi}\mathbf{x}
$$

这里的 `Π` 是正交矩阵。你可以把它理解成高维空间里的“旋转操作”。

正交矩阵最重要的性质是：

$$
\|\mathbf{\Pi}\mathbf{x}\|_2 = \|\mathbf{x}\|_2
$$

也就是说，旋转不会改变长度。

它也不会改变两个向量的点积：

$$
\langle \mathbf{\Pi}\mathbf{x}, \mathbf{\Pi}\mathbf{y} \rangle
=
\langle \mathbf{x}, \mathbf{y} \rangle
$$

所以随机旋转不会凭空破坏几何关系。它只是换了一个观察角度。

为什么这有用？

因为对任何固定的单位向量 `x`，乘上随机旋转后，`Πx` 会像“单位球面上随机取到的一个点”。论文用到的事实是：它的每个坐标服从下面这个分布：

$$
f_X(t) =
\frac{\Gamma(d/2)}
{\sqrt{\pi}\Gamma((d-1)/2)}
(1 - t^2)^{(d-3)/2},
\quad t \in [-1, 1]
$$

这个公式看起来吓人，但它只说一件事：

> 随机旋转后，每个坐标的大小会变得很可预测。高维时，每个坐标大概像均值为 0、方差为 `1/d` 的正态分布。

写成近似就是：

$$
(\mathbf{\Pi}\mathbf{x})_j \approx \mathcal{N}(0, 1/d)
$$

普通解释：

- 原来某些坐标可能特别大。
- 随机旋转后，能量被摊到很多坐标上。
- 每个坐标大概都在 `1/sqrt(d)` 这个量级。
- 这样逐坐标量化就变得合理了。

## 10. TurboQuant_mse：第一版算法

TurboQuant 的 MSE 版本目标是：

> 让解压后的向量尽量接近原向量。

它叫：

$$
Q_{\text{mse}}
$$

流程：

```text
输入向量 x
  -> 随机旋转 z = Πx
  -> 每个坐标 z_j 找最近的 codebook 中心 c_k
  -> 存每个坐标的中心编号 idx_j
  -> 解码时把 idx_j 换回 c_k
  -> 反向旋转 x_hat = Π^T z_hat
```

这里的 codebook 是一组代表值。

如果 bit-width 是 `b`，每个坐标有：

$$
2^b
$$

个可选中心。

例如 `b=2`，每个坐标可以选 4 个中心。论文给出的高维近似中心是：

$$
\left\{
-\frac{1.51}{\sqrt d},
-\frac{0.453}{\sqrt d},
\frac{0.453}{\sqrt d},
\frac{1.51}{\sqrt d}
\right\}
$$

`b=1` 时，只有两个中心：

$$
\left\{
-\sqrt{\frac{2}{\pi d}},
\sqrt{\frac{2}{\pi d}}
\right\}
$$

![PolarQuant直觉](/learning/assets/turboquant-polar.svg)

这张图用二维箭头解释“方向 + 长度”的直觉。但 TurboQuant_mse 并不存二维角度；它随机旋转后，对每个坐标做最优标量量化。

## 11. codebook 是怎么来的：一维 k-means

现在我们把 codebook 说清楚。

随机旋转后，每个坐标的分布是 `f_X(t)`。我们要用 `2^b` 个中心去表示这个一维随机变量。

假设中心是：

$$
c_1, c_2, \ldots, c_{2^b}
$$

每个真实值 `t` 会被分到最近的中心。

中心之间的边界是相邻中心的中点：

$$
\frac{c_i + c_{i+1}}{2}
$$

所以最优 codebook 的目标是：

$$
\mathcal{C}(f_X, b)
=
\min_{c_1,\ldots,c_{2^b}}
\sum_i
\int_{\text{第 i 个区间}}
|t - c_i|^2 f_X(t)\,dt
$$

普通解释：

- 横轴是一个坐标可能出现的数值。
- `f_X(t)` 表示这个数值出现的概率。
- `|t - c_i|^2` 表示把 `t` 近似成中心 `c_i` 的损失。
- 积分就是把所有可能的 `t` 按概率加权平均。
- 最小化这个值，就是找最好的代表点。

这就是连续版的一维 k-means，也叫 Lloyd-Max 量化。

论文预先算好常用 bit-width 的 codebook，实际运行时直接查表，并不会每次量化都重新算。

## 12. 为什么总 MSE 等于 d 乘一维误差

解码后：

$$
\tilde{\mathbf{x}} = \mathbf{\Pi}^\top \tilde{\mathbf{z}}
$$

因为旋转不改变长度：

$$
\|\mathbf{x} - \tilde{\mathbf{x}}\|_2
=
\|\mathbf{\Pi}\mathbf{x} - \tilde{\mathbf{z}}\|_2
=
\|\mathbf{z} - \tilde{\mathbf{z}}\|_2
$$

平方后：

$$
\|\mathbf{z} - \tilde{\mathbf{z}}\|_2^2
=
\sum_{j=1}^{d}(z_j - \tilde z_j)^2
$$

每个坐标的分布一样，所以期望误差一样：

$$
D_{\text{mse}}
=
d \cdot \mathcal{C}(f_X,b)
$$

这一步很关键。它说明：高维向量量化被随机旋转后，可以拆成很多个一维量化问题。

这里不要被外面的 `d` 吓到。随机旋转后，单个坐标 `z_j` 的方差大约是 `1/d`，所以一维误差 `\mathcal{C}(f_X,b)` 自己也带着一个大约 `1/d` 的缩放。外面的 `d` 和里面的 `1/d` 会抵消。因此总 MSE 不会因为维度变大就线性暴涨；在论文的单位向量设定下，主导误差主要由 bit-width `b` 决定。

论文证明的上界是：

$$
D_{\text{mse}}
\le
\frac{\sqrt{3}\pi}{2}\cdot \frac{1}{4^b}
$$

这里最重要的是 `1/4^b`。

因为：

- `b` 增加 1，每个坐标多 1 bit。
- `4^b` 会乘 4。
- 误差大约按 4 倍速度下降。

论文还给了低 bit 的更精细数值：

| bit-width b | MSE 近似上界 |
| --- | --- |
| 1 | 0.36 |
| 2 | 0.117 |
| 3 | 0.03 |
| 4 | 0.009 |

## 13. TurboQuant_mse 的伪代码

论文里的 MSE 版算法可以写成这样：

```text
Setup(d, b):
    生成随机旋转矩阵 Π
    预计算 2^b 个 codebook 中心 c_1 ... c_{2^b}

Quant_mse(x):
    z = Πx
    对每个坐标 z_j:
        idx_j = 离 z_j 最近的中心编号
    返回 idx

DeQuant_mse(idx):
    对每个坐标 j:
        z_hat_j = c_{idx_j}
    x_hat = Π^T z_hat
    返回 x_hat
```

对应 NumPy 教学版如下。

注意：这不是生产级代码。论文使用的是 Beta 分布对应的最优 codebook；下面为了可读性，用高维下的正态近似 `N(0, 1/d)` 来用 Lloyd-Max 数值求 codebook。生产系统还需要 bit packing、CUDA kernel、快速旋转、norm 存储和 outlier channel 处理。

代码里用 `np.trapz` 做数值积分，是为了兼容 NumPy 1.x。NumPy 2.0 之后也可以用等价的 `np.trapezoid`。

```python
import numpy as np


def random_rotation(d, rng):
    """生成一个 d x d 的随机正交矩阵。教学版用 QR 分解。"""
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)

    # 修正 QR 分解的符号，让随机性更接近 Haar random rotation。
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def lloyd_max_normal(bit_width, d, steps=200, grid_size=20001):
    """为 N(0, 1/d) 求一维 Lloyd-Max codebook。"""
    k = 2 ** bit_width
    if k == 1:
        return np.array([0.0], dtype=np.float64)

    sigma = 1.0 / np.sqrt(d)
    grid = np.linspace(-6 * sigma, 6 * sigma, grid_size)
    pdf = np.exp(-0.5 * (grid / sigma) ** 2)
    pdf = pdf / np.trapz(pdf, grid)

    centers = np.linspace(-2.5 * sigma, 2.5 * sigma, k)

    for _ in range(steps):
        boundaries = (centers[:-1] + centers[1:]) / 2
        new_centers = centers.copy()

        left_edges = np.r_[-np.inf, boundaries]
        right_edges = np.r_[boundaries, np.inf]

        for i, (left, right) in enumerate(zip(left_edges, right_edges)):
            mask = (grid >= left) & (grid < right)
            mass = np.trapz(pdf[mask], grid[mask])
            if mass > 1e-12:
                new_centers[i] = np.trapz(
                    grid[mask] * pdf[mask],
                    grid[mask],
                ) / mass

        if np.max(np.abs(new_centers - centers)) < 1e-12:
            break
        centers = new_centers

    return centers


class TurboQuantMSE:
    def __init__(self, d, bit_width, rng):
        self.d = d
        self.bit_width = bit_width
        self.rotation = random_rotation(d, rng)
        self.codebook = lloyd_max_normal(bit_width, d)

    def quant(self, x):
        z = self.rotation @ x
        distance = np.abs(z[:, None] - self.codebook[None, :])
        idx = np.argmin(distance, axis=1)
        return idx.astype(np.uint16)

    def dequant(self, idx):
        z_hat = self.codebook[idx]
        x_hat = self.rotation.T @ z_hat
        return x_hat
```

逐行对应：

- `random_rotation` 对应论文里的 `Π`。
- `z = self.rotation @ x` 对应 `z = Πx`。
- `self.codebook` 对应中心 `c_1 ... c_{2^b}`。
- `idx = argmin(...)` 对应找最近中心。
- `z_hat = self.codebook[idx]` 对应从编号恢复中心。
- `rotation.T @ z_hat` 对应 `Π^T z_hat`，也就是反向旋转。

## 14. 但是 MSE 最优不等于内积最优

到这里，`TurboQuant_mse` 已经能把向量压小，并且重建误差不错。

但如果我们的目标是点积，它还有一个问题：**有偏**。

无偏是什么意思？

假设真实点积是：

$$
\langle \mathbf{y}, \mathbf{x} \rangle
$$

压缩后估计的是：

$$
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
$$

如果多次随机量化后，平均值等于真实值，就叫无偏：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
\right]
=
\langle \mathbf{y}, \mathbf{x} \rangle
$$

如果平均值总是偏小或偏大，就叫有偏。

论文举了 `b=1` 的例子。高维下，MSE 最优的两个中心是：

$$
\pm \sqrt{\frac{2}{\pi d}}
$$

于是：

$$
Q_{\text{mse}}^{-1}(Q_{\text{mse}}(\mathbf{x}))
=
\sqrt{\frac{2}{\pi d}}\,
\mathbf{\Pi}^\top
\operatorname{sign}(\mathbf{\Pi}\mathbf{x})
$$

此时它的内积期望大约是：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
\right]
=
\frac{2}{\pi}
\langle \mathbf{y}, \mathbf{x} \rangle
$$

`2/pi ≈ 0.637`。

也就是说，如果真实点积是 `1.0`，它平均估计成 `0.637`。这是系统性偏小，不是随机噪声。

这就是为什么 TurboQuant 还需要第二版算法：`TurboQuant_prod`。

## 15. QJL：用 1 bit 做无偏内积估计

QJL 全称是 Quantized Johnson-Lindenstrauss。

名字很吓人，但算法本身可以先这样看：

```text
随机投影 -> 只保留正负号 -> 用特殊缩放还原成一个估计向量
```

给定一个向量 `u`，QJL 先生成一个随机矩阵：

$$
\mathbf{S}_{ij} \sim \mathcal{N}(0, 1)
$$

然后只存：

$$
\operatorname{sign}(\mathbf{S}\mathbf{u})
$$

也就是每个投影结果是正还是负。

这就是 1 bit：

```text
正 -> +1
负 -> -1
```

QJL 的反量化是：

$$
Q_{\text{qjl}}^{-1}(\mathbf{s})
=
\frac{\sqrt{\pi/2}}{d}
\mathbf{S}^\top \mathbf{s}
$$

如果 `u` 是单位向量，则 QJL 有两个关键性质：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{u})) \rangle
\right]
=
\langle \mathbf{y}, \mathbf{u} \rangle
$$

这就是无偏。

它的方差上界是：

$$
\operatorname{Var}
\left(
\langle \mathbf{y}, Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{u})) \rangle
\right)
\le
\frac{\pi}{2d}\|\mathbf{y}\|_2^2
$$

普通解释：

- QJL 单次估计会有噪声。
- 但平均起来是对的。
- 维度 `d` 越大，方差越小。
- 所以高维反而帮了忙。

再强调一次：上面这个无偏公式的前提是 `u` 是单位向量。对于非单位向量 `v`，要先写成：

$$
\mathbf{v} = \gamma \mathbf{u},
\quad
\gamma = \|\mathbf{v}\|_2,
\quad
\|\mathbf{u}\|_2 = 1
$$

然后对 `u` 做 QJL，并在反量化时乘回 `\gamma`。如果直接忘掉这个范数缩放，就不能把 QJL 理解成“任意向量都能直接还原”的方法。

![QJL残差修正直觉](/learning/assets/turboquant-qjl.svg)

## 16. TurboQuant_prod：主量化 + QJL 残差

`TurboQuant_prod` 目标是让内积估计无偏、低误差，而不追求让 `x_hat` 本身最像 `x`。

它把总 bit-width `b` 拆成两部分：

- 前 `b-1` bit：用 `TurboQuant_mse` 做主量化。
- 最后 `1` bit：用 QJL 量化残差。

先做 MSE 主量化：

$$
\tilde{\mathbf{x}}_{\text{mse}}
=
Q_{\text{mse}}^{-1}(Q_{\text{mse}}(\mathbf{x}))
$$

残差是：

$$
\mathbf{r}
=
\mathbf{x}
-
\tilde{\mathbf{x}}_{\text{mse}}
$$

也就是：

```text
残差 = 原始向量 - 主量化恢复出来的向量
```

然后对残差做 QJL：

$$
\operatorname{sign}(\mathbf{S}\mathbf{r})
$$

还要额外保存残差长度：

$$
\gamma = \|\mathbf{r}\|_2
$$

这一步就是在处理“残差不是单位向量”的问题。严格地说，我们是把残差写成：

$$
\mathbf{r} = \gamma \mathbf{u},
\quad
\mathbf{u} = \mathbf{r} / \gamma
$$

因为 `\gamma > 0` 时：

$$
\operatorname{sign}(\mathbf{S}\mathbf{r})
=
\operatorname{sign}(\gamma\mathbf{S}\mathbf{u})
=
\operatorname{sign}(\mathbf{S}\mathbf{u})
$$

所以符号草图可以照常存，但解码时必须乘回 `\gamma`。如果 `\gamma = 0`，说明残差本来就是零，QJL 项也应该是零。

最终解码：

$$
\tilde{\mathbf{x}}
=
\tilde{\mathbf{x}}_{\text{mse}}
+
\frac{\sqrt{\pi/2}}{d}
\gamma
\mathbf{S}^\top
\operatorname{sign}(\mathbf{S}\mathbf{r})
$$

这就是论文里的 `TurboQuant_prod`。

它不完整恢复残差，只用 1 bit 草图帮忙修正内积。

## 17. 为什么这样能无偏

这一段是整篇文章最关键的数学。

我们想证明：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
\right]
=
\langle \mathbf{y}, \mathbf{x} \rangle
$$

先把最终估计拆成两部分：

$$
\tilde{\mathbf{x}}
=
\tilde{\mathbf{x}}_{\text{mse}}
+
\tilde{\mathbf{x}}_{\text{qjl}}
$$

所以：

$$
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
=
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{mse}} \rangle
+
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{qjl}} \rangle
$$

严格地说，QJL 的无偏性先作用在单位残差方向：

$$
\mathbf{u} = \mathbf{r}/\gamma
$$

反量化时乘回 `\gamma`，于是：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, \gamma Q_{\text{qjl}}^{-1}(Q_{\text{qjl}}(\mathbf{u})) \rangle
\mid
\tilde{\mathbf{x}}_{\text{mse}}
\right]
=
\gamma \langle \mathbf{y}, \mathbf{u} \rangle
=
\langle \mathbf{y}, \mathbf{r} \rangle
$$

这正是 `TurboQuant_prod` 里保存并乘回残差长度 `\gamma = \|\mathbf{r}\|_2` 的原因。少了这个缩放，残差非单位向量会破坏无偏性。

因此我们可以写：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{qjl}} \rangle
\mid
\tilde{\mathbf{x}}_{\text{mse}}
\right]
=
\langle \mathbf{y}, \mathbf{r} \rangle
$$

把它代回去：

$$
\mathbb{E}
\left[
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
\mid
\tilde{\mathbf{x}}_{\text{mse}}
\right]
=
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{mse}} \rangle
+
\langle \mathbf{y}, \mathbf{r} \rangle
$$

而残差定义是：

$$
\mathbf{r}
=
\mathbf{x}
-
\tilde{\mathbf{x}}_{\text{mse}}
$$

所以：

$$
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{mse}} \rangle
+
\langle \mathbf{y}, \mathbf{x} - \tilde{\mathbf{x}}_{\text{mse}} \rangle
=
\langle \mathbf{y}, \mathbf{x} \rangle
$$

这就是无偏。

普通话总结：

> 主量化可能有偏，但它留下了残差。QJL 不需要完整恢复残差，只要在内积意义上平均能补回残差，整体点积就无偏。

## 18. 为什么误差也小

QJL 的方差上界告诉我们：

$$
\text{残差修正造成的内积方差}
\le
\frac{\pi}{2d}
\|\mathbf{r}\|_2^2
\|\mathbf{y}\|_2^2
$$

这里最重要的是：

$$
\|\mathbf{r}\|_2^2
$$

残差越小，QJL 修正越稳。

而 `TurboQuant_mse` 正好负责让残差尽量小：

$$
\mathbb{E}\|\mathbf{r}\|_2^2
=
D_{\text{mse}}
$$

所以：

$$
D_{\text{prod}}
\le
\frac{\pi}{2d}
\|\mathbf{y}\|_2^2
D_{\text{mse}}(b-1)
$$

再把 MSE 上界代进去，得到论文里的内积误差上界：

$$
D_{\text{prod}}
\le
\frac{\sqrt{3}\pi^2\|\mathbf{y}\|_2^2}{d}
\cdot
\frac{1}{4^b}
$$

这就是 TurboQuant 的组合逻辑：

1. 用 `b-1` bit 把残差压小。
2. 用最后 1 bit 的 QJL 保证内积无偏。
3. 残差越小，QJL 的噪声越小。

## 19. TurboQuant_prod 的伪代码

```text
Setup(d, b):
    初始化 TurboQuant_mse(d, b-1)
    生成随机投影矩阵 S

Quant_prod(x):
    idx = Quant_mse(x)
    x_mse = DeQuant_mse(idx)
    r = x - x_mse
    qjl = sign(Sr)
    gamma = ||r||
    返回 idx, qjl, gamma

DeQuant_prod(idx, qjl, gamma):
    x_mse = DeQuant_mse(idx)
    x_qjl = sqrt(pi/2) / d * gamma * S^T qjl
    返回 x_mse + x_qjl
```

对应 NumPy 教学版：

```python
def sign_pm1(v):
    return np.where(v >= 0, 1.0, -1.0)


class TurboQuantProd:
    def __init__(self, d, bit_width, rng):
        assert bit_width >= 1
        self.d = d
        self.bit_width = bit_width
        self.mse = TurboQuantMSE(d, bit_width - 1, rng)
        self.proj = rng.normal(size=(d, d))

    def quant(self, x):
        idx = self.mse.quant(x)
        x_mse = self.mse.dequant(idx)

        residual = x - x_mse
        gamma = np.linalg.norm(residual)

        if gamma == 0:
            qjl = np.ones(self.d, dtype=np.float64)
        else:
            qjl = sign_pm1(self.proj @ residual)

        return idx, qjl, gamma

    def dequant(self, idx, qjl, gamma):
        x_mse = self.mse.dequant(idx)
        x_qjl = np.sqrt(np.pi / 2.0) / self.d * gamma * (self.proj.T @ qjl)
        return x_mse + x_qjl
```

逐行对应：

- `self.mse = TurboQuantMSE(d, bit_width - 1, rng)` 对应先用 `b-1` bit 做主量化。
- `residual = x - x_mse` 对应残差 `r = x - x_mse`。
- `gamma = np.linalg.norm(residual)` 对应保存 `||r||`。
- `qjl = sign_pm1(self.proj @ residual)` 对应 `sign(Sr)`。
- `sqrt(pi / 2) / d * gamma * S.T @ qjl` 对应 QJL 反量化。
- `return x_mse + x_qjl` 对应主量化结果加残差修正。

## 20. 一个完整小实验

下面这段代码可以测试 `TurboQuant_mse` 和 `TurboQuant_prod` 在内积估计上的区别。

```python
def unit(v):
    return v / np.linalg.norm(v)


def demo():
    d = 128
    bit_width = 2

    base_rng = np.random.default_rng(123)
    x = unit(base_rng.normal(size=d))
    y = unit(base_rng.normal(size=d))
    true_ip = float(y @ x)

    mse_estimates = []
    prod_estimates = []

    # 多换几次随机旋转和随机投影，看平均值。
    for seed in range(200):
        rng = np.random.default_rng(seed)

        tq_mse = TurboQuantMSE(d, bit_width, rng)
        idx = tq_mse.quant(x)
        x_mse = tq_mse.dequant(idx)
        mse_estimates.append(float(y @ x_mse))

        rng = np.random.default_rng(seed)
        tq_prod = TurboQuantProd(d, bit_width, rng)
        idx, qjl, gamma = tq_prod.quant(x)
        x_prod = tq_prod.dequant(idx, qjl, gamma)
        prod_estimates.append(float(y @ x_prod))

    print("真实内积:", true_ip)
    print("MSE 量化平均估计:", np.mean(mse_estimates))
    print("Prod 量化平均估计:", np.mean(prod_estimates))
    print("MSE 量化估计方差:", np.var(mse_estimates))
    print("Prod 量化估计方差:", np.var(prod_estimates))


demo()
```

你应该关注平均值。

理论上：

- `TurboQuant_mse` 追求重建误差小，不保证内积平均值完全正确。
- `TurboQuant_prod` 通过 QJL 残差修正，保证内积估计无偏。

实际实验会有有限采样误差，但 `prod` 的平均值应该更接近真实内积。

## 21. 代码和论文实现差在哪里

上面的代码是教学版，不是论文或生产版实现。主要差别有：

| 部分 | 教学版 | 论文/工程实现 |
| --- | --- | --- |
| 随机旋转 | QR 生成 dense rotation | 可用更适合硬件的旋转实现 |
| codebook | 用正态近似数值求 | 用 Beta 分布的 Lloyd-Max 最优 codebook，预先存表 |
| 存储 | `uint16` 数组，没有真 bit packing | 真实按 bit 压紧存储 |
| QJL 矩阵 | 显式存 `d x d` Gaussian 矩阵 | 工程上会考虑计算和存储优化 |
| norm | 简单存 `gamma` 浮点数 | 真实系统要计入额外存储开销 |
| KV cache | 只演示单个向量 | 真实要按 layer/head/token/channel 批量处理 |
| outlier | 没有处理异常通道 | 论文实验会给 outlier channel 分配不同 bit |

所以这段代码的价值是帮你理解算法结构，不是直接拿去部署。

## 22. 从算法到 CUDA kernel：先看数据布局

前面的 NumPy 代码是“数学长什么样”。放到 GPU 上，第一件事是决定数据怎么摆。

KV cache 通常是这种形状：

```text
[layer, head, token, channel]
```

对某一层、某一个 head、某一个 token 来说，`channel` 方向上的一整行就是一个向量：

```text
x = KV[layer, head, token, :]
```

假设 head_dim 是 `d = 128`，原本用 float16 存：

```text
128 channels * 16 bit = 2048 bit = 256 bytes
```

如果用 2-bit idx 存主量化结果：

```text
128 channels * 2 bit = 256 bit = 32 bytes
```

只看 idx，体积是原来的 `1/8`。真实系统还要额外存 norm、scale、QJL sign、outlier channel 等元数据，所以最终压缩比不会这么理想，但方向是对的。

![TurboQuant kernel数据布局](/learning/assets/turboquant-kernel-layout.svg)

一个比较自然的 kernel 映射是：

```text
一个 CUDA block 处理一个向量
blockIdx.x = 第几个向量
threadIdx.x = 负责若干 channel 或若干 packed word
```

为什么一个 block 处理一个向量？

因为一个向量的维度通常不大，比如 128、256。一个 block 里的线程可以一起处理这些 channel，最后把结果打包写回全局内存。

但是要先讲清楚一个重要限制：

> 论文里的随机旋转 `Πx` 是算法定义。工程热路径里，如果真的每来一个 KV 向量都做 dense `d x d` 矩阵乘法，代价会很重。

所以 kernel 实现常见会拆成两种讲法：

1. **数学直译版**：先算好 `z = Πx`，然后量化 `z`。这最容易理解。
2. **工程优化版**：用结构化随机旋转、融合上游算子、或者把旋转和量化放在同一个 kernel 里，减少 HBM 读写。

下面的代码主要讲数学直译版和 kernel 设计模式。它不是直接可部署的完整 TurboQuant 库。

## 23. Kernel 0：2-bit 打包和解包

先从最基础的 bit packing 讲。

如果 `b = 2`，每个 channel 的量化编号只有 4 种：

```text
0, 1, 2, 3
```

每个编号只要 2 bit。一个 `uint32_t` 有 32 bit，所以可以装：

```text
32 / 2 = 16
```

个 channel 的编号。

![2-bit打包](/learning/assets/turboquant-bitpacking.svg)

第 `j` 个 channel 存在哪里？

```text
word_id = j / 16
slot    = j % 16
shift   = 2 * slot
```

写入：

```text
word |= idx << shift
```

读取：

```text
idx = (word >> shift) & 0b11
```

CUDA helper 可以写成：

```cpp
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t get_2bit_idx(
    const uint32_t* packed,
    int channel
) {
    int word_id = channel >> 4;          // channel / 16
    int slot = channel & 15;             // channel % 16
    int shift = slot * 2;
    return (packed[word_id] >> shift) & 3u;
}

__device__ __forceinline__ void set_2bit_idx_in_word(
    uint32_t& word,
    int slot,
    uint32_t idx
) {
    word |= (idx & 3u) << (slot * 2);
}
```

这里没有用 atomic，因为更好的方式是：**让一个线程负责一个 packed word**。这个线程顺序处理 16 个 channel，最后一次性写出一个 `uint32_t`。这样比多个线程抢着写同一个 word 更简单。

## 24. Kernel 1：`Quant_mse` 的 2-bit 教学实现

这一节实现：

```text
z = Πx 已经算好
对每个 z_j 找最近 codebook 中心
把 idx_j 用 2-bit 打包写出
```

![Quant kernel线程映射](/learning/assets/turboquant-quant-kernel.svg)

为了让代码短一点，我们先只写 `b = 2`。这时 codebook 有 4 个中心：

```text
c0, c1, c2, c3
```

实际系统会把 codebook 放在 constant memory 或 shared memory。因为 codebook 很小，每个线程都会反复读它。

```cpp
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

// b = 2 时有 4 个中心。真实实现会按 head_dim、bit_width 选择不同表。
__constant__ float kCodebook2[4];

__global__ void quant_mse_2bit_kernel(
    const half* __restrict__ rotated,  // [num_vecs, d]，这里假设已经是 z = Πx
    uint32_t* __restrict__ packed,     // [num_vecs, ceil(d / 16)]
    int d,
    int words_per_vec
) {
    int vec = blockIdx.x;

    // 一个线程负责若干个 packed word。
    for (int word_id = threadIdx.x; word_id < words_per_vec; word_id += blockDim.x) {
        uint32_t out = 0;
        int base_ch = word_id * 16;

        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            int ch = base_ch + slot;
            if (ch >= d) {
                continue;
            }

            float z = __half2float(rotated[vec * d + ch]);

            int best = 0;
            float diff0 = z - kCodebook2[0];
            float best_dist = diff0 * diff0;

            #pragma unroll
            for (int k = 1; k < 4; ++k) {
                float diff = z - kCodebook2[k];
                float dist = diff * diff;
                if (dist < best_dist) {
                    best_dist = dist;
                    best = k;
                }
            }

            out |= (static_cast<uint32_t>(best) << (2 * slot));
        }

        packed[vec * words_per_vec + word_id] = out;
    }
}
```

这段 kernel 对应前面的数学：

$$
\text{idx}_j
=
\arg\min_k (z_j - c_k)^2
$$

因为这里是一维标量，比较 `|z_j - c_k|` 和比较 `(z_j - c_k)^2` 会选出同一个最近中心；代码里写平方误差，是为了和前面的 MSE 准则在概念上保持一致。

代码里的对应关系：

| 数学 | CUDA |
| --- | --- |
| `z_j` | `rotated[vec * d + ch]` |
| `c_k` | `kCodebook2[k]` |
| `argmin` | `for (int k = 1; k < 4; ++k)` |
| `idx_j` | `best` |
| bit-packed idx | `out |= best << (2 * slot)` |

真实工程里还要加：

- 对 `b = 3`、`b = 4` 的不同 packing。
- 对 outlier channel 的不同 bit-width。
- 对 scale/norm 的写出。
- 旋转和量化的融合。
- 更适合硬件的内存对齐和 vectorized load/store。

## 25. Kernel 2：不要先完整解压，再算 attention

如果我们先把 packed KV cache 全部解压回 float16 大矩阵，再做 attention，会浪费很多带宽。

更好的思路是：

```text
从 HBM 读取 packed word
  -> 在寄存器里 unpack 出 idx
  -> 查 codebook 得到近似值
  -> 立刻和 query channel 相乘
  -> 在线程块内 reduction 得到 q·K 分数
```

也就是边解码，边计算。

![融合解码attention](/learning/assets/turboquant-fused-attention.svg)

还有一个细节：如果 Key 存的是旋转后的 `z = Πk` 的量化结果，那么 Query 也要旋转：

$$
\langle \mathbf{q}, \mathbf{k} \rangle
=
\langle \mathbf{\Pi}\mathbf{q}, \mathbf{\Pi}\mathbf{k} \rangle
$$

所以融合打分 kernel 里应该用旋转后的 query：

```text
q_rot = Πq
```

教学版 kernel 如下。它只演示 MSE 主量化部分，不含 QJL 残差项。

下面的版本为了清楚，内层循环直接读 `q_rot[ch]`。生产级 attention kernel 通常不会让每个 token block 都反复从 global memory 读同一个 query。更常见的做法是：block 启动时先让线程协作把 `q_rot` 搬到 shared memory，或者把 query tile 放进寄存器并通过 warp 内广播复用。这样可以明显减少对 HBM/L2 的重复读压力。

```cpp
__global__ void score_packed_keys_2bit_kernel(
    const half* __restrict__ q_rot,       // [d]，当前 query 已经旋转
    const uint32_t* __restrict__ packed_k,// [num_tokens, words_per_vec]
    float* __restrict__ scores,           // [num_tokens]
    int d,
    int words_per_vec,
    float inv_sqrt_d
) {
    extern __shared__ float smem[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    float acc = 0.0f;

    const uint32_t* token_words = packed_k + token * words_per_vec;

    for (int word_id = tid; word_id < words_per_vec; word_id += blockDim.x) {
        uint32_t word = token_words[word_id];
        int base_ch = word_id * 16;

        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            int ch = base_ch + slot;
            if (ch >= d) {
                continue;
            }

            uint32_t idx = (word >> (2 * slot)) & 3u;
            float k_approx = kCodebook2[idx];
            float q = __half2float(q_rot[ch]);
            acc += q * k_approx;
        }
    }

    smem[tid] = acc;
    __syncthreads();

    // 简单 block reduction。真实实现会用 warp-level primitive 优化。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scores[token] = smem[0] * inv_sqrt_d;
    }
}
```

这个规约也有教学简化：它默认 `blockDim.x` 是 2 的幂。真实 kernel 要么处理任意线程数，要么干脆使用 warp shuffle primitive，例如 `__shfl_down_sync`。常见写法是先在每个 warp 内用寄存器 shuffle 完成求和，再让每个 warp 的 lane 0 写一次 shared memory，最后做一次跨 warp 规约。这样可以少很多 `__syncthreads()`。

这个 kernel 做的是：

$$
\langle \mathbf{q}_{rot}, \tilde{\mathbf{z}} \rangle
$$

因为：

$$
\tilde{\mathbf{k}}
=
\mathbf{\Pi}^{T}\tilde{\mathbf{z}}
$$

所以：

$$
\langle \mathbf{q}, \tilde{\mathbf{k}} \rangle
=
\langle \mathbf{q}, \mathbf{\Pi}^{T}\tilde{\mathbf{z}} \rangle
=
\langle \mathbf{\Pi}\mathbf{q}, \tilde{\mathbf{z}} \rangle
$$

这就是为什么我们不用真的先算出 `k_hat = Π^T z_hat`。只要把 query 旋转到同一个坐标系，直接和 `z_hat` 做点积就行。

这一步非常重要，因为它省掉了大量解压写回。

## 26. Kernel 3：QJL 残差项怎么融合

`TurboQuant_prod` 的最终估计是：

$$
\tilde{\mathbf{x}}
=
\tilde{\mathbf{x}}_{\text{mse}}
+
\tilde{\mathbf{x}}_{\text{qjl}}
$$

所以点积也可以拆开：

$$
\langle \mathbf{y}, \tilde{\mathbf{x}} \rangle
=
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{mse}} \rangle
+
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{qjl}} \rangle
$$

MSE 主量化部分可以用上一节的 packed idx 查 codebook 来算。

QJL 残差部分根据公式是：

$$
\tilde{\mathbf{x}}_{\text{qjl}}
=
\frac{\sqrt{\pi/2}}{d}
\gamma
\mathbf{S}^{T}
\mathbf{s}
$$

其中：

- `γ = ||r||_2`
- `s = sign(Sr)`，也就是 packed QJL signs

如果直接按公式算点积：

$$
\langle \mathbf{y}, \tilde{\mathbf{x}}_{\text{qjl}} \rangle
=
\frac{\sqrt{\pi/2}}{d}
\gamma
\langle \mathbf{y}, \mathbf{S}^{T}\mathbf{s} \rangle
$$

利用内积转置关系：

$$
\langle \mathbf{y}, \mathbf{S}^{T}\mathbf{s} \rangle
=
\langle \mathbf{S}\mathbf{y}, \mathbf{s} \rangle
$$

这说明工程上可以这样做：

```text
先为当前 query 算 u = S y
然后对每个历史 token:
    读取 packed sign s
    qjl_score = dot(u, s)
    乘上 sqrt(pi/2) / d * gamma
```

这样不用为每个 token 都显式恢复 `S^T s`。

教学版代码可以写成：

```cpp
__global__ void add_qjl_scores_kernel(
    const half* __restrict__ sy,          // [d]，提前算好的 S*y
    const uint32_t* __restrict__ signs,   // [num_tokens, ceil(d / 32)]，1 bit sign
    const float* __restrict__ gamma,      // [num_tokens]
    float* __restrict__ scores,           // [num_tokens]，累加到主量化 score 上
    int d,
    int sign_words_per_vec
) {
    extern __shared__ float smem[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    float acc = 0.0f;

    const uint32_t* token_signs = signs + token * sign_words_per_vec;

    for (int word_id = tid; word_id < sign_words_per_vec; word_id += blockDim.x) {
        uint32_t word = token_signs[word_id];
        int base_ch = word_id * 32;

        #pragma unroll
        for (int slot = 0; slot < 32; ++slot) {
            int ch = base_ch + slot;
            if (ch >= d) {
                continue;
            }

            // bit 1 表示 +1，bit 0 表示 -1。
            float s = ((word >> slot) & 1u) ? 1.0f : -1.0f;
            acc += __half2float(sy[ch]) * s;
        }
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float scale = sqrtf(CUDART_PI_F * 0.5f) / static_cast<float>(d);
        scores[token] += scale * gamma[token] * smem[0];
    }
}
```

这段代码对应的是：

$$
\frac{\sqrt{\pi/2}}{d}
\gamma
\langle \mathbf{S}\mathbf{y}, \mathbf{s} \rangle
$$

不过要注意：如果 `S` 是普通 dense Gaussian 矩阵，提前算 `Sy` 本身也要 `O(d^2)`，这在热路径里很贵。生产实现需要考虑：

- 用结构化随机投影降低 `S*y` 的代价。
- 把 QJL 修正只用于最需要的路径。
- 和 attention kernel 融合，减少中间数组。
- 在质量和吞吐之间选择是否启用完整 `TurboQuant_prod`。

这里的“结构化随机投影”是关键。JL/QJL 在数学上常用 dense Gaussian 矩阵，因为它好分析；但工程上更希望用随机正负号对角矩阵和 Hadamard 变换这类结构来近似随机投影。

![结构化随机投影](/learning/assets/turboquant-structured-projection.svg)

一个典型形式可以写成：

$$
\mathbf{S}
\approx
\mathbf{D}_1\mathbf{H}\mathbf{D}_2\mathbf{H}\mathbf{D}_3
$$

这里：

- `D_i` 是随机正负号对角矩阵，只需要给每个 channel 乘 `+1` 或 `-1`。
- `H` 是 Hadamard 矩阵，可以用 Fast Walsh-Hadamard Transform, FWHT, 的蝴蝶结构快速计算。

这样原本 dense `S*y` 的 `O(d^2)` 乘法，可以降到 `O(d log d)`。如果 `d` 是 128 或 256，这个差别在热路径里非常实在。更进一步，结构化投影还更容易和量化、解码、attention 打分融合到同一个 kernel 里。

这属于工程实现路线，不等于说论文中每个定理都可以无条件把 dense Gaussian `S` 换成任意结构化矩阵。数学保证和工程实现之间通常还要补额外分析或实验验证。但从性能角度看，没有结构化随机投影，`TurboQuant_prod` 的 QJL 项很难成为真正轻量的热路径组件。

## 27. outlier channel 和 2.5-bit / 3.5-bit 是怎么来的

论文实验里提到 2.5-bit 和 3.5-bit。它们是把 channel 分组，而不是某个单独 channel 用了半个 bit。

例如某些模型配置的 head_dim 是 `d = 128`，具体数值要以模型 config 为准。如果 32 个 outlier channel 用 3 bit，其余 96 个普通 channel 用 2 bit，平均 bit 是：

```text
32 个 outlier channel 用 3 bit
96 个普通 channel 用 2 bit
平均 bit = (32 * 3 + 96 * 2) / 128 = 2.25
```

注意这个组合的平均值是 `2.25`，不是 `2.5`。所以看到非整数 bit-width 时，最好按通道数自己复核一遍。要得到正好 2.5-bit，可以换不同分配，比如：

```text
64 个 channel 用 3 bit
64 个 channel 用 2 bit
平均 bit = (64 * 3 + 64 * 2) / 128 = 2.5
```

论文正文的核心意思是：把 outlier 和 regular channel 拆开，用两个独立 TurboQuant 实例，给 outlier 更高 bit。无论具体比例怎么选，核心思想都是：

> 不同 channel 的重要性不一样。特别容易出大值、对质量更敏感的 channel，可以多给一点 bit。

kernel 层面上，这会让实现复杂一些：

```text
普通 channel -> packed_2bit
outlier channel -> packed_3bit 或 packed_4bit
metadata -> 记录哪些 channel 是 outlier
```

一个简单策略是把 outlier channel 重排到前面，让 kernel 读写连续内存：

```text
[outlier channels][regular channels]
```

这样比在 kernel 里频繁查一个稀疏 channel list 更容易优化。

## 28. 一个更像真实路径的 kernel 流水线

把上面几段合起来，在线 KV cache 量化大概会长这样：

```text
生成当前 token 的 K/V
  -> 对 K/V 做旋转或等价变换
  -> Quant_mse kernel:
       找最近 codebook
       写 packed idx
       统计 residual / norm
  -> QJL kernel:
       对 residual 写 packed sign
       写 gamma
  -> 后续 token 做 attention:
       旋转 query
       从 packed K cache 边解码边算 q·K
       加上 QJL residual score
       softmax
       对 packed V cache 做类似 fused dequant + weighted sum
```

工程上最应该避免的是这条慢路径：

```text
packed cache
  -> 全量解压成 float16 K/V 矩阵
  -> 写回 HBM
  -> 再从 HBM 读出来做 attention
```

这样会把量化省下来的带宽又浪费回去。

更理想的是：

```text
packed cache
  -> 寄存器里 unpack
  -> 查 codebook
  -> 直接参与 dot / weighted sum
```

这就是 kernel 实现的核心：

> 量化算法解决“存什么”，kernel 解决“不要把存下来的东西反复搬来搬去”。

## 29. TurboQuant 用在 KV cache 时怎么想

Transformer 注意力的简化形式是：

$$
\text{Attn}(\mathbf{q}, \mathbf{K}, \mathbf{V})
=
\text{softmax}(\mathbf{q}\mathbf{K}^\top)\mathbf{V}
$$

不用怕这个公式。它分两步：

1. `qK^T`：当前 query 和历史 key 做点积，得到每个历史 token 的重要性分数。
2. `softmax(...)V`：根据重要性分数，把历史 value 加权平均。

KV cache 里存的就是很多 `K` 和 `V` 向量。

如果把 Key 量化坏了，`qK^T` 的分数会错，模型会看错上下文位置。

如果把 Value 量化坏了，即使位置看对了，拿回来的信息也会变脏。

TurboQuant 的意义在于：

- 对 Key：尽量保住 Query-Key 点积。
- 对 Value：尽量保住向量信息和后续线性计算。
- 对长上下文：显著减少缓存体积。

论文报告说，在 KV cache 量化实验里，3.5 bits per channel 可以达到接近无质量损失，2.5 bits per channel 只有轻微质量下降。具体效果当然依赖模型、任务和实现。

## 30. 和 PolarQuant、QJL 的关系

你可以把三者关系理解成：

| 名字 | 解决的问题 | 在 TurboQuant 里的位置 |
| --- | --- | --- |
| PolarQuant | 从长度和方向角度理解/量化向量 | 提供直觉和相关背景 |
| QJL | 1 bit 下做无偏内积估计 | 用来量化残差 |
| TurboQuant_mse | 用随机旋转 + 标量最优量化降低 MSE | 第一阶段主量化 |
| TurboQuant_prod | MSE 主量化 + QJL 残差修正 | 面向内积的完整版本 |

TurboQuant 的核心数学是下面这几步：

1. 随机旋转后坐标服从可分析分布。
2. 用 Lloyd-Max 做每个坐标的最优标量量化。
3. MSE 版有接近最优的重建误差。
4. 内积版用 QJL 修残差，解决 MSE 量化的偏差。

## 31. 为什么说它接近最优

论文还证明了信息论下界。

任何量化算法，只要每个坐标只给 `b` bit，都不可能无限好。对于单位球上的最坏情况输入，MSE 至少有：

$$
D_{\text{mse}}
\ge
\frac{1}{4^b}
$$

内积误差也至少有：

$$
D_{\text{prod}}
\ge
\frac{\|\mathbf{y}\|_2^2}{d}
\cdot
\frac{1}{4^b}
$$

TurboQuant_mse 的上界是：

$$
D_{\text{mse}}
\le
\frac{\sqrt{3}\pi}{2}
\cdot
\frac{1}{4^b}
$$

常数：

$$
\frac{\sqrt{3}\pi}{2}
\approx 2.7
$$

也就是说，它和理论下界只差一个常数倍，而不是差一个数量级。

这就是论文标题里 “near-optimal distortion rate” 的意思。

## 32. 从零复述一遍算法

第一步，向量太大，直接存很贵。

第二步，AI 里我们常常不需要完整恢复向量，只需要距离、点积、相似度别错太多。

第三步，直接逐坐标量化容易被异常坐标影响，所以先随机旋转，把能量摊平。

第四步，随机旋转后，每个坐标分布变得可预测，高维下接近 `N(0, 1/d)`。

第五步，对这个一维分布求最优 codebook。每个坐标只存最近中心的编号，这就是 `TurboQuant_mse`。

第六步，`TurboQuant_mse` 重建误差小，但用于内积可能有偏，尤其低 bit 下很明显。

第七步，用 `b-1` bit 先做 MSE 主量化，得到一个主近似。

第八步，计算残差 `r = x - x_mse`。

第九步，对残差做 1 bit QJL，只存随机投影的正负号和残差长度。

第十步，解码时返回：

$$
\tilde x_{\text{mse}} + \tilde x_{\text{qjl}}
$$

这样得到的 `TurboQuant_prod` 对内积是无偏的，误差也接近理论最优。

## 33. 常见误解

### 误解一：TurboQuant 是无损压缩

不是。它是有损压缩。

它厉害的地方是让丢掉的信息尽量不影响最终任务。

### 误解二：MSE 小就一定适合注意力

不一定。

注意力大量依赖 Query-Key 点积。如果点积估计有系统性偏差，MSE 小也可能不够。`TurboQuant_prod` 专门处理这个问题。

### 误解三：QJL 能恢复残差

不能。

QJL 只保存正负号草图，它不能完整恢复残差。它的目标是内积无偏估计。

### 误解四：教学代码就是生产实现

不是。

教学代码为了让公式和代码一一对应，故意牺牲了性能。真正部署要处理 bit packing、GPU kernel、batch layout、KV cache 内存布局、outlier channel、norm 开销等工程问题。

## 34. 建议阅读路线

如果你是第一次读这方向，建议顺序是：

1. 先理解本文的第 1 到 6 节：向量、量化、点积、MSE、内积误差。
2. 再理解第 9 到 13 节：随机旋转和 `TurboQuant_mse`。
3. 再理解第 14 到 19 节：为什么有偏，QJL 如何修正。
4. 如果关心实现，再读第 22 到 28 节：packed cache、量化 kernel、融合 attention 打分和 QJL 残差项。
5. 最后去看原论文的定理和实验。

不要一开始就硬啃所有公式。先知道每个公式在解决什么问题，再回头看证明。

## Reference

- [Google TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Google QJL arxiv paper](https://arxiv.org/abs/2406.03482)
- [Google PolarQuant paper](https://arxiv.org/abs/2502.02617)
- [Google TurboQuant arxiv paper](https://arxiv.org/abs/2504.19874)
