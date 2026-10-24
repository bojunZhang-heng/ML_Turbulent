# RopeLinearNO_v3

## 1. 模型目标

`RopeLinearNO_v3.py` 在普通 LinearNO 的基础上加入了一个可学习、非负、平移不变的相对位置核，用于增强点云中不同节点之间的空间关系。

该模型当前面向 ShapeNet-Car 表面压强预测，输入为三维坐标：

```text
data:        (B, N, 3)  经过 min-max 归一化的坐标，用于特征编码
rope_coords: (B, N, 3)  原始物理坐标，用于构造相对位置
output:      (B, N, 1)  归一化后的表面压强预测
```

这里保留普通 LinearNO 的内容注意力作为主分支，相对位置分支通过一个较小的可学习门控逐渐参与训练。因此模型初始化时接近普通 LinearNO，而不是直接用位置分支替换原模型。

## 2. 普通 LinearNO 主分支

输入特征首先被投影到多个 attention head：

```python
x_mid = rearrange(
    self.in_project_x(x),
    "b n (h d) -> b h n d",
    h=self.heads,
    d=self.dim_head,
)
q = self.to_q(x_mid)
k = self.to_k(x_mid)
v = self.to_v(x_mid)
```

其张量形状为：

```text
q, k: (B, H, N, K)
v:    (B, H, N, D)
```

其中：

- `B`：batch size；
- `H`：attention head 数量；
- `N`：表面网格节点数；
- `K = key_ratio * dim_head`：Q、K 的特征维度；
- `D = dim_head`：每个 head 的 value 维度。

LinearNO 没有显式计算 `N × N` 的注意力矩阵，而是分别执行：

```python
q = softmax(q / temperature_q, dim=-1)
k = softmax(k / temperature_k, dim=-2)

kv = einsum("bhnd,bhnc->bhdc", k, v)
base_out = einsum("bhnd,bhdc->bhnc", q, kv)
```

对应公式为：

\[
T=K^{\mathrm T}V,
\qquad
Y_{\mathrm{base}}=QT.
\]

将乘法顺序从 `(QKᵀ)V` 改成 `Q(KᵀV)` 后，不再需要保存 `N × N` 矩阵。与节点数有关的主要复杂度由二次复杂度降为线性复杂度。

## 3. 相对位置坐标处理

模型的特征编码仍然使用经过 min-max 归一化的 `data`：

```python
fx = self.preprocess(coords)
fx = fx + self.placeholder[None, None, :]
```

相对位置核使用 `rope_coords`。在计算位置相位前，每个样本内部会先转换为无量纲坐标：

\[
\hat p_i=
\frac{p_i-p_{\mathrm{center}}}
{\lVert p_{\max}-p_{\min}\rVert_2}.
\]

对应代码位于 `PositiveRelativePositionKernel.normalize_coordinates()`。

这样处理具有三个作用：

1. 平移整个汽车不会改变相对位置核；
2. 不同汽车整体尺度不同时，相位大小仍处于相近范围；
3. 网络输入和位置核可以分别使用最合适的坐标表示。

因此当前模型中的坐标分工是：

| 坐标 | 用途 | 处理方式 |
| --- | --- | --- |
| `data` | 输入 MLP、生成节点内容特征 | 数据预处理阶段 min-max 归一化 |
| `rope_coords` | 生成相对位置相位 | 原始坐标传入后，模型内部无量纲化 |

## 4. 可学习的非负相对位置核

对第 `h` 个 head、第 `r` 个频率，位置相位为：

\[
\phi_{i,h,r}=\omega_{h,r}^{\mathrm T}\hat p_i,
\]

其中 `ω` 是可学习频率参数，形状为：

```text
(H, R, 3)
```

`R` 为 `num_position_frequencies`。

节点 `i` 和节点 `j` 之间的相对位置核为：

\[
\kappa_{ij}^{(h)}=
\frac{1}{2}+
\frac{1}{2R}
\sum_{r=1}^{R}
\cos\left(\phi_{i,h,r}-\phi_{j,h,r}\right).
\]

因为每个余弦项都属于 `[-1, 1]`，所以：

\[
0\leq\kappa_{ij}^{(h)}\leq1.
\]

该核具有以下性质：

- 非负，不会产生负的 attention 权重；
- 只依赖 `p_i-p_j`，因此具有平移不变性；
- 频率可以学习，使网络自行寻找有效的空间尺度和方向；
- 不需要固定的绝对正余弦位置编码。

## 5. 如何避免构造 N × N 相对位置矩阵

理想的相对位置 attention 可以写为：

\[
Y_i^{\mathrm{rel}}=
\frac{
\sum_j (q_i^{\mathrm T}k_j)\kappa_{ij}v_j
}{
\sum_j(q_i^{\mathrm T}k_j)\kappa_{ij}
}.
\]

如果直接计算 `κ_ij`，仍然会产生 `N × N` 矩阵。当前实现利用恒等式：

\[
\cos(a-b)=\cos(a)\cos(b)+\sin(a)\sin(b),
\]

把每个频率拆成 cosine 和 sine 两个可分离项。例如 cosine 项可以按照以下顺序计算：

```text
1. 用 cos(phi_j) 对 K_j 加权
2. 先聚合 Σ_j [K_j cos(phi_j)] V_j
3. 再与 Q_i 相乘
4. 最后乘 cos(phi_i)
```

sine 项采用相同方式。这样就把两点之间的相对位置作用隐式地加入 `KᵀV` 聚合过程，而不需要显式保存所有点对。

核心实现位于：

```python
LinearNO._positive_relative_attention()
```

该函数同时计算 numerator 和 denominator，最后执行归一化：

```python
relative_out = numerator / denominator.clamp_min(eps).unsqueeze(-1)
```

`clamp_min(eps)` 用于避免数值除零。

## 6. 门控融合

普通 LinearNO 输出和相对位置输出通过门控融合：

\[
g=\operatorname{sigmoid}(\alpha),
\]

\[
Y=(1-g)Y_{\mathrm{base}}+gY_{\mathrm{rel}}.
\]

代码形式为：

```python
gate = torch.sigmoid(self.relative_gate_logit)
output = base_out + gate * (relative_out - base_out)
```

默认 `initial_relative_gate=0.05`，因此初始化时：

```text
95% 普通 LinearNO 输出 + 5% 相对位置输出
```

门控按 head 独立学习，形状为 `(1, H, 1, 1)`。这样不同 head 可以自行决定需要多少相对位置信息。

训练后可以检查各层门控：

```python
print(model.relative_gate_values())
```

如果门控长期接近 0，说明当前数据和训练设置下位置分支没有提供额外收益；如果不同层、不同 head 出现不同门控值，则说明网络正在选择性使用空间信息。

## 7. 一个完整 block 的计算顺序

每个 `LinearNOBlock` 的计算流程为：

```text
输入 fx
  │
  ├─ LayerNorm
  ├─ 普通 LinearNO + 相对位置核
  ├─ 残差连接
  │
  ├─ LayerNorm
  ├─ MLP
  ├─ 残差连接
  │
  └─ 最后一层映射到压力 output
```

对应代码：

```python
fx = self.attn(self.ln_1(fx), position_phases=position_phases) + fx
fx = self.mlp(self.ln_2(fx)) + fx
```

位置相位只修改 attention，不替换输入编码器和 MLP 主干。

## 8. 复杂度与显存

设频率数为 `R`、Q/K 维度为 `K`、value 维度为 `D`，相对位置分支的主要复杂度近似为：

\[
O(NRKD).
\]

它对节点数 `N` 仍是线性的，但相对于普通 LinearNO，大约增加 `2R` 次 cosine/sine 聚合，因此 `R` 不宜设置过大。

模型训练时还采用分段 gradient checkpoint：

```python
self.checkpoint_segment_size = 2
```

每两个 block 组成一个 checkpoint segment，以计算时间换取较低的激活显存。验证和测试阶段不会启用 checkpoint。

## 9. main.py 中的调用

模型创建方式：

```python
model = RopeLinearNO_v3(
    space_dim=3,
    n_layers=8,
    n_hidden=256,
    n_head=8,
    mlp_ratio=2,
    fun_dim=0,
    out_dim=1,
    key_ratio=1,
    num_position_frequencies=args.num_position_frequencies,
    initial_relative_gate=args.initial_relative_gate,
)
```

`ml_SATO.py` 中的实际 forward 调用为：

```python
model(coorf[..., :3], rope_coords=raw_coords)
```

其中 `coorf` 是归一化输入坐标，`raw_coords` 是相同节点顺序下的原始坐标。两者必须满足：

```text
coorf.shape == raw_coords.shape == (B, N, 3)
```

## 10. 推荐配置

```yaml
model_name: RopeLinearNO_v3
phase: restart_train
space_dim: 3

num_position_frequencies: 6
initial_relative_gate: 0.05

batch_size: 1
```

建议首先测试：

```text
num_position_frequencies: 3, 6, 9
initial_relative_gate:    0.01, 0.05, 0.10
```

每次改变结构或位置频率数量后，应使用 `restart_train` 重新训练，不应继续加载结构不兼容的旧 checkpoint。

## 11. 与传统 RoPE 的区别

传统 Transformer 中的 RoPE 作用于 Q、K，并通过显式的 `QKᵀ` 把相对位置写入 `N × N` attention logits。

RopeLinearNO_v3 不生成 `N × N` logits。它使用可分离的 Fourier 相对位置核，将点对位置关系展开为 cosine/sine 聚合，并嵌入 `Q(KᵀV)` 的线性 attention 过程。

因此该模型更准确的描述是：

> 带有可学习非负 Fourier 相对位置核的 LinearNO，而不是直接把传统 RoPE 原样放进 LinearNO。

