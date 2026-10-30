# LinearNO_se

## 1. 模型目标

`LinearNO_se.py` 在普通 LinearNO 的全局线性 attention 主干上，增加了基于空间序列化的局部 attention 分支。

模型希望同时学习：

- 普通 LinearNO 提供的全局流场相关性；
- 空间上相邻节点之间的局部几何和压力变化；
- 不规则点云在不同空间遍历顺序下的邻域关系。

当前 ShapeNet-Car 压力预测的数据接口为：

```text
coorf:   (B, N, 3)     min-max 归一化坐标，输入模型编码器
order:   (B, 4, N)     四种空间序列化顺序
inverse: (B, 4, N)     四种序列化顺序对应的逆排列
output:  (B, N, 1)     归一化压力预测
```

原始坐标不直接输入 `LinearNO_se.forward()`，而是在 `ml_SATO.py` 中用于生成 `order` 和 `inverse`。

## 2. 为什么需要空间序列化

汽车表面网格通常以任意节点编号保存。原始数组中相邻的两个索引，不一定代表几何空间中相邻的两个节点。

如果直接把点云数组切成固定长度 patch，那么一个 patch 内可能包含来自车头、车尾和底盘的无关节点，局部 attention 就失去了几何意义。

空间序列化先根据三维坐标把节点转换为一维空间遍历顺序，使一维序列中邻近的节点大概率也在三维空间中邻近。当前代码使用四种顺序：

```python
["z", "z-trans", "hilbert", "hilbert-trans"]
```

对应：

- Morton/Z-order；
- 转置后的 Morton/Z-order；
- Hilbert order；
- 转置后的 Hilbert order。

不同顺序会保留不同方向和尺度的邻域关系，因此各层循环使用不同的序列化顺序。

## 3. order 和 inverse 的含义

设原始节点特征为：

```text
fx: (B, N, C)
```

`order` 表示按照空间曲线重新读取原始节点的索引：

```python
serialized = torch.gather(fx, dim=1, index=gather_order)
```

如果一个简单样本的：

```text
order = [2, 0, 3, 1]
```

那么序列化后的节点顺序为：

```text
[fx[2], fx[0], fx[3], fx[1]]
```

`inverse` 用于把局部 attention 输出恢复到原网格节点顺序：

```python
local_out = torch.gather(local_out, dim=1, index=gather_inverse)
```

必须满足：

```python
torch.gather(order, dim=2, index=inverse) == arange(N)
```

模型只改变临时计算顺序，最终输出节点仍和原 VTK/PKL 网格节点一一对应，因此预测结果可以映射回原始网格。

## 4. 序列化索引的生成

训练、验证和测试阶段都通过以下函数生成索引：

```python
pos_to_order_inverse_index(coords, device, tensor=True)
```

当前 `ml_SATO.py` 中只有序列化模型会执行该步骤：

```python
if model_name in SERIALIZED_MODELS:
    order, inverse = pos_to_order_inverse_index(...)
```

其中 `LinearNO_se` 特别使用原始物理坐标：

```python
if model_name == "LinearNO_se":
    return raw_coords
```

坐标和特征的用途因此被明确分开：

| 数据 | 用途 |
| --- | --- |
| 归一化坐标 `coorf` | 输入 `preprocess`，生成网络特征 |
| 原始坐标 `raw_coords` | 生成 Morton/Hilbert 排序索引 |

序列化函数会对 batch 内每个样本独立处理，输出：

```text
order.shape   = (B, 4, N)
inverse.shape = (B, 4, N)
```

因此当前实现支持 `batch_size > 1`，前提是同一 batch 中的样本能够被 dataloader 整理为相同节点数。

## 5. 普通 LinearNO 全局分支

每个 block 首先执行与普通 LinearNO 相同的全局残差路径：

```python
fx = fx + self.global_attention(self.ln_1(fx))
```

LinearNO 的核心计算为：

\[
T=K^{\mathrm T}V,
\qquad
Y=QT.
\]

代码为：

```python
kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
qkv = torch.einsum("bhnd,bhdc->bhnc", q, kv)
```

该分支负责全局信息传递，并保持普通 LinearNO 的线性复杂度优势。

## 6. 序列化局部分支

全局 attention 更新后，特征按照当前 block 选择的空间顺序重新排列：

```python
serialized = torch.gather(self.local_ln_1(fx), 1, gather_order)
```

然后进入 `Serialized_Attention`：

```python
local_out = self.local_attention(serialized)
```

序列会被切分为长度为 `patch_size` 的局部 patch，并在每个 patch 内执行普通的 QKV attention：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^{\mathrm T}}{\sqrt{d}}
\right)V.
\]

局部 attention 的矩阵大小为：

```text
patch_size × patch_size
```

而不是整个点云的 `N × N`。

局部计算完成后，通过 `inverse` 恢复原始节点顺序：

```python
local_out = torch.gather(local_out, 1, gather_inverse)
```

## 7. patch_size 和 shift

`patch_size` 表示每个局部 attention patch 包含的节点数量。

`shift` 用于构造交错子序列。例如：

```text
patch_size = 4
shift = 2
```

一个长度为 8 的序列会形成两组交错索引：

```text
第 1 组：[0, 2, 4, 6]
第 2 组：[1, 3, 5, 7]
```

这样可以让局部 attention 不只作用于严格连续的序列点，还能连接一定间隔内的节点。

每个完整索引块的长度为：

\[
\text{block_size}=\text{patch_size}\times\text{shift}.
\]

如果节点数不能整除 `block_size`，代码使用：

```python
pad_size = (-N) % block_size
```

补齐到最近的完整块。padding 节点会通过 key mask 从 attention 权重中排除，不会作为有效 key 影响真实节点：

```python
dots = dots.masked_fill(~key_mask, torch.finfo(dots.dtype).min)
```

## 8. 局部门控

局部输出不会直接覆盖普通 LinearNO，而是作为额外残差加入：

\[
g=\operatorname{sigmoid}(\alpha),
\qquad
X\leftarrow X+gY_{\mathrm{local}}.
\]

代码为：

```python
local_gate = torch.sigmoid(self.local_gate_logit)
fx = fx + local_gate * local_out
```

当前门控初始值为：

```text
g = 0.05
```

这样初始化时仍以普通 LinearNO 为主，但局部分支从第一个优化步骤开始就能获得梯度。

每个 block 有一个独立的标量门控。训练后可以检查：

```python
print(model.local_gate_values())
```

如果浅层门控较大，通常表示模型需要较早提取局部几何关系；如果深层门控较大，则可能表示全局信息聚合后仍需要局部修正。

## 9. 一个完整 block 的计算流程

当前 `LinearNO_block.forward()` 的顺序为：

```text
输入 fx
  │
  ├─ LayerNorm
  ├─ Global LinearNO Attention
  ├─ Global 残差连接
  │
  ├─ 使用 order 进行空间序列化
  ├─ Local Patch Attention
  ├─ 使用 inverse 恢复节点顺序
  ├─ Sigmoid 门控局部残差
  │
  ├─ LayerNorm
  ├─ MLP
  ├─ MLP 残差连接
  │
  └─ 最后一层映射到压力 output
```

对应核心代码：

```python
fx = fx + self.global_attention(self.ln_1(fx))

serialized = torch.gather(self.local_ln_1(fx), 1, gather_order)
local_out = self.local_ln_2(self.local_attention(serialized))
local_out = torch.gather(local_out, 1, gather_inverse)
fx = fx + torch.sigmoid(self.local_gate_logit) * local_out

fx = fx + self.mlp(self.ln_2(fx))
```

最重要的一点是：全局分支和 MLP 分支保留了普通 LinearNO 的结构，序列化 attention 是额外增加的局部修正，而不是替代原模型。

## 10. 各层如何选择序列化顺序

模型包含四种序列化顺序，每层通过以下方式循环选择：

```python
serialization_type = i % order.shape[1]
fx = block(
    fx,
    order[:, serialization_type, :],
    inverse[:, serialization_type, :],
)
```

当模型为 8 层时，顺序使用方式为：

```text
第 1、5 层：z
第 2、6 层：z-trans
第 3、7 层：hilbert
第 4、8 层：hilbert-trans
```

这样可以减小单一空间曲线产生的方向偏置。

## 11. 计算复杂度

设节点数为 `N`、patch 大小为 `P`：

- 全局 LinearNO attention 对 `N` 是线性复杂度；
- 序列化排序通常约为 `O(N log N)`；
- 局部 attention 共有约 `N/P` 个 patch，每个 patch 的 attention 为 `O(P²)`；
- 因此局部 attention 总体约为 `O(NP)`。

当 `P` 为固定小常数时，局部 attention 对节点数仍近似线性增长。显存中也只保存局部 `P × P` attention，而不是完整 `N × N` attention。

`patch_size` 的取值存在权衡：

| 设置 | 优点 | 风险 |
| --- | --- | --- |
| 较小 | 显存低，更强调局部细节 | 感受野小，patch 边界影响更明显 |
| 较大 | 能学习更宽的局部关联 | 计算量和显存按 `P²` 增长 |

## 12. main.py 中的配置

模型创建方式：

```python
model = LinearNO_se(
    space_dim=3,
    n_layers=8,
    n_hidden=256,
    n_head=8,
    mlp_ratio=2,
    fun_dim=0,
    out_dim=1,
    key_ratio=1,
    patch_size=args.patch_size,
    shift=args.shift,
)
```

推荐的基础配置：

```yaml
model_name: LinearNO_se
phase: restart_train
space_dim: 3

patch_size: 20
shift: 2

batch_size: 1
```

建议先固定其他训练条件，仅对以下组合进行消融实验：

```text
patch_size: 10, 20, 40
shift:      1, 2
```

必须与普通 LinearNO 使用相同的数据划分、随机种子、epoch、学习率和误差计算方式，结果才具有可比性。

## 13. 与 SATO 和普通 LinearNO 的关系

普通 LinearNO：

```text
Global Linear Attention → MLP
```

LinearNO_se：

```text
Global Linear Attention
       +
Serialized Local Attention（门控残差）
       ↓
      MLP
```

它借鉴了 SATO 使用空间序列化增强局部几何学习的思想，但全局主干仍是 LinearNO，而不是 Transolver/Physics-Attention。

## 14. 使用注意事项

1. `order`、`inverse`、输入特征和目标压力的节点顺序必须来自同一个网格样本。
2. 不能只排序输入而不执行逆排序，否则输出无法与原网格节点对应。
3. 验证和测试阶段必须针对当前 batch 重新生成排序索引，不能沿用训练阶段或上一个 batch 的索引。
4. `order` 和 `inverse` 必须为整数张量，并且形状为 `(B, 4, N)`。
5. 修改 `patch_size`、`shift` 或模型结构后，建议使用 `phase: restart_train`。
6. 结构修复前生成的旧 `LinearNO_se` checkpoint 可能与当前模型不兼容，不应直接用于当前结构的效果比较。

