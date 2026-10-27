
# CANO_GatedRope

## 1. 文件与用途

模型文件：

```text
models/CANO_GatedRope.py
```

该模型同时保留：

1. 原始 CANO 内容 logits；
2. 节点级连续三维 RoPE logits；
3. 每层、每个 head 独立的可学习门控。

输入输出：

```text
data:        (B, N, 3)  min-max 归一化坐标
rope_coords: (B, N, 3)  原始物理坐标
output:      (B, N, 1)  归一化压力预测
```

这是三个 CANO 版本中最适合正式训练和稳定消融的版本，因为初始化时可以非常接近普通 CANO。

Q、K、V 使用三个完全独立的直接投影：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

不存在共享的 `in_project_x`，不同 head 对应各自独立的投影输出通道。

## 2. 两条 attention 路径

Cluster assignment 由未旋转 Query 产生：

\[
C=\operatorname{Softmax}_{K_c}(QW_c).
\]

普通 CANO cluster query：

\[
Q_c=C^\top Q.
\]

节点级 RoPE 后的 cluster query：

\[
\widetilde Q_c=C^\top R(p_i)Q_i.
\]

完整 Key 的 RoPE 形式为：

\[
\widetilde K_j=R(p_j)K_j.
\]

因此两组 logits 分别为：

\[
S_{base}=\frac{Q_cK^\top}{\sqrt d},
\]

\[
S_{rope}=\frac{\widetilde Q_c\widetilde K^\top}{\sqrt d}.
\]

## 3. 门控融合

每层、每个 attention head 都有一个可学习参数 `α`：

\[
g=\operatorname{sigmoid}(\alpha),
\qquad 0<g<1.
\]

最终 logits 为：

\[
S=(1-g)S_{base}+gS_{rope}.
\]

然后执行：

\[
Z_c=\operatorname{Softmax}(S)V,
\qquad
Y=CZ_c.
\]

默认：

```text
initial_rope_gate = 0.05
```

也就是模型初始化时约为：

```text
95% 普通 CANO logits + 5% RoPE logits
```

相对位置分支从第一个训练步骤开始就能获得梯度，但不会立即主导模型。

## 4. 不显式保存两份 logits

直接计算 `S_base` 和 `S_rope` 会生成两份 `(B,H,Kc,N)` 矩阵。

当前实现利用特征拼接：

\[
Q_*=
\left[
\sqrt{1-g}Q_c,
\sqrt g\widetilde Q_c
\right],
\]

\[
K_*=
\left[
\sqrt{1-g}K,
\sqrt g\widetilde K
\right].
\]

则：

\[
Q_*K_*^\top
=(1-g)Q_cK^\top
+g\widetilde Q_c\widetilde K^\top.
\]

因为两组特征位于拼接后的不同通道区间，不会产生交叉项。

核心代码：

```python
combined_q = torch.cat(
    (
        torch.sqrt(1.0 - gate) * clustered_q,
        torch.sqrt(gate) * clustered_q_rope,
    ),
    dim=-1,
)

combined_k = torch.cat(
    (
        torch.sqrt(1.0 - gate) * k,
        torch.sqrt(gate) * k_rope,
    ),
    dim=-1,
)

cluster_features = F.scaled_dot_product_attention(
    combined_q,
    combined_k,
    v,
    scale=self.scale,
)
```

这里显式传入原始 `key_dim**-0.5`，避免 PyTorch 因拼接后维度翻倍而使用错误的缩放系数。

## 5. 为什么融合 logits 而不是输出

如果分别计算：

```text
Y_base = softmax(S_base) V
Y_rope = softmax(S_rope) V
```

再融合输出，那么两条路径会分别形成独立的概率分布。

当前方法在 softmax 之前融合 logits：

```text
softmax((1-g) S_base + g S_rope)
```

这意味着相对位置直接参与“应该关注哪些节点”的竞争，更符合 RoPE 修改 QK 相关性的原始目的。

## 6. 门控粒度

门控形状：

```text
(1, H, 1, 1)
```

因此：

- 每一层有独立门控；
- 每个 attention head 有独立门控；
- 同一个 head 内的所有 cluster 和节点共享门控。

这比单个全局标量更灵活，同时不会引入 `N` 相关的额外参数。

训练后查看门控：

```python
print(model.rope_gate_values())
```

返回形状：

```text
(n_layers, n_head)
```

## 7. 如何解释门控结果

如果门控接近 0：

```text
该层/head 更依赖普通内容相关性。
```

如果门控明显增大：

```text
该层/head 从相对位置关系中获得了训练收益。
```

如果所有层迅速接近 1：

```text
模型倾向于纯 RoPE CANO，需要检查是否真正提高验证集误差。
```

如果训练损失震荡且门控快速变化，可以降低初始门控或学习率。

## 8. 复杂度与显存

虽然 Q/K 的通道在 SDPA 内临时拼接为两倍，但 attention 矩阵仍然为：

```text
B × H × Kc × N
```

不存在 `N × N` attention。

复杂度仍然为：

\[
O(NK_cd),
\]

但常数开销高于普通 CANO 和纯 CANO_Rope，因为需要同时保存普通与旋转后的 Q/K 特征。

## 9. main.py 调用

模型名：

```yaml
model_name: CANO_GatedRope
```

构造方式：

```python
model = CANO_GatedRope(
    space_dim=3,
    n_layers=8,
    n_hidden=256,
    n_head=8,
    mlp_ratio=2,
    fun_dim=0,
    out_dim=1,
    key_ratio=1,
    cluster_num=args.cluster_num,
    num_position_frequencies=args.num_position_frequencies,
    initial_rope_gate=args.initial_rope_gate,
)
```

`ml_SATO.py` 中调用：

```python
model(coorf[..., :3], rope_coords=raw_coords)
```

该调用只在模型名为 `CANO_GatedRope` 时生效，不会改变其他模型的 forward 参数。

## 10. 推荐配置

```yaml
model_name: CANO_GatedRope
phase: restart_train
space_dim: 3

cluster_num: 64
num_position_frequencies: 6
initial_rope_gate: 0.05

batch_size: 1
```

建议消融：

```text
cluster_num:              32, 64, 128
num_position_frequencies: 3, 6, 9
initial_rope_gate:        0.01, 0.05, 0.10
```

优先固定 `cluster_num=64`，先比较三个门控初始值；确定门控设置后，再改变 cluster 数量。

## 11. 三个模型的实验关系

| 模型 | Attention logits | 目的 |
| --- | --- | --- |
| `CANO` | 只有普通内容 logits | 基线 |
| `CANO_Rope` | 只有 RoPE logits | 验证强位置注入 |
| `CANO_GatedRope` | 普通与 RoPE logits 门控融合 | 稳定学习位置贡献 |

为了保证结论有效，三个模型必须使用完全相同的数据划分、归一化、网络宽度、层数、cluster 数和训练设置。

## 12. Checkpoint 注意事项

三个模型参数结构不同：

- CANO_Rope 多出可学习三维频率；
- CANO_GatedRope 还多出每层、每个 head 的门控；
- 三者不能相互直接加载完整 checkpoint。
- 早期共享前置 Q/K/V 投影版本也不能加载到当前独立投影结构。

首次训练应设置：

```yaml
phase: restart_train
```

