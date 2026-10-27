
# CANO

## 1. 文件与用途

模型文件：

```text
models/CANO.py
```

该文件实现 Cluster Attention Neural Operator（CANO）的 coordinate-only 版本，当前用于 ShapeNet-Car 表面压力预测。

输入输出：

```text
输入 data: (B, N, 3)  min-max 归一化后的 XYZ 坐标
输出:      (B, N, 1)  mean-std 归一化后的压力预测
```

模型保持所有 Key 和 Value 节点的完整分辨率，只把 Query 动态聚合为较少的 cluster query。

## 2. 核心思想

普通 self-attention 需要计算：

\[
\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}\right)V,
\]

其中 `QKᵀ` 的形状是 `N × N`。

CANO 首先根据 Query 生成节点到 cluster 的软分配矩阵：

\[
C=\operatorname{Softmax}_{K_c}(QW_c),
\qquad
C\in\mathbb{R}^{N\times K_c}.
\]

然后只聚合 Query：

\[
Q_c=C^\top Q,
\qquad
Q_c\in\mathbb{R}^{K_c\times d}.
\]

聚类后的 Query 对全分辨率 Key/Value 执行 cross-attention：

\[
Z_c=
\operatorname{Softmax}
\left(
\frac{Q_cK^\top}{\sqrt d}
\right)V.
\]

最后使用相同的分配矩阵广播回原始节点：

\[
Y=CZ_c.
\]

## 3. 张量形状

设：

```text
B  = batch size
N  = 节点数
H  = attention head 数量
Kc = cluster_num
K  = key_ratio * dim_head
D  = dim_head
```

核心张量为：

| 张量 | 形状 |
| --- | --- |
| `q`, `k` | `(B, H, N, K)` |
| `v` | `(B, H, N, D)` |
| `cluster_assignment` | `(B, H, N, Kc)` |
| `clustered_q` | `(B, H, Kc, K)` |
| cluster-to-node attention | `(B, H, Kc, N)` |
| `cluster_features` | `(B, H, Kc, D)` |
| 广播后的输出 | `(B, H, N, D)` |

每个 head 独立产生 cluster assignment，不共享聚类权重。

## 4. 核心代码流程

Query、Key、Value 投影：

```python
q, k, v = self._project_qkv(x)
```

三者直接从完整隐藏特征通过三个完全独立的线性层产生：

```python
self.to_q = nn.Linear(dim, heads * key_dim, bias=False)
self.to_k = nn.Linear(dim, heads * key_dim, bias=False)
self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
```

对应：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

不存在共享的前置 `in_project_x`。每个投影输出后才拆分为多个 head，因此不同 head 也占用各自独立的权重通道。

动态聚类并聚合 Query：

```python
cluster_logits = torch.einsum(
    "bhnd,hdk->bhnk", q, self.cluster_projection
)
cluster_assignment = F.softmax(cluster_logits, dim=-1)
clustered_q = torch.einsum(
    "bhnk,bhnd->bhkd", cluster_assignment, q
)
```

cluster query 对完整 K/V 做 cross-attention：

```python
cluster_features = F.scaled_dot_product_attention(
    clustered_q,
    k,
    v,
    scale=self.scale,
)
```

广播回所有节点：

```python
output = torch.einsum(
    "bhnk,bhkd->bhnd", cluster_assignment, cluster_features
)
```

## 5. 与 Transolver 的区别

Transolver 通常把物理状态压缩成 slice token，并在压缩后的 token 空间内构造 Q、K、V。

CANO 只压缩 Query：

```text
Transolver：压缩后的 Q / K / V 相互作用
CANO：      压缩 Q，完整 K / V 保持不变
```

因此 CANO 的 cluster query 仍然能够直接访问每个原始节点的 Key 和 Value，有利于保留局部高频信息。

## 6. Block 结构

每个 block 保持和 `RopeLinearNO_v3.py` 一致的 Pre-LayerNorm 残差形式：

```python
fx = self.attn(self.ln_1(fx)) + fx
fx = self.mlp(self.ln_2(fx)) + fx
```

最后一个 block 使用：

```python
self.mlp2(self.ln_3(fx))
```

将隐藏特征映射为单通道压力。

## 7. 复杂度

Cross-attention 的主要复杂度为：

\[
O(NK_cd).
\]

当 `Kc` 固定且远小于 `N` 时，对节点数 `N` 为线性复杂度。

但模型仍需要保存以下两个与 `N × Kc` 成正比的张量：

```text
cluster_assignment: B × H × N × Kc
attention:          B × H × Kc × N
```

因此 `cluster_num` 对显存影响明显，不建议一开始设置过大。

## 8. Gradient Checkpoint

训练阶段默认每两个 block 使用一次 gradient checkpoint：

```python
checkpoint_segment_size = 2
```

这会减少激活显存，但增加一部分反向重计算时间。验证和测试阶段不启用 checkpoint。

## 9. main.py 调用

模型名：

```yaml
model_name: CANO
```

`main.py` 中的构造参数：

```python
model = CANO(
    space_dim=3,
    n_layers=8,
    n_hidden=256,
    n_head=8,
    mlp_ratio=2,
    fun_dim=0,
    out_dim=1,
    key_ratio=1,
    cluster_num=args.cluster_num,
)
```

`ml_SATO.py` 中调用：

```python
model(coorf)
```

普通 CANO 不使用 `raw_coords`。

## 10. 推荐配置

```yaml
model_name: CANO
phase: restart_train
space_dim: 3
cluster_num: 64
batch_size: 1
```

建议消融：

```text
cluster_num: 32, 64, 128
```

增加 cluster 数量通常能提高表示能力，但计算量和显存都会近似线性增加。

## 11. 公平对比要求

对比 CANO、CANO_Rope 和 CANO_GatedRope 时，应保持以下参数一致：

```text
n_layers
n_hidden
n_head
mlp_ratio
key_ratio
cluster_num
数据划分
归一化方式
训练 epoch
学习率和随机种子
```

每个模型应使用 `phase: restart_train` 重新训练，不能交叉加载 checkpoint。

当前独立 Q/K/V 结构与早期共享前置投影版本的参数形状不同，早期 CANO checkpoint 也不能直接加载。

