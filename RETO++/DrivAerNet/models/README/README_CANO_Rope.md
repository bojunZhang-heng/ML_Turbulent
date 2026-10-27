
# CANO_Rope

## 1. 文件与用途

模型文件：

```text
models/CANO_Rope.py
```

该模型在 CANO 的 asymmetric cluster cross-attention 中加入连续三维 RoPE。

输入输出：

```text
data:        (B, N, 3)  min-max 归一化坐标，用于输入编码
rope_coords: (B, N, 3)  原始物理坐标，用于计算三维 RoPE
output:      (B, N, 1)  归一化压力预测
```

该版本完全使用 RoPE attention logits，不保留普通 CANO logits，主要用于验证“直接把相对位置写入 CANO cross-attention”是否有效。

Q、K、V 均通过各自独立的线性层直接从完整隐藏特征生成：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

它们不共享输入投影，各 attention head 也使用独立的输出权重通道。

## 2. 为什么不能简单旋转 Qc

CANO 的 cluster query 为：

\[
Q_c=C^\top Q.
\]

一个 cluster 可能包含多个不同位置的节点，因此 `Qc` 没有唯一、准确的空间位置。

如果先计算 cluster 中心 `p_c`，再执行：

\[
R(p_c)Q_c,
\]

则会丢失 cluster 内部各节点的位置，而且一般有：

\[
R(p_c)C^\top Q
\neq
C^\top R(p_i)Q_i.
\]

因此当前实现先旋转每个节点的 Q，再执行 Query 聚类。

## 3. 当前 RoPE 融合公式

对节点坐标 `p_i`，首先产生连续三维旋转：

\[
\widetilde Q_i=R(p_i)Q_i,
\qquad
\widetilde K_i=R(p_i)K_i.
\]

cluster assignment 仍由未旋转的内容 Query 产生：

\[
C=\operatorname{Softmax}_{K_c}(QW_c).
\]

然后聚合旋转后的 Query：

\[
\widetilde Q_c=C^\top\widetilde Q.
\]

Cross-attention 为：

\[
Z_c=
\operatorname{Softmax}
\left(
\frac{\widetilde Q_c\widetilde K^\top}{\sqrt d}
\right)V.
\]

最后广播：

\[
Y=CZ_c.
\]

展开某个 cluster 和节点之间的打分：

\[
S_{cj}
=
\sum_i C_{ic}
Q_i^\top R(p_i)^\top R(p_j)K_j.
\]

RoPE 满足：

\[
R(p_i)^\top R(p_j)=R(p_j-p_i),
\]

因此打分包含 cluster 内节点 `i` 与完整 Key 节点 `j` 的相对位置，而不是只包含一个近似 cluster 中心。

## 4. 连续三维 RoPE

`ContinuousRotaryEmbedding3D` 首先对原始坐标做样本内无量纲化：

\[
\hat p_i=
\frac{p_i-p_{center}}
{\lVert p_{max}-p_{min}\rVert_2}.
\]

然后使用每个 head 独立的可学习三维频率：

\[
\phi_{i,h,r}=\omega_{h,r}^\top\hat p_i.
\]

每两个 Q/K 通道组成一个二维旋转对：

\[
\begin{bmatrix}
x'_{2r}\\
x'_{2r+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos\phi_r & -\sin\phi_r\\
\sin\phi_r & \cos\phi_r
\end{bmatrix}
\begin{bmatrix}
x_{2r}\\
x_{2r+1}
\end{bmatrix}.
\]

频率初始化覆盖 X/Y/Z 三个方向和多个空间尺度，训练过程中频率方向与幅值可以继续学习。

## 5. 坐标分工

| 坐标 | 用途 | 处理方式 |
| --- | --- | --- |
| `data` | 输入 MLP，生成节点内容特征 | 数据集预处理阶段 min-max |
| `rope_coords` | 构造 RoPE 相位 | 原始坐标传入，模型内部无量纲化 |

两者必须保持完全相同的节点顺序：

```text
data[b, i] 与 rope_coords[b, i] 必须代表同一个网格节点
```

## 6. 核心代码

```python
q, k, v = self._project_qkv(x)
cluster_assignment = self._cluster_assignment(q)

q_rope = apply_rotary_embedding(q, position_phases)
k_rope = apply_rotary_embedding(k, position_phases)

clustered_q_rope = torch.einsum(
    "bhnk,bhnd->bhkd", cluster_assignment, q_rope
)

cluster_features = self._cross_attention(
    clustered_q_rope, k_rope, v
)
output = self._broadcast_clusters(
    cluster_assignment, cluster_features
)
```

注意：Value 不执行 RoPE。RoPE 只修改 Query/Key 的相关性计算。

## 7. 复杂度

相位生成和 Q/K 旋转为：

\[
O(Nd).
\]

Cluster cross-attention 为：

\[
O(NK_cd).
\]

总复杂度仍为：

\[
O(NK_cd),
\]

不会产生 `N × N` attention 矩阵。

## 8. 参数限制

RoPE 要求每个 head 的 Q/K 维度为偶数：

```text
key_dim = key_ratio * (n_hidden // n_head)
key_dim % 2 == 0
```

当前主配置：

```text
n_hidden = 256
n_head = 8
key_ratio = 1
key_dim = 32
```

满足要求。

## 9. main.py 调用

模型名：

```yaml
model_name: CANO_Rope
```

构造方式：

```python
model = CANO_Rope(
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
)
```

`ml_SATO.py` 中调用：

```python
model(coorf[..., :3], rope_coords=raw_coords)
```

## 10. 推荐配置

```yaml
model_name: CANO_Rope
phase: restart_train
space_dim: 3

cluster_num: 64
num_position_frequencies: 6
batch_size: 1
```

建议分别测试：

```text
cluster_num:              32, 64, 128
num_position_frequencies: 3, 6, 9
```

## 11. 风险与观察指标

纯 RoPE 版本完全替换普通 CANO logits，位置影响较强，可能出现：

- 初期训练不稳定；
- 内容相关性被位置旋转过度扰动；
- 动态 cluster 包含相距较远节点时，相位发生抵消；
- 频率过高时对坐标噪声敏感。

因此该模型适合作为明确的消融实验。如果它不如普通 CANO，应进一步观察 `CANO_GatedRope`，而不是直接判定相对位置思想无效。

独立 Q/K/V 版本不能加载早期共享前置投影版本的 checkpoint，修改后必须使用 `phase: restart_train`。

