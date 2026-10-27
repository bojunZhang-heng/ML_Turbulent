# InfSA：用于汽车表面物理场预测的 Linear-InfSA

## 1. 来源与实现范围

该模型依据论文：

> Giorgio Roffo, Hazem Abdelkawy, Nilli Lavie, Luke Palmer.  
> *Self-Attention And Beyond the Infinite: Towards Linear Transformers with Infinite Self-Attention*.  
> arXiv:2603.00175v5, 2026.

论文包含两种模型：

- `Pure InfSA`：显式构造 `(N,N)` attention，复杂度为 `O(N²d)`；
- `Linear-InfSA`：计算每个 token 的中心性权重，复杂度对节点数为线性。

汽车表面网格可能具有几十万个节点，因此 `InfSA.py` 只实现
`Linear-InfSA`。如果对 DrivAerNet 使用 Pure InfSA，显式 attention
矩阵会远超 GPU 显存。

## 2. 输入和输出

```text
input:  (B, N, 3)  归一化后的 XYZ 坐标
output: (B, N, 1)  归一化后的 pressure 或 WSS 分量
```

当前模型只使用前三维坐标。数据预处理规则仍由现有数据集负责：

- 坐标 `x`：max-min 归一化；
- 物理量 `y`：mean-std 归一化；
- 测试误差：由训练流程反归一化后计算。

InfSA 不需要 `seg_matrix`，也不需要原始物理坐标。

## 3. Linear-InfSA 核心逻辑

### 3.1 共享 Q/K 投影

论文明确要求：

\[
Q=K=XW_Q.
\]

共享投影让隐式 attention 算子成为对称相似度算子，这是其
Perron-Frobenius 主特征向量解释的基础。代码只有：

```python
self.qk_project = nn.Linear(dim, dim)
```

不存在独立的 `q_proj` 和 `k_proj`。

### 3.2 构造软中心查询

每个 token 的能量为：

\[
e_i=\|Q_i\|_2,
\qquad
\alpha_i=\frac{e_i}{\sum_j e_j+\varepsilon}.
\]

使用能量权重得到中心查询：

\[
\bar q=\sum_i\alpha_iQ_i.
\]

这里不使用 softmax，只进行非负 L1 归一化。

### 3.3 中心性权重

中心查询与所有 Key 计算正相似度：

\[
S_j=[\bar q^TK_j]_+,
\qquad
a_j=\frac{S_j}{\sum_lS_l+\varepsilon}.
\]

`ReLU` 去掉反向相关 token，`a` 是论文用于近似隐式 attention
算子主特征向量的中心性权重。

### 3.4 全局上下文

每个 head 的输出为：

\[
h=\sum_{t=1}^N w_tV_t,
\qquad
w=\gamma a.
\]

`h` 随后广播到所有节点。每个 head 对节点维度是 rank-1，但多个 head
学习不同中心性模式；残差连接继续保存每个节点自己的局部特征：

```python
fx = attention(layer_norm(fx)) + fx
fx = mlp(layer_norm(fx)) + fx
```

因此最终各节点的压力预测不会完全相同。

## 4. gamma 的实现

论文消融实验中 `gamma=0.7` 获得最佳 ImageNet Top-1 和 MoRF-AOC，
因此本项目默认：

```yaml
infsa_gamma: 0.7
infsa_trainable_gamma: true
```

每层、每个 head 都有独立 gamma，并通过 sigmoid 限制在 `(0,1)`：

```python
gamma = sigmoid(gamma_logit)
```

实现中不能使用 `gamma.item()`，否则 gamma 会与计算图断开，无法学习。

训练后可以查看：

```python
print(model.gamma_values())  # (layers, heads)
```

## 5. 为什么适合大型汽车网格

设：

- `N`：网格节点数；
- `C`：隐藏维度；
- `H`：head 数量；
- `d_h=C/H`。

模型的全局 attention 辅助状态仅包含每个 head 的中心查询和上下文：

```text
q_bar:   (B, H, d_h)
context: (B, H, d_h)
```

attention 计算复杂度约为：

\[
O(BNC),
\]

不计必须存在的逐节点输入、Q/V 和输出激活时，全局 attention 辅助状态为：

\[
O(BC),
\]

不会生成 `(B,H,N,N)` 矩阵。整个网络仍需保存逐节点隐藏特征，因此总训练
显存依然随 `N` 线性增长。

## 6. 与公众号参考代码的差异

本实现修正了参考代码中的几个关键问题：

1. 严格共享 Q/K 投影，而不是使用独立 `q_proj`、`k_proj`；
2. gamma 始终作为 Tensor 参与计算，不调用 `.item()`；
3. 不提供会在汽车网格上产生 `N²` 显存的 Pure InfSA 路径；
4. 使用 Pre-LN、残差 MLP 和逐节点 decoder，适配当前 CFD 回归任务；
5. 输入输出接口与 `transolver_model.py` 一致，直接接收 `(B,N,3)`。

## 7. main.py 默认结构

```text
n_layers = 4
n_hidden = 256
n_head = 64
dim_head = 4
mlp_ratio = 2
gamma_init = 0.7
```

论文采用 4 层、64 heads 的 Linear InfViT。本文代码保留 4 层和 64 heads，
但将隐藏维度调整为本项目统一使用的 256，以适配汽车表面网格显存。

可以通过 YAML 修改：

```yaml
model_name: InfSA
phase: restart_train
space_dim: 3

infsa_gamma: 0.7
infsa_trainable_gamma: true
infsa_n_layers: 4
infsa_n_head: 64

batch_size: 1
```

仓库中已经提供可直接修改的示例：

```text
/Users/zhangbojun/ML_Turbulent/SegLinearNO/driver/config_InfSA.yml
```

其中 `data_path` 使用服务器上的 DrivAerNet 示例路径；如果实际目录不同，
只需修改该字段。

`n_hidden=256` 时，`infsa_n_head` 必须是 256 的因数，例如：

```text
8, 16, 32, 64
```

## 8. 训练命令

从 driver 目录运行：

```bash
cd /Users/zhangbojun/ML_Turbulent/SegLinearNO/driver
/opt/anaconda3/envs/pgd-no/bin/python main.py --config config_InfSA.yml
```

首次训练必须使用：

```yaml
phase: restart_train
```

模型权重会保存为：

```text
experiment_<exp>/best_model_InfSA_<physical_quantity>.pth
```

## 9. 需要谨慎理解的地方

论文实验对象是规则图像 patch 和分类任务，而汽车数据是非规则表面节点及
逐节点回归任务。论文尚未验证 Linear-InfSA 在 CFD 稠密预测上的精度，因此：

- 线性复杂度优势可以直接成立；
- 分类精度结论不能直接外推到压力预测；
- rank-1/head 的广播上下文可能限制细粒度局部变化；
- 多 head、残差路径和逐点 MLP 是当前实现恢复局部表达能力的关键。

建议与相同数据划分下的 Transolver、LinearNO 比较相对 L2 误差、训练时间、
峰值显存和每个样本的推理时间。
