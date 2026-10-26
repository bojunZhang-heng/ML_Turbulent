# RopeLinearNO_v4

## 1. 模型目标

`RopeLinearNO_v4.py` 是一个完全由相对位置核驱动的线性 attention
神经算子，用于三维汽车表面压力预测。

与 `RopeLinearNO_v3` 的根本区别是：

- 不再保留普通 LinearNO 输出分支；
- 不再包含门控网络或 `relative_gate_logit`；
- 不执行普通输出与位置输出之间的插值；
- 每一个 attention block 都直接输出相对位置核 attention 的结果。

模型输入和输出为：

```text
data:        (B, N, 3)  min-max 归一化坐标，用于输入特征编码
rope_coords: (B, N, 3)  原始坐标，用于建立相对位置核
output:      (B, N, 1)  mean-std 归一化后的表面压力预测
```

## 2. 核心公式

对于节点 `i` 和 `j`，首先由内容特征生成 `Q/K/V`。位置作用由非负
Fourier 核给出：

\[
\kappa_{ij}^{(h)}=
\frac{1}{2}+
\frac{1}{2R}\sum_{r=1}^{R}
\cos\left(\phi_{i,h,r}-\phi_{j,h,r}\right),
\]

其中：

\[
\phi_{i,h,r}=\omega_{h,r}^{\mathrm T}\hat p_i.
\]

`ω` 是可学习频率，`R` 是 `num_position_frequencies`。由于余弦平均值
属于 `[-1,1]`，因此：

\[
0\leq\kappa_{ij}^{(h)}\leq1.
\]

V4 的唯一 attention 输出为：

\[
Y_i=
\frac{
\sum_j(q_i^{\mathrm T}k_j)\kappa_{ij}v_j
}{
\sum_j(q_i^{\mathrm T}k_j)\kappa_{ij}
}.
\]

分母用于重新归一化加入位置核后的权重。

## 3. 为什么公式中仍有常数 1/2

代码中的：

```python
numerator = self._constant_kernel_term(q, k, v)
denominator = torch.full_like(q[..., 0], 0.5)
```

不是保留普通 LinearNO 分支。常数 `1/2` 是非负位置核 `κ` 的组成部分，
负责把 Fourier 余弦平均值从 `[-1,1]` 平移到 `[0,1]`。

如果删掉这个常数，位置核可能为负，attention 权重也可能变成负数。因此
V4 虽然完全取消了普通 LinearNO 并行输出，仍必须保留该核常数项。

## 4. 无 N × N 矩阵的相对位置计算

模型利用：

\[
\cos(a-b)=\cos(a)\cos(b)+\sin(a)\sin(b)
\]

把点对位置核拆成可分离的 cosine/sine 项。每项按照以下顺序计算：

```text
K_j × position_feature_j
        ↓
先聚合 K_position^T V
        ↓
再与 Q_i 相乘
        ↓
乘 position_feature_i
```

因此不会显式生成 `(B,H,N,N)` attention 矩阵，对节点数 `N` 保持线性
显存复杂度。

## 5. 坐标处理

归一化坐标 `data` 进入输入编码器：

```python
fx = preprocess(data)
```

原始坐标 `rope_coords` 只用于位置核。模型内部首先转换为无量纲坐标：

\[
\hat p_i=
\frac{p_i-p_{center}}
{\|p_{max}-p_{min}\|_2}.
\]

这样相对位置核具有平移不变性，并降低不同车辆整体尺度对频率相位的影响。

`data` 与 `rope_coords` 必须使用完全相同的节点顺序，并具有相同的
`(B,N,3)` 形状。

## 6. DrivAerNet 显存优化

DrivAerNet 单个表面可能接近 50 万节点。V4 默认启用两级 checkpoint：

```python
checkpoint_relative_terms=True
checkpoint_segment_size=1
```

- 每个 cosine/sine 低秩项单独 checkpoint；
- 每个网络 block 单独 checkpoint；
- 不降低隐藏维度、层数或位置频率数；
- 用额外的反向重计算换取更低的峰值显存。

验证和测试阶段处于 `eval()` 和 `no_grad()`，不会执行训练 checkpoint。

## 7. main.py 配置

配置文件中使用：

```yaml
model_name: RopeLinearNO_v4
space_dim: 3
num_position_frequencies: 6
batch_size: 1
```

V4 不读取 `initial_relative_gate`，因为模型中已经不存在门控参数。

`main.py` 当前使用以下结构：

```text
n_layers = 8
n_hidden = 256
n_head = 8
mlp_ratio = 2
key_ratio = 1
out_dim = 1
```

## 8. 训练调用链

`ml.py` 和 `ml_SATO.py` 都将 V4 归入需要原始坐标的位置模型：

```python
model(coorf[..., :3], rope_coords=raw_coords)
```

其中：

- `coorf[..., :3]` 是归一化后的三维坐标；
- `raw_coords` 是同一网格节点对应的原始三维坐标。

## 9. 与 V3 的对比

| 项目 | RopeLinearNO_v3 | RopeLinearNO_v4 |
| --- | --- | --- |
| 普通 LinearNO 输出 | 保留 | 删除 |
| 相对位置输出 | 保留 | 唯一 attention 输出 |
| 门控参数 | 每层、每个 head 一个 | 无 |
| 输出融合 | `(1-g)Y_base + gY_rel` | `Y_rel` |
| 非负 Fourier 核 | 有 | 有 |
| 逐项 checkpoint | 有 | 有 |
| block checkpoint | 每层 | 每层 |

V4 是新的模型结构，不能直接严格加载 V3 checkpoint。训练时应使用：

```yaml
phase: restart_train
```

从头训练并生成独立的 `best_model_RopeLinearNO_v4_pressure.pth`。
