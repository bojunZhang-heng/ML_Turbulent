# `main.py` 逻辑梳理

本文档说明仓库根目录 [`main.py`](main.py) 的整体逻辑、函数调用关系、数据流以及训练和评估过程。

> 本文档讨论的是仓库根目录的 `main.py`，不是 `ShapeNetCar/main.py`。前者是基于 YAML 配置的统一入口，后者是较早的命令行脚本。

## 1. 整体调用关系

```text
main()
├── parse_args()
├── read_config()
├── set_seed()
├── select_device()
├── load_datasets()
│   ├── prepare_archives()
│   ├── get_samples()
│   ├── cache_is_complete()
│   ├── load_train_val_fold_file()
│   └── GraphDataset()
├── train_model()
│   ├── build_model()
│   └── ShapeNetCar/train.py::main()
├── load_model()
└── evaluate_model()
    └── utils.drag_coefficient::cal_coefficient()
```

`main.py` 本身主要负责流程编排，具体的数据解析、模型结构和训练循环分别位于 `ShapeNetCar` 子目录中。

## 2. 程序入口

直接执行：

```bash
python main.py --config config_LinearNO.yml
```

程序从 `main()` 开始：

1. 读取命令行中的配置文件路径；
2. 加载 YAML 配置；
3. 检查运行模式；
4. 设置随机种子和运行设备；
5. 准备训练集和验证集；
6. 根据模式选择准备数据、训练、评估或训练后评估。

## 3. 配置读取与运行模式

### 3.1 命令行参数

`parse_args()` 只定义一个必填参数：

```bash
--config
```

例如：

```bash
python main.py --config config_LinearNO.yml
```

### 3.2 YAML 配置

`read_config()` 使用 `yaml.safe_load()` 读取配置，并要求 YAML 根节点是字典。

当前配置文件 [`config_LinearNO.yml`](config_LinearNO.yml) 主要包含以下部分：

```yaml
mode: prepare
runtime: ...
dataset: ...
model: ...
training: ...
output: ...
evaluation: ...
```

### 3.3 运行模式

支持四种模式：

| 模式 | 行为 |
|---|---|
| `prepare` | 解压数据并构建或检查预处理缓存，不训练模型 |
| `train` | 准备数据并训练模型 |
| `evaluate` | 加载已有检查点并评估 |
| `train_and_evaluate` | 训练模型后立即评估 |

如果未设置 `mode`，默认使用 `train_and_evaluate`。

## 4. 运行环境设置

### 4.1 随机种子

`set_seed()` 同时设置 Python `random`、NumPy、PyTorch CPU 和 PyTorch CUDA 的随机种子。

当配置中的 `runtime.deterministic` 为 `true` 时，还会关闭 cuDNN benchmark 并启用确定性行为，以提高实验可复现性。

### 4.2 计算设备

`select_device()` 支持：

```yaml
device: auto
```

此时有 CUDA 使用 `cuda:0`，否则使用 CPU。也可以显式指定 `cpu` 或 `cuda:0` 等设备；如果请求 CUDA 但当前环境不可用，会抛出异常。

## 5. 数据准备流程

`main()` 调用：

```python
train_dataset, val_dataset, coef_norm, val_samples, data_dir = load_datasets(config)
```

该函数返回：

| 返回值 | 含义 |
|---|---|
| `train_dataset` | 图形式训练数据集 |
| `val_dataset` | 图形式验证数据集 |
| `coef_norm` | 输入和输出的归一化统计量 |
| `val_samples` | 验证样本名称列表 |
| `data_dir` | 原始 ShapeNetCar 数据目录 |

### 5.1 数据路径

配置：

```yaml
dataset:
  root: /data/zhangbojun/mlcfd_data
  training_subdir: training_data
  cache_subdir: preprocessed_data
```

对应目录：

```text
/data/zhangbojun/mlcfd_data/training_data
/data/zhangbojun/mlcfd_data/preprocessed_data
```

相对路径会相对于仓库根目录解析，绝对路径则直接使用。

### 5.2 自动解压

ShapeNetCar 数据包含 9 个 fold：

```text
param0, param1, ..., param8
```

`prepare_archives()` 会检查这些目录。如果目录不存在、对应的 `param*.tar.gz` 存在且 `dataset.auto_extract` 为 `true`，就调用 `safe_extract()` 自动解压。

`safe_extract()` 会检查压缩包成员路径，阻止 `../` 等路径穿越，避免文件被解压到目标目录之外。

如果目录和压缩包都不存在，程序会报告缺失的 fold。

### 5.3 fold 划分

`get_samples()` 扫描每个 `param*` 目录下的样本子目录。

配置：

```yaml
dataset:
  fold_id: 0
```

对应划分为：

```text
验证集：param0
训练集：param1-param8
```

一般地，`fold_id` 指定的 fold 作为验证集，其余 8 个 fold 合并为训练集。

### 5.4 预处理缓存

当：

```yaml
preprocessed: auto
```

`cache_is_complete()` 会检查每个样本目录是否包含：

```text
x.npy
y.npy
pos.npy
surf.npy
edge_index.npy
```

如果全部存在，则直接读取缓存；否则重新解析 VTK 文件并构建缓存。也可以显式设置：

```yaml
preprocessed: true
```

或：

```yaml
preprocessed: false
```

如果训练集或验证集为空，程序会提示删除不完整缓存目录，或设置 `preprocessed: false` 重新构建。

## 6. 单个样本的数据结构

原始样本主要来自：

```text
quadpress_smpl.vtk
hexvelo_smpl.vtk
```

数据预处理逻辑位于 [`ShapeNetCar/dataset/dataset.py`](ShapeNetCar/dataset/dataset.py)。

每个样本被组织成 PyTorch Geometric 的 `Data` 对象，主要字段为：

| 字段 | 含义 |
|---|---|
| `x` | 每个节点的输入特征 |
| `y` | 每个节点的监督目标 |
| `pos` | 节点空间坐标 |
| `surf` | 表面节点布尔标记 |
| `edge_index` | 图的边连接关系 |

### 6.1 输入特征

输入特征由以下内容组成：

```text
3 维空间坐标
1 维 signed distance
3 维表面法向量
```

总输入维度为 7，与模型配置中的：

```yaml
space_dim: 3
fun_dim: 4
```

一致，即 `space_dim + fun_dim = 7`。

### 6.2 输出目标

输出由：

```text
3 维速度
1 维压力
```

组成，总输出维度为 4：

```yaml
out_dim: 4
```

通道顺序为：

```text
y[:, 0:3]：速度
y[:, 3]：压力
```

### 6.3 归一化

训练集首先计算输入和输出的均值、标准差：

```text
coef_norm = (mean_in, std_in, mean_out, std_out)
```

训练数据的 `x` 和 `y` 都会被标准化。验证集使用训练集的 `coef_norm`，不能单独计算验证集统计量，否则会造成数据分布不一致。

## 7. `GraphDataset` 的作用

`load_datasets()` 将原始 `Data` 列表包装为：

```python
GraphDataset(train_data, use_cfd_mesh=use_cfd_mesh, r=radius)
GraphDataset(val_data, use_cfd_mesh=use_cfd_mesh, r=radius)
```

每次取样返回：

```python
(cfd_data, geometry)
```

其中：

- `cfd_data` 是完整的 CFD 图数据；
- `geometry` 是由表面点坐标组成的几何点云。

当 `use_cfd_mesh` 为 `false` 时，`GraphDataset` 会根据 `radius` 重新构建半径图；为 `true` 时沿用原始 CFD 网格的边。

## 8. 模型构建和前向传播

`build_model()` 实例化：

```python
LinearAttentionNeuralOperator(**config["model"])
```

模型定义位于 [`ShapeNetCar/models/LinearAttnNeuralOperator.py`](ShapeNetCar/models/LinearAttnNeuralOperator.py)。

整体结构可以理解为：

```text
7 维节点输入
    ↓
预处理 MLP
    ↓
多个 LinearNO block
    ↓
每个节点输出 4 维
```

模型配置中的：

```yaml
n_layers: 8
n_hidden: 256
n_head: 8
```

表示使用 8 个 LinearNO block、隐藏维度 256、多头数 8。

前向传播的核心流程是：

```python
cfd_data, geom_data = data
x = cfd_data.x
x = x[None, :, :]
fx = self.preprocess(x)
fx = fx + self.placeholder[None, None, :]
for block in self.blocks:
    fx = block(fx)
return fx[0]
```

当前默认配置中 `unified_pos: false`，并且没有额外的 `fx` 输入，因此模型主要使用 `cfd_data.x`。虽然 `geometry` 被传入模型，但在当前默认路径中没有直接参与计算。

## 9. 训练流程

训练由 `train_model()` 发起，并调用 [`ShapeNetCar/train.py`](ShapeNetCar/train.py) 中的 `main()`。

配置示例：

```yaml
training:
  epochs: 200
  batch_size: 1
  learning_rate: 0.001
  pressure_loss_weight: 0.5
  validation_interval: 10
```

当前实现强制要求：

```text
batch_size = 1
epochs >= 1
```

这是因为不同 CFD 图的节点数量可能不同，当前实现没有做统一图批处理。

### 9.1 单个 batch 的训练

每个 batch 的步骤为：

```text
取出 (cfd_data, geometry)
↓
移动到 device
↓
模型前向传播
↓
计算速度损失和压力损失
↓
反向传播
↓
Adam 更新参数
↓
OneCycleLR 更新学习率
```

### 9.2 损失函数

压力损失只在表面节点上计算：

```python
loss_press = MSE(out[cfd_data.surf, -1], targets[cfd_data.surf, -1])
```

速度损失在所有节点的前三个输出通道上计算：

```python
loss_velo = MSE(out[:, :-1], targets[:, :-1])
```

总损失为：

```text
total_loss = loss_velo + pressure_loss_weight × loss_press
```

默认情况下：

```text
total_loss = loss_velo + 0.5 × loss_press
```

### 9.3 验证和保存

每隔 `validation_interval` 个 epoch 验证一次，最后一个 epoch 必定验证。

训练结束后保存：

```text
output.directory/model_<epochs>.pth
output.directory/log_<epochs>.json
```

模型保存的是完整的 PyTorch 模型对象，而不是单独的 `state_dict`。

## 10. 检查点加载

`checkpoint_path()` 的优先级如下：

1. 如果 `evaluation.checkpoint` 被显式设置，使用该路径；
2. 否则根据训练轮数自动推导：

```text
output.directory/model_<epochs>.pth
```

例如：

```yaml
output:
  directory: outputs/shapenet/fold0

training:
  epochs: 200
```

默认检查点为：

```text
outputs/shapenet/fold0/model_200.pth
```

## 11. 评估流程

`evaluate_model()` 会：

1. 将模型移动到指定设备；
2. 切换到 `eval()` 模式；
3. 使用 `torch.no_grad()` 关闭梯度；
4. 遍历验证集；
5. 计算节点级误差；
6. 可选地计算阻力系数；
7. 保存预测结果和指标。

### 11.1 反归一化

模型输出仍然位于归一化空间，因此先恢复物理量：

```python
prediction_denorm = prediction * std + mean
target_denorm = target * std + mean
```

其中 `mean` 和 `std` 来自训练集输出统计量 `coef_norm[2]` 和 `coef_norm[3]`。

### 11.2 压力和速度的评估区域

压力只在表面节点上提取：

```python
prediction[surface, -1]
```

速度指标只在非表面节点上提取：

```python
prediction[~surface, :-1]
```

表面速度则单独提取，用于后续阻力系数计算。

### 11.3 评估指标

最终指标包括：

| 指标 | 含义 |
|---|---|
| `relative_l2_pressure` | 表面压力相对 L2 误差 |
| `relative_l2_velocity` | 非表面节点速度相对 L2 误差 |
| `pressure_rmse` | 压力 RMSE |
| `velocity_rmse_components` | 三个速度分量的 RMSE |
| `velocity_rmse` | 综合速度 RMSE |
| `drag_relative_error` | 阻力系数平均绝对相对误差 |
| `drag_spearman_rho` | 真实与预测阻力系数的 Spearman 相关系数 |

阻力相关指标仅在 `compute_drag: true` 时计算。

## 12. 输出文件

评估配置示例：

```yaml
evaluation:
  results_directory: results/shapenet/fold0
  save_predictions: true
  compute_drag: true
```

如果开启 `save_predictions`，每个验证样本会生成：

```text
<index>_pred.npy
<index>_gt.npy
```

这些数组已经反归一化，每个节点包含：

```text
velocity_x, velocity_y, velocity_z, pressure
```

所有评估指标保存到：

```text
results/shapenet/fold0/metrics.json
```

## 13. 四种模式的执行路径

### `prepare`

```text
读取配置
→ 设置环境
→ 解压数据
→ 检查或构建缓存
→ 加载数据集
→ 结束
```

注意：该模式仍然会调用 `load_datasets()`，因此当缓存不完整时会实际执行数据预处理。

### `train`

```text
准备数据
→ 构建模型
→ 训练
→ 保存模型和训练日志
```

### `evaluate`

```text
准备数据
→ 加载 checkpoint
→ 评估验证集
→ 保存预测和指标
```

### `train_and_evaluate`

```text
准备数据
→ 训练模型
→ 使用训练后的模型评估
→ 保存模型、预测和指标
```

## 14. 当前实现需要注意的地方

### 14.1 根目录 `main.py` 与 `ShapeNetCar/main.py` 不同

根目录脚本使用 YAML：

```bash
python main.py --config config_LinearNO.yml
```

`ShapeNetCar/main.py` 使用较早的命令行参数接口，例如 `--data_dir`、`--gpu` 和 `--nb_epochs`。两者不要混用配置方式。

### 14.2 训练和评估的速度区域不同

训练时速度损失使用所有节点：

```python
out[:, :-1]
```

评估时速度指标只使用非表面节点：

```python
prediction[~surface, :-1]
```

因此训练损失与最终报告的速度指标并不是完全相同的统计区域。

### 14.3 几何点云目前没有直接参与默认前向计算

数据集返回了 `geometry`，模型也接收了它，但在 `unified_pos: false` 的默认路径中，模型主要使用 `cfd_data.x`。如果希望显式使用几何点云，需要进一步修改模型的前向传播逻辑。

### 14.4 模型保存格式是完整模型

当前使用：

```python
torch.save(model, path)
```

因此加载时依赖模型类的导入路径和代码结构。若后续需要更好的跨环境兼容性，可以考虑保存和加载 `state_dict`。

## 15. 一次完整训练评估示例

将配置中的：

```yaml
mode: prepare
```

改为：

```yaml
mode: train_and_evaluate
```

然后执行：

```bash
python main.py --config config_LinearNO.yml
```

完整流程为：

```text
1. 读取配置
2. 设置随机种子和设备
3. 检查 param0-param8 数据
4. 自动解压缺失的 fold
5. 读取或构建预处理缓存
6. 使用 param0 作为验证集
7. 使用 param1-param8 作为训练集
8. 构建 GraphDataset
9. 创建 LinearAttentionNeuralOperator
10. 训练 200 个 epoch
11. 保存 model_200.pth 和 log_200.json
12. 评估 param0
13. 保存预测文件
14. 保存 metrics.json
```
