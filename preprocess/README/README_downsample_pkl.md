# 全车点云 PKL 下采样说明

本文档介绍脚本：

```text
preprocess/downsample_pkl.py
```

该脚本用于对已经完成预处理和归一化的全车点云 PKL 文件进行下采样。采样点数由命令行参数指定。脚本只改变节点数量，不会重新计算或重复执行归一化。

## 1. 测试数据目录

默认输入目录为：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/E_S_WW_WM_small/pklFile
```

该目录当前包含：

```text
1.pkl
2.pkl
...
10.pkl
normalization_scalars.pkl
```

单个样本的主要数据格式为：

```text
coor       : (N, 3)  float32，归一化后的三维坐标
normals    : (N, 3)  float32，单位法向量
pressure   : (N, 1)  float32，归一化后的压力
```

其中 `N` 是当前汽车表面的节点数量。例如测试数据中的单个样本约有 48 万个节点。

## 2. 下采样的核心原则

假设某个样本需要从 `N` 个节点下采样为 `M` 个节点，脚本首先生成一组公共节点索引：

```python
indices.shape == (M,)
```

随后将同一组索引用于所有逐节点字段：

```python
coor_downsampled = coor[indices]
normals_downsampled = normals[indices]
pressure_downsampled = pressure[indices]
```

因此，输出文件中的坐标、法向量和压力仍然严格一一对应，不会出现坐标来自节点 `i`、法向量或压力却来自其他节点的问题。

除了上述三个字段，只要某个 NumPy 数组满足：

```text
array.shape[0] == N
```

脚本就会将其视为逐节点数据，并使用相同索引同步采样，例如：

```text
coor_raw
features_6d
original_point_ids
node_cluster_flags
wss
```

非数组字段以及不以节点数为第一维的数组会原样保留。

## 3. 支持的采样方法

### 3.1 体素分层采样 `voxel`

这是默认方法，适合汽车表面点云。

其主要流程为：

1. 读取 `coor`；
2. 在当前样本内部将坐标临时转换到 `[0,1]` 空间；
3. 将三维空间划分为规则体素；
4. 从每个被占用的体素中选取一个代表节点；
5. 调整体素分辨率，直到获得足够的候选节点；
6. 从候选节点中选出指定数量的节点；
7. 如果特殊退化几何导致体素代表点不足，则从尚未选中的节点中补足。

注意：坐标转换到 `[0,1]` 只用于确定采样分区，不会修改最终保存的 `coor` 数值。

体素采样的优点：

- 空间分布通常比纯随机采样更均匀；
- 不容易在局部高密度区域保留过多节点；
- 更有利于保留汽车整体几何轮廓。

### 3.2 均匀随机采样 `random`

随机采样直接从 `[0,N)` 中进行无放回抽样：

```python
indices = rng.choice(N, size=M, replace=False)
```

其优点是速度快、逻辑简单。缺点是可能出现局部节点较密、其他区域节点较稀的情况。

脚本使用 `--seed` 控制随机数。同样的输入数据、采样方法、采样数量和随机种子将得到相同结果。

## 4. 分区矩阵的处理

如果样本中包含：

```text
seg_matrix.shape == (S, N)
```

其中 `S` 是几何分区数量，脚本会对节点维度进行采样：

```python
seg_matrix_downsampled = seg_matrix[:, indices]
```

随后对每个非空分区重新执行行归一化，使其保持聚合权重的含义：

```text
每个非空分区的权重和 = 1
```

如果下采样后某个分区没有保留任何节点，该分区行会保留为全零，不会自动删除。因此 `S` 不会改变。

## 5. 输出文件新增字段

每个下采样后的样本会增加：

```text
downsample_indices : (M,) int64
downsampling       : dict
```

其中：

```python
downsampling = {
    "method": "voxel",          # 或 random
    "original_num_points": N,
    "sampled_num_points": M,
}
```

`downsample_indices` 表示每个输出节点在输入 PKL 点云中的位置：

```python
sampled_coor == original_coor[downsample_indices]
```

如果输入文件还包含 `original_point_ids`，该字段也会同步采样。此时：

- `downsample_indices`：映射到本次输入 PKL；
- `original_point_ids`：映射到最初保存的 VTK 节点编号。

## 6. normalization_scalars.pkl 如何处理

脚本不会重新计算归一化参数，而是将输入目录中的：

```text
normalization_scalars.pkl
```

原样复制到输出目录。

原因是输入 PKL 已经使用全数据集统计量完成归一化：

- 坐标使用 min-max；
- 压力等物理量使用 mean-std；
- 法向量保持单位向量。

下采样只是在已有节点中选择子集，不应该再次归一化，否则训练集、验证集和测试集可能使用不同的尺度。

下采样后，某一个样本自身的压力均值不一定严格为 0、标准差也不一定严格为 1，这是正常现象。反归一化时仍应使用复制后的原始全数据集统计量。

## 7. 基本使用方法

首先进入项目目录：

```bash
cd /Users/zhangbojun/ML_Turbulent
```

### 7.1 默认目录下采样为 100000 点

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_pkl.py \
  --num-points 100000
```

默认使用：

```text
method = voxel
seed = 42
```

默认输出目录为：

```text
preprocess/E_S_WW_WM_small/pklFile_voxel_100000
```

### 7.2 下采样为 50000 点

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_pkl.py \
  --num-points 50000
```

### 7.3 使用随机采样

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_pkl.py \
  --num-points 50000 \
  --method random \
  --seed 42
```

### 7.4 指定输入和输出目录

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_pkl.py \
  --input-dir /Users/zhangbojun/ML_Turbulent/preprocess/E_S_WW_WM_small/pklFile \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/E_S_WW_WM_small/pklFile_100000 \
  --num-points 100000 \
  --method voxel \
  --seed 42
```

### 7.5 覆盖已有输出

为了避免误删已有结果，默认情况下如果目标文件已经存在，程序会停止并提示错误。

确认需要覆盖时使用：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_pkl.py \
  --num-points 100000 \
  --overwrite
```

## 8. 参数说明

| 参数 | 是否必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--num-points` | 是 | 无 | 每个样本最终保留的节点数 |
| `--input-dir` | 否 | 测试 PKL 目录 | 输入 PKL 文件目录 |
| `--output-dir` | 否 | 自动生成 | 输出目录 |
| `--method` | 否 | `voxel` | `voxel` 或 `random` |
| `--seed` | 否 | `42` | 随机种子 |
| `--overwrite` | 否 | 关闭 | 允许覆盖已有输出文件 |

查看完整命令帮助：

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/downsample_pkl.py --help
```

## 9. 输出目录示例

假设执行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_pkl.py \
  --num-points 100000 \
  --method voxel
```

输出目录结构为：

```text
pklFile_voxel_100000/
├── 1.pkl
├── 2.pkl
├── 3.pkl
├── ...
├── 10.pkl
└── normalization_scalars.pkl
```

样本编号和文件名保持不变，因此原有 dataloader 通常只需要将 `data_path` 修改为新的输出目录。

## 10. 查看下采样后的文件

可以使用以下代码检查结果：

```python
import pickle

sample_path = (
    "/Users/zhangbojun/ML_Turbulent/preprocess/"
    "E_S_WW_WM_small/pklFile_voxel_100000/1.pkl"
)

with open(sample_path, "rb") as file:
    sample = pickle.load(file)

print(sample["coor"].shape)
print(sample["normals"].shape)
print(sample["pressure"].shape)
print(sample["downsample_indices"].shape)
print(sample["downsampling"])
```

预期输出类似：

```text
(100000, 3)
(100000, 3)
(100000, 1)
(100000,)
{'method': 'voxel', 'original_num_points': 482634, 'sampled_num_points': 100000}
```

## 11. 将下采样预测映射回原点云

假设模型只预测了下采样节点上的压力：

```python
prediction.shape == (M, 1)
```

可以先构造一个原节点数大小的数组，再将预测值放回采样位置：

```python
import numpy as np

num_original_points = sample["downsampling"]["original_num_points"]
indices = sample["downsample_indices"]

prediction_on_original = np.full(
    (num_original_points, prediction.shape[1]),
    np.nan,
    dtype=prediction.dtype,
)
prediction_on_original[indices] = prediction
```

这样只能在采样节点位置恢复预测值，未采样节点仍然是 `NaN`。如果希望获得完整原网格上的连续预测场，还需要使用最近邻、反距离加权或基于网格拓扑的插值方法。下采样索引本身不能恢复从未送入模型的节点预测值。

## 12. 注意事项

1. `--num-points` 不能大于某个样本原有的节点数；脚本不使用有放回采样。
2. 输出目录不能与输入目录相同，避免覆盖原始数据。
3. 所有样本都会采样到相同的目标节点数，因此适合组成 batch。
4. 如果不同样本的节点数不一致，目标数量必须不大于最小样本的节点数。
5. 对汽车表面压力预测，建议优先使用 `voxel`，因为它通常比纯随机采样更均匀。
6. 下采样会损失局部高曲率区域和压力峰值附近的信息。采样数量过小时，即使整体轮廓仍然完整，局部压力预测精度也可能下降。
7. 如果需要重点保留车头、A 柱、后视镜、轮胎或尾部等区域，应进一步使用曲率或物理量加权采样，而不是只使用均匀体素采样。

