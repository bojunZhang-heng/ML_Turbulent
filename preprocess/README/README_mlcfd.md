# MLCFD ShapeNet-Car 前处理说明

本说明对应脚本：

```text
$HOME/ML_Turbulent/preprocess/process_mlcfd.py
```

下面的命令可以直接在 Terminal 中运行，无需手动替换用户名。

脚本直接读取 `param0.tar.gz` 至 `param8.tar.gz`，将每个 CFD 算例转换为完成归一化的数字编号 pkl，并保存可以用于预测结果回写的原始 VTK 网格。

## 1. 输入数据

默认输入目录：

```text
$HOME/Desktop/mlcfd_data/training_data
```

```text
training_data/
├── param0.tar.gz
├── param1.tar.gz
├── ...
└── param8.tar.gz
```

每个算例使用：

```text
hexvelo_smpl.vtk     三维体网格和速度场
quadpress_smpl.vtk   汽车表面网格和压力场
```

不需要预先解压。脚本也兼容已经解压为 `param*/<case_id>/` 的数据。

## 2. 环境

```bash
conda activate pgd-no
```

需要 `numpy`、`scipy` 和 `vtk`。可以运行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python -c "import numpy, scipy, vtk; print(vtk.vtkVersion.GetVTKVersion())"
```

## 3. 输出结构

默认生成两个并列目录：

```text
mlcfd_data/
├── pkl_data/
│   ├── 1.pkl
│   ├── 2.pkl
│   ├── ...
│   ├── n.pkl
│   └── normalization_scalars.pkl
└── raw_data/
    ├── 1_pressure.vtk
    ├── 1_velocity.vtk
    ├── 2_pressure.vtk
    ├── 2_velocity.vtk
    └── ...
```

`pkl_data` 中不建立 fold 子目录，只包含 `1.pkl` 至 `n.pkl` 和最后的归一化参数文件。同一个编号属于同一个算例：

```text
1.pkl
1_pressure.vtk
1_velocity.vtk
```

`raw_data` 中的 VTK 保留原来的节点、单元拓扑和物理场。每个 pkl 保存网络节点到这两个 VTK 点号的映射。

## 4. 单样本测试

```bash
cd "$HOME/ML_Turbulent"

/opt/anaconda3/envs/pgd-no/bin/python \
preprocess/process_mlcfd.py preprocess \
--input-dir "$HOME/Desktop/mlcfd_data/training_data" \
--output-dir "$HOME/Desktop/mlcfd_data/pkl_test" \
--raw-data-dir "$HOME/Desktop/mlcfd_data/raw_test" \
--folds param0 \
--limit 1
```

应生成：

```text
pkl_test/1.pkl
pkl_test/normalization_scalars.pkl
raw_test/1_pressure.vtk
raw_test/1_velocity.vtk
```

## 5. 处理全部数据

使用默认目录时：

```bash
cd "$HOME/ML_Turbulent"
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_mlcfd.py preprocess
```

显式指定目录：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
preprocess/process_mlcfd.py preprocess \
--input-dir "$HOME/Desktop/mlcfd_data/training_data" \
--output-dir "$HOME/Desktop/mlcfd_data/pkl_data" \
--raw-data-dir "$HOME/Desktop/mlcfd_data/raw_data"
```

省略 `--raw-data-dir` 时，会自动在 `pkl_data` 的同级创建 `raw_data`。

已有同名文件时默认停止。确认重新生成可添加：

```bash
--overwrite
```

## 6. 归一化方式

所有数字编号 pkl 都保存完成归一化的数据，不需要 `--normalize` 参数。

### 坐标 XYZ

使用全局 min-max：

```text
xyz_normalized = (xyz - coords_min) / coords_range
coords_range = coords_max - coords_min
```

### SDF

节点到汽车表面的距离使用全局 min-max：

```text
sdf_normalized = (sdf - sdf_min) / sdf_range
sdf_range = sdf_max - sdf_min
```

当前 SDF 是非负距离，且表面节点的 SDF 为 0，因此通常 `sdf_min = 0`。
归一化后，表面节点仍为 0，最远处节点为 1。

### 法向量

法向量已经是单位向量，保持不变。

### 物理量

速度和压力使用 mean-std：

```text
output_normalized = (output - output_mean) / output_std
```

物理量顺序为：

```text
[velocity_x, velocity_y, velocity_z, pressure]
```

`normalization_scalars.pkl` 主要包含：

```text
coords_min
coords_max
coords_range
sdf_min
sdf_max
sdf_range
output_mean
output_std
samples
```

`samples` 记录数字编号、原始 `case_id`、所属 fold、pkl 文件和两个 VTK 文件的对应关系。

## 7. 仅使用训练 fold 计算统计量

例如将 `param8` 作为验证集，使用其余 fold 计算归一化参数：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
preprocess/process_mlcfd.py preprocess \
--input-dir "$HOME/Desktop/mlcfd_data/training_data" \
--output-dir "$HOME/Desktop/mlcfd_data/pkl_data" \
--raw-data-dir "$HOME/Desktop/mlcfd_data/raw_data" \
--statistics-folds param0 param1 param2 param3 param4 param5 param6 param7
```

所有样本都会保存，但统一使用指定训练 fold 的统计量进行归一化，避免验证集信息泄漏。不指定时，使用本次处理的全部样本计算统计量。

## 8. pkl 格式

| 字段 | 形状 | 说明 |
|---|---:|---|
| `sample_index` | 标量 | 数字编号，例如 `1` |
| `sample_id` | 字符串 | 原始算例 ID |
| `fold` | 字符串 | 原始 `param0` 至 `param8` |
| `input` / `x` | `(N, 7)` | 完成归一化的输入 |
| `output` / `y` | `(N, 4)` | 完成归一化的监督目标 |
| `pos` | `(N, 3)` | min-max 后的 XYZ |
| `surface_mask` / `surf` | `(N,)` | 表面节点为 `True` |
| `mapping` | 字典 | 网络节点到原始 VTK 点号的映射 |
| `source` | 字典 | 编号 VTK 文件与原始数据来源 |
| `normalization` | 字典 | 归一化方法 |

输入七维：

```text
[x, y, z, sdf, normal_x, normal_y, normal_z]
```

输出四维：

```text
[velocity_x, velocity_y, velocity_z, pressure]
```

节点排列为：

```text
[体网格内部节点, 汽车表面节点]
```

## 9. 查看 pkl

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
preprocess/process_mlcfd.py inspect \
--sample-pkl "$HOME/Desktop/mlcfd_data/pkl_data/1.pkl"
```

Python 读取示例：

```python
import pickle
from pathlib import Path

path = Path.home() / "Desktop" / "mlcfd_data" / "pkl_data" / "1.pkl"
with open(path, "rb") as file:
    sample = pickle.load(file)

print(sample["sample_index"])
print(sample["sample_id"])
print(sample["input"].shape)
print(sample["output"].shape)
print(sample["source"])
```

## 10. 预测写回 VTK

模型输出通常仍处于标准化尺度，因此使用 `--normalized` 反归一化。

完整 `(N, 4)` 速度和压力预测：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
preprocess/process_mlcfd.py export \
--sample-pkl "$HOME/Desktop/mlcfd_data/pkl_data/1.pkl" \
--prediction "$HOME/Desktop/predictions/1.npy" \
--output-dir "$HOME/Desktop/prediction_vtk" \
--normalized
```

输出：

```text
1_velocity_prediction.vtk
1_pressure_prediction.vtk
```

仅预测表面压力时，支持 `(N_surface, 1)`、`(N, 1)` 以及 batch size 为 1 的对应三维形状，此时只生成压力 VTK。

如果预测已经恢复到原始物理尺度，省略 `--normalized`。

自定义 VTK 字段名：

```bash
--velocity-name LinearNO_velocity \
--pressure-name LinearNO_pressure
```

默认字段名为 `predicted_velocity` 和 `predicted_pressure`。

## 11. ParaView 后处理

1. 打开 `*_pressure_prediction.vtk` 或 `*_velocity_prediction.vtk`。
2. 点击 `Apply`。
3. 选择 `predicted_pressure`，或选择 `predicted_velocity` 的 `Magnitude`。
4. 根据需要使用 `Slice`、`Contour`、`Glyph` 等功能。

## 12. 查看帮助

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_mlcfd.py --help
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_mlcfd.py preprocess --help
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_mlcfd.py inspect --help
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_mlcfd.py export --help
```
