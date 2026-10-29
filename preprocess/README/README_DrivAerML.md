# DrivAerML 数据集 PKL 转换说明

本文档介绍以下预处理脚本的代码逻辑和使用方法：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/process_DrivAerML.py
```

该脚本用于将 DrivAerML 数据集原始的 `run_*` 目录和 NPY 文件转换为逐样本 PKL 文件，并完成：

- 坐标的全数据集 min-max 归一化；
- 物理量的全数据集、逐分量 mean-std 归一化；
- 归一化统计量保存；
- 完整样本与残缺样本标记；
- 输出文件逐一验证；
- 可选删除原始 `run_*` 目录。

## 1. 当前数据目录

处理后的数据位于：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000
```

当前目录已经完成转换，原来的 `run_*` 目录已经删除。目录中直接保存逐样本 PKL：

```text
drivaerML_N100_000/
├── 1.pkl
├── 2.pkl
├── 3.pkl
├── ...
├── 500.pkl
├── normalization_scalars.pkl
├── data.ps1
└── data_process.py
```

原始数据中存在空目录，因此样本编号并非严格连续。当前统计为：

```text
原始 run_* 目录：500
完整样本：465
部分样本：19
空目录：16
生成的样本 PKL：484
normalization_scalars.pkl：1
```

## 2. 原始数据格式

一个完整的原始目录结构类似：

```text
run_1/
├── surface_position_vtp.npy
├── surface_pressure.npy
├── surface_wallshearstress.npy
├── volume_cell_position.npy
├── volume_cell_velocity.npy
└── volume_cell_totalpcoeff.npy
```

原始数组格式为：

| 文件 | 原始形状 | 含义 |
|---|---:|---|
| `surface_position_vtp.npy` | `(100000, 3)` | 表面点坐标 |
| `surface_pressure.npy` | `(100000,)` | 表面压力 |
| `surface_wallshearstress.npy` | `(100000, 3)` | 壁面剪切应力 |
| `volume_cell_position.npy` | `(100000, 3)` | 体网格单元中心坐标 |
| `volume_cell_velocity.npy` | `(100000, 3)` | 体网格速度 |
| `volume_cell_totalpcoeff.npy` | `(100000,)` | 总压系数 |

## 3. PKL 字段映射

脚本采用以下字段映射：

| 原始 NPY 文件 | PKL 字段 | 输出形状 |
|---|---|---:|
| `surface_position_vtp.npy` | `x_surface` | `(N_surface, 3)` |
| `surface_pressure.npy` | `y_spressure` | `(N_surface, 1)` |
| `volume_cell_position.npy` | `x_volume` | `(N_volume, 3)` |
| `volume_cell_velocity.npy` | `y_velocity` | `(N_volume, 3)` |
| `surface_wallshearstress.npy` | `y_wss` | `(N_surface, 3)` |
| `volume_cell_totalpcoeff.npy` | `volume_cell_totalpcoeff` | `(N_volume, 1)` |

`volume_cell_totalpcoeff.npy` 在原始数据中真实存在，但没有指定新的名称，因此脚本保留其原文件主名称，防止数据被静默丢弃。

标量数组会统一转换为二维列向量：

```python
surface_pressure:          (N,) -> (N, 1)
volume_cell_totalpcoeff:   (N,) -> (N, 1)
```

所有输出数组均保存为：

```text
numpy.float32
```

## 4. 单个 PKL 的数据结构

完整样本结构为：

```python
{
    "schema_version": "drivaerml-normalized-v1",
    "sample_id": 1,
    "source_run": "run_1",
    "complete": True,
    "missing_required_files": (),

    "x_surface": np.ndarray,                # (100000, 3)
    "y_spressure": np.ndarray,              # (100000, 1)
    "x_volume": np.ndarray,                 # (100000, 3)
    "y_velocity": np.ndarray,               # (100000, 3)
    "y_wss": np.ndarray,                    # (100000, 3)
    "volume_cell_totalpcoeff": np.ndarray,  # (100000, 1)

    "normalization": {
        "x_surface": "global_shared_minmax",
        "x_volume": "global_shared_minmax",
        "y_spressure": "global_componentwise_mean_std",
        "y_velocity": "global_componentwise_mean_std",
        "y_wss": "global_componentwise_mean_std",
        "volume_cell_totalpcoeff": "global_componentwise_mean_std",
    },
}
```

## 5. 完整样本、部分样本和空目录

原始数据并非所有 run 都包含完整的 6 个 NPY 文件。

### 5.1 完整样本

完整样本包含表面和体积数据：

```python
sample["complete"] is True
sample["missing_required_files"] == ()
```

总数为 465。

### 5.2 部分样本

19 个样本只有：

```text
x_surface
y_spressure
y_wss
```

例如：

```python
sample["complete"] is False
sample["missing_required_files"] == (
    "volume_cell_position.npy",
    "volume_cell_velocity.npy",
)
```

脚本不会伪造缺失的体积数据，也不会用零数组填充，而是保留已有表面数据并记录缺失字段。

### 5.3 空目录

16 个原始目录完全没有 NPY 数据，因此不会生成空的 PKL 样本。其目录名称保存在：

```python
normalization_scalars["empty_runs"]
```

## 6. 坐标归一化

`x_surface` 和 `x_volume` 属于同一个物理坐标系，因此二者共同统计一套全局坐标范围：

```python
coords_min
coords_max
coords_range = coords_max - coords_min
```

归一化公式为：

```text
x_normalized = (x - coords_min) / coords_range
```

数学形式：

\[
\hat{\boldsymbol x}
=
\frac{\boldsymbol x-\boldsymbol x_{\min}}
{\boldsymbol x_{\max}-\boldsymbol x_{\min}}
\]

这里的最小值和最大值是按 XYZ 三个分量分别统计的，不是每个样本单独归一化。

当前数据统计为：

```text
coords_min   = [-39.673187, -21.620352, -0.31757465]
coords_max   = [ 79.66567,   21.620361, 19.329884]
coords_range = [119.33886,   43.240715, 19.647459]
```

全数据集复核结果为：

```text
归一化坐标全局最小值：[0, 0, 0]
归一化坐标全局最大值：[1, 1, 1]
```

### 6.1 坐标反归一化

```python
x_physical = x_normalized * coords_range + coords_min
```

## 7. 物理量归一化

每一种物理量单独计算全数据集统计量：

```text
y_spressure
y_velocity
y_wss
volume_cell_totalpcoeff
```

三维速度和三维 WSS 按分量分别统计均值和标准差。例如速度统计为：

```python
velocity_mean.shape == (3,)
velocity_std.shape == (3,)
```

归一化公式为：

```text
y_normalized = (y - mean) / std
```

数学形式：

\[
\hat{\boldsymbol y}
=
\frac{\boldsymbol y-\boldsymbol\mu_y}
{\boldsymbol\sigma_y}
\]

如果某一分量的标准差非常接近零，脚本会将该分量的标准差设置为 1，避免除零。

### 7.1 物理量反归一化

```python
y_physical = y_normalized * std + mean
```

例如压力：

```python
pressure = (
    sample["y_spressure"]
    * scalars["y_spressure_std"]
    + scalars["y_spressure_mean"]
)
```

速度：

```python
velocity = (
    sample["y_velocity"]
    * scalars["y_velocity_std"]
    + scalars["y_velocity_mean"]
)
```

## 8. 两遍处理逻辑

脚本不能读取一个样本后立即归一化，因为全局均值、标准差和坐标范围只有遍历整个数据集后才能确定。

因此采用两遍处理。

### 第一遍：数据审计和统计

对每个 `run_*`：

1. 查找所有规定的 NPY 文件；
2. 检查数组形状；
3. 检查 NaN 和 Inf；
4. 检查表面数据节点数是否一致；
5. 检查体积数据节点数是否一致；
6. 统计所有表面和体积坐标的全局 min/max；
7. 使用稳定的在线算法统计各物理量 mean/std；
8. 记录完整、部分和空样本。

物理量统计使用组合方差算法，而不是简单累计 `sum(x²)` 后相减，可减少大规模数据统计中的浮点误差。

### 第二遍：归一化和写入

获得全局统计量后，再次遍历所有非空样本：

1. 读取 NPY；
2. 标量转换成 `(N,1)`；
3. 坐标执行 min-max；
4. 物理量执行 mean-std；
5. 转换成 `float32`；
6. 写入临时 PKL；
7. 原子替换成最终文件；
8. 重新打开 PKL，验证字段、类型和有限性。

原子写入可以避免程序中断时留下一个只有部分内容、但文件名看起来正常的 PKL。

## 9. normalization_scalars.pkl

文件位于：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000/normalization_scalars.pkl
```

主要字段包括：

```text
schema_version
coords_min
coords_max
coords_range
coordinate_fields
physical_fields
field_mapping
normalization

y_spressure_mean
y_spressure_std
y_spressure_count

y_velocity_mean
y_velocity_std
y_velocity_count

y_wss_mean
y_wss_std
y_wss_count

volume_cell_totalpcoeff_mean
volume_cell_totalpcoeff_std
volume_cell_totalpcoeff_count

samples
empty_runs
num_run_directories
num_converted_samples
num_complete_samples
num_partial_samples
```

其中 `samples` 保存每个非空样本的：

```text
run_id
source_run
available_files
missing_required_files
complete
surface_points
volume_points
```

## 10. 使用方法

### 10.1 重要前提

当前数据目录中的原始 `run_*` 已经删除，因此不能直接在当前目录重新运行转换。

如果以后重新下载或恢复原始数据，需要保证目录类似：

```text
some_original_directory/
├── run_1/
├── run_2/
├── ...
└── run_500/
```

### 10.2 保留原始 run 目录进行转换

推荐先使用单独的输出目录：

```bash
cd /Users/zhangbojun/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerML.py \
  --data-dir /path/to/original/drivaerML_N100_000 \
  --output-dir /path/to/output/drivaerML_pkl
```

这种模式不会修改或删除原始 NPY 数据。

### 10.3 直接在数据根目录生成 PKL

不指定 `--output-dir` 时，PKL 会直接写入 `--data-dir`：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerML.py \
  --data-dir /path/to/original/drivaerML_N100_000
```

执行完成后，PKL 和 `run_*` 会暂时共存。

### 10.4 覆盖已有 PKL

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerML.py \
  --data-dir /path/to/original/drivaerML_N100_000 \
  --overwrite
```

不加 `--overwrite` 时，如果目标 PKL 已存在，程序会停止，防止意外覆盖。

### 10.5 转换完成后删除 run 目录

只有明确不再需要原始 NPY 时才使用：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerML.py \
  --data-dir /path/to/original/drivaerML_N100_000 \
  --overwrite \
  --remove-source-runs
```

`--remove-source-runs` 会永久删除所有匹配的 `run_*` 目录，不能通过脚本恢复。

删除只会在以下步骤全部成功后执行：

1. 所有非空 run 转换完成；
2. 所有输出 PKL 可以重新打开；
3. NPY 字段和 PKL 字段一一对应；
4. `normalization_scalars.pkl` 写入并验证；
5. 所有预期输出文件都存在。

## 11. 查看命令行帮助

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerML.py \
  --help
```

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-dir` | 当前 DrivAerML 路径 | 原始 `run_*` 所在目录 |
| `--output-dir` | 与 `data-dir` 相同 | PKL 输出目录 |
| `--overwrite` | 关闭 | 覆盖已有 PKL |
| `--remove-source-runs` | 关闭 | 验证成功后永久删除 `run_*` |

## 12. 读取样本

```python
import pickle

sample_path = (
    "/Users/zhangbojun/ML_Turbulent/preprocess/"
    "drivaerML_N100_000/1.pkl"
)

with open(sample_path, "rb") as file:
    sample = pickle.load(file)

print(sample["complete"])
print(sample["x_surface"].shape)
print(sample["y_spressure"].shape)
print(sample["x_volume"].shape)
print(sample["y_velocity"].shape)
print(sample["y_wss"].shape)
```

完整样本输出类似：

```text
True
(100000, 3)
(100000, 1)
(100000, 3)
(100000, 3)
(100000, 3)
```

## 13. 读取归一化参数

```python
import pickle

scalars_path = (
    "/Users/zhangbojun/ML_Turbulent/preprocess/"
    "drivaerML_N100_000/normalization_scalars.pkl"
)

with open(scalars_path, "rb") as file:
    scalars = pickle.load(file)

print(scalars["coords_min"])
print(scalars["coords_max"])
print(scalars["y_spressure_mean"])
print(scalars["y_spressure_std"])
print(scalars["y_velocity_mean"])
print(scalars["y_velocity_std"])
```

## 14. 训练时筛选完整样本

如果模型同时使用表面和体积数据，必须排除部分样本：

```python
import pickle
from pathlib import Path

data_dir = Path(
    "/Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000"
)

complete_files = []
for sample_path in data_dir.glob("[0-9]*.pkl"):
    with sample_path.open("rb") as file:
        sample = pickle.load(file)
    if sample["complete"]:
        complete_files.append(sample_path)

complete_files.sort(key=lambda path: int(path.stem))
print(len(complete_files))  # 465
```

更高效的方式是直接读取 `normalization_scalars.pkl` 中的样本清单：

```python
complete_ids = [
    record["run_id"]
    for record in scalars["samples"]
    if record["complete"]
]
```

如果只训练表面压力或壁面剪切应力，可以使用所有 484 个非空样本。

## 15. 组装模型输入和输出

### 15.1 表面压力预测

```python
x = sample["x_surface"]       # (100000, 3)
y = sample["y_spressure"]     # (100000, 1)
```

### 15.2 表面 WSS 预测

```python
x = sample["x_surface"]       # (100000, 3)
y = sample["y_wss"]           # (100000, 3)
```

### 15.3 体积速度预测

```python
x = sample["x_volume"]        # (100000, 3)
y = sample["y_velocity"]      # (100000, 3)
```

体积速度训练必须使用 `complete=True` 的样本。

## 16. 注意事项

1. 坐标采用全数据集统一 min-max，不是每辆车独立归一化。
2. `x_surface` 和 `x_volume` 共用坐标标尺，能够保持二者的相对空间关系。
3. 不同物理量分别计算 mean/std，不能使用压力统计量反归一化速度。
4. 三维速度和 WSS 均为逐分量标准化。
5. 部分样本不能直接参与表面和体积联合训练。
6. `--remove-source-runs` 是不可恢复操作，只应在完整验证后使用。
7. 当前目录已经没有原始 NPY；若需要重新生成，必须重新下载或恢复原始 `run_*` 数据。

