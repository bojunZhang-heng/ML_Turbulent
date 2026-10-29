# DrivAerStar 数据集 PKL 转换说明

本文档对应脚本：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/process_DrivAerStar.py
```

脚本将 DrivAerStar 的 EnSight Gold CFD 文件转换为逐案例 PKL，并执行：

- 从 EnSight 网格提取表面压力、壁面剪切应力和体积速度；
- 使用单元中心作为物理量对应的坐标；
- 坐标按全数据集 XYZ 分量进行 min-max 归一化；
- 物理量按全数据集、逐分量进行 mean-std 归一化；
- 保存统一的 `normalization_scalars.pkl`；
- 检查坐标与物理量是否逐单元对齐；
- 写入后重新读取并验证每个 PKL；
- 可选在全部转换成功后保留少量原始案例，并删除其余批次目录。

## 1. 输入目录和格式

默认数据目录为：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar
```

本次已经实际完成 6 个案例的转换。当前目录为：

```text
DrivAerStar/
├── 1301.pkl
├── 1302.pkl
├── 1304.pkl
├── 1305.pkl
├── 3210.pkl
├── 3211.pkl
└── normalization_scalars.pkl
```

本次转换统计：

```text
发现案例：6
完整案例：6
不完整案例：0
样本 PKL：6
目录大小：约 1.5 GB
```

这 6 个本地测试案例是在“保留原始案例”功能加入前处理的，因此原始 `case_E_batch_*` 已全部删除。新脚本在远端处理数据时会默认把前 10 个完整案例保存在 `raw_data/`，用于拓扑可视化。

原始案例采用 EnSight Gold 格式，例如：

```text
DrivAerStar/
└── case_E_batch_399/
    └── case_E/
        └── 01301/
            └── case/
                ├── E_01301.case
                ├── E_01301.geo
                ├── E_01301.Pressure
                ├── E_01301.Velocity
                └── E_01301.WallShearStress
```

脚本递归搜索所有 `.case` 文件，因此批次编号和案例目录的层数可以不同；但同一个案例的 `.case`、`.geo` 和三个物理量文件必须在同一目录、使用相同文件前缀。

## 2. 为什么使用单元中心坐标

`.case` 文件将三个物理量声明为 `per element`：

```text
scalar per element: Pressure ...
vector per element: Velocity ...
vector per element: WallShearStress ...
```

这表示物理量属于网格单元，而不是网格顶点。脚本因此通过 `vtkCellCenters` 为每个单元计算一个中心坐标。这样可以保证：

```text
x_surface[i]  <-> y_spressure[i] <-> y_wss[i]
x_volume[i]   <-> y_velocity[i]
```

如果直接使用网格顶点坐标，顶点数量通常与单元物理量数量不同，会破坏一一对应关系。

## 3. 表面和体积区域的选择

### 3.1 汽车表面

一个 EnSight 案例包含车身、车轮、风洞边界和接口等多个 part。脚本仅选取同时满足以下条件的 part：

- 单元维度为 2D；
- 包含 `Pressure`；
- 包含 `WallShearStress`；
- 名称不以 `Block.` 开头；
- 名称不包含接口标志 `[交界面`。

这会保留车身、底盘、车轮、后视镜等汽车表面，排除入口、出口、风洞壁面等外部边界。

### 3.2 流场体积

体积数据选择：

- 单元维度为 3D；
- 包含 `Velocity`。

每个 part 在合并后的起止索引都会保存到 `metadata`，便于后处理时恢复 part 的范围。

## 4. PKL 字段

每个案例生成一个以数字 ID 命名的文件，例如：

```text
E_01301.case -> 1301.pkl
E_03210.case -> 3210.pkl
```

字段如下：

| PKL 字段 | 形状 | 含义 | 归一化 |
|---|---:|---|---|
| `x_surface` | `(Ns, 3)` | 汽车表面单元中心 | 全局共享 min-max |
| `y_spressure` | `(Ns, 1)` | 表面压力 | 全局 mean-std |
| `y_wss` | `(Ns, 3)` | 壁面剪切应力 XYZ | 全局逐分量 mean-std |
| `x_volume` | `(Nv, 3)` | 体网格单元中心 | 全局共享 min-max |
| `y_velocity` | `(Nv, 3)` | 速度 XYZ | 全局逐分量 mean-std |

所有数组均保存为 `numpy.float32`。

单个 PKL 的结构为：

```python
{
    "schema_version": "drivaerstar-normalized-v1",
    "sample_id": 1301,
    "x_surface": np.ndarray,
    "y_spressure": np.ndarray,
    "y_wss": np.ndarray,
    "x_volume": np.ndarray,
    "y_velocity": np.ndarray,
    "metadata": {
        "source_case": ".../E_01301.case",
        "surface_parts": [...],
        "volume_parts": [...],
        "surface_cells": Ns,
        "volume_cells": Nv,
        "coordinate_semantics": "cell_centers",
    },
    "normalization": {...},
}
```

## 5. 归一化逻辑

脚本采用两遍处理：

1. 第一遍读取所有完整案例，统计全数据集归一化参数；
2. 第二遍重新读取案例，归一化并写入 PKL。

因此每个样本使用同一套统计量，不会发生“每辆车各自归一化”导致绝对尺度丢失的问题。

### 5.1 坐标 min-max

`x_surface` 与 `x_volume` 位于相同物理坐标系，共享同一套 XYZ 范围：

```python
x_norm = (x_raw - coords_min) / coords_range
coords_range = coords_max - coords_min
```

三个分量分别归一化。正常情况下，全数据集归一化坐标范围接近 `[0, 1]`。

### 5.2 物理量 mean-std

压力、速度和 WSS 各自独立统计：

```python
y_norm = (y_raw - y_mean) / y_std
```

向量物理量按 XYZ 分量分别统计。例如 `y_velocity_mean` 和 `y_velocity_std` 的形状均为 `(3,)`。

### 5.3 反归一化

模型输出需要恢复到物理尺度后再计算 Pa、m/s 等物理指标：

```python
import pickle

with open("normalization_scalars.pkl", "rb") as f:
    scalars = pickle.load(f)

pressure = pressure_norm * scalars["y_spressure_std"] \
           + scalars["y_spressure_mean"]

velocity = velocity_norm * scalars["y_velocity_std"] \
           + scalars["y_velocity_mean"]

wss = wss_norm * scalars["y_wss_std"] \
      + scalars["y_wss_mean"]

x_surface = x_surface_norm * scalars["coords_range"] \
            + scalars["coords_min"]
```

## 6. normalization_scalars.pkl

该文件包含：

```text
coords_min / coords_max / coords_range
y_spressure_mean / y_spressure_std / y_spressure_count
y_velocity_mean / y_velocity_std / y_velocity_count
y_wss_mean / y_wss_std / y_wss_count
samples
num_samples
num_discovered_cases
num_complete_cases
num_incomplete_cases
incomplete_cases
field_mapping
normalization
```

`samples` 还保存每个案例的表面/体积单元数量与 part 索引范围。

本次全数据统计得到：

```text
coords_min   = [-14.880000, -14.940001, -0.405689]
coords_max   = [ 44.880001,  14.940001, 14.807159]
coords_range = [ 59.760002,  29.880001, 15.212848]

y_spressure_mean = [-139.510696]
y_spressure_std  = [ 280.012207]

y_velocity_mean = [29.446531, -0.057177, -0.396590]
y_velocity_std  = [14.927553,  3.965619,  3.973519]

y_wss_mean = [1.109495, -0.001293, 0.007897]
y_wss_std  = [2.273741,  0.939913, 2.686800]
```

对全部输出 PKL 重新聚合后的检查结果为：

```text
y_spressure mean ≈ 0，std ≈ 1
y_velocity  mean ≈ 0，std ≈ 1
y_wss       mean ≈ 0，std ≈ 1
```

## 7. 依赖

推荐在 `pgd-no` 环境中运行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python -c \
  "import vtk, numpy; print(vtk.vtkVersion.GetVTKVersion())"
```

核心依赖为：

```text
numpy
vtk
```

## 8. 使用方法

先进入工程目录：

```bash
cd /Users/zhangbojun/ML_Turbulent
```

### 8.1 默认转换并保留原文件

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py
```

默认把编号 PKL 和 `normalization_scalars.pkl` 写到 `DrivAerStar` 根目录。原始 EnSight 文件不会被删除。

### 8.2 覆盖已有输出

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --overwrite
```

脚本总是从原始 EnSight 文件重新计算统计量，因此不会对旧 PKL 中已经归一化的数据再次归一化。

### 8.3 指定输入和输出目录

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir /path/to/DrivAerStar \
  --output-dir /path/to/pkl_output
```

### 8.4 保留 10 个原始案例并清理其余原始目录

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --overwrite \
  --remove-source-directories
```

`--keep-source-cases` 的默认值为 10，因此上面的命令等价于：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --overwrite \
  --remove-source-directories \
  --keep-source-cases 10
```

脚本按案例数字 ID 升序保留前 10 个完整案例。假设远端共有 1000 个完整案例，最终目录类似：

```text
DrivAerStar/
├── 1.pkl
├── 2.pkl
├── ...
├── 1000.pkl
├── normalization_scalars.pkl
└── raw_data/
    ├── 00001/
    │   ├── E_00001.case
    │   ├── E_00001.geo
    │   ├── E_00001.Pressure
    │   ├── E_00001.Velocity
    │   └── E_00001.WallShearStress
    ├── 00002/
    └── ...
```

训练仍使用全部 1000 个 PKL。`raw_data/` 中的 10 个案例用于把对应预测写回原始网格，并通过 ParaView 等软件可视化。

如需改变保留数量，例如保留 20 个：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --overwrite \
  --remove-source-directories \
  --keep-source-cases 20
```

设置为 `0` 表示不保留任何原始案例：

```bash
--keep-source-cases 0
```

清理未保留的案例是不可恢复操作。脚本只有在以下条件全部满足后才删除 `case_E_batch_*`：

- 所有案例的必要文件完整；
- 所有样本已经写入；
- 每个 PKL 已重新读取并通过字段、形状、类型和有限值检查；
- `normalization_scalars.pkl` 已通过验证。

只要发现一个不完整案例，脚本就拒绝删除任何批次目录。

保留案例的 ID 和新 `.case` 路径同时写入：

```python
scalars["retained_source_case_ids"]
scalars["retained_source_cases"]
```

对应样本 PKL 的 `metadata["visualization_case"]` 也会记录 `.case` 文件位置。

## 9. 读取训练样本

```python
import pickle

with open("/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar/1301.pkl", "rb") as f:
    sample = pickle.load(f)

x_surface = sample["x_surface"]
y_pressure = sample["y_spressure"]
y_wss = sample["y_wss"]
x_volume = sample["x_volume"]
y_velocity = sample["y_velocity"]

print(x_surface.shape, y_pressure.shape, y_wss.shape)
print(x_volume.shape, y_velocity.shape)
```

表面压力任务可使用：

```text
input  = x_surface
target = y_spressure
```

体积速度任务可使用：

```text
input  = x_volume
target = y_velocity
```

## 10. 后处理与原网格映射

数组采用“按选中 part 的顺序依次拼接”的规则。每个 part 的信息保存在：

```python
sample["metadata"]["surface_parts"]
sample["metadata"]["volume_parts"]
```

其中每条记录包含 `block_index`、`name`、`start`、`end` 和 `cells`。例如：

```python
for part in sample["metadata"]["surface_parts"]:
    part_prediction = pressure[part["start"]:part["end"]]
```

这使预测结果可以重新拆分到原 EnSight part。需要写回可视化文件时，应先反归一化，再依据 `block_index` 和 `[start:end]` 将结果赋给对应 part 的 cell data。只有 `retained_source_case_ids` 中的案例保留了原始拓扑，可以直接执行这种后处理。

## 11. 注意事项

- 单个体网格可能超过一千万个单元，PKL 会达到数百 MB；训练前通常需要体积下采样。
- 不要在 dataloader 中再次归一化这些 PKL。
- 数据量较大时，两遍读取是预期行为：第一遍统计，第二遍转换。
- `source_case` 是追踪信息；移动数据后其旧绝对路径不影响训练数组。
- 如需计算汽车阻力系数，还需要自由来流密度、来流速度、参考面积、表面法向量和表面积权重；当前 PKL 尚未保存法向量与面积。
