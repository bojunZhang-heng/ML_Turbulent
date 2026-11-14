# DrivAerML 瞬态 OpenFOAM 数据前处理

对应脚本：`preprocess/process_DrivAerML_transient.py`。

脚本直接读取 OpenFOAM `constant/polyMesh` 以及各时间目录中的 `p/U`。解析
逻辑已集成在脚本内，不依赖 `ML_preprocess.py`。源文件只读，不会被修改或删除。

## 1. 输出结构

所有时间步共用相同网格，因此坐标只保存一次：

```text
pklFile_transient/
├── geometry.pkl
├── normalization_scalars.pkl
├── 1.pkl
├── 2.pkl
└── ...
```

`geometry.pkl`：

```python
{
    "x_surface": ...,  # (Ns, 3)，归一化汽车表面边界面面心
    "x_volume":  ...,  # (Nv, 3)，归一化体单元中心
    "surface_patches": ...,
}
```

每个编号 PKL 只保存一个时间步的物理场，不重复保存坐标：

```python
{
    "sample_id": 1,
    "time_name": "1",
    "time_value": 1.0,
    "y_spressure": ...,  # (Ns, 1)
    "y_velocity":  ...,  # (Nv, 3)
    "geometry_file": "geometry.pkl",
    "normalization_scalars_file": "normalization_scalars.pkl",
}
```

编号按真实时间排序。例如时间目录为 `1` 和 `1.005`：

```text
1.pkl -> time_name = "1"
2.pkl -> time_name = "1.005"
```

`normalization_scalars.pkl` 保存全部时间共同统计的归一化参数，以及编号和真实
时间的映射。`samples` 中每条记录包含 `sample_id`、`output_file`、
`time_name`、`time_value`、表面点数和体场点数。

## 2. 数据对应关系

```text
geometry["x_surface"][i] <-> timestep["y_spressure"][i]
geometry["x_volume"][i]  <-> timestep["y_velocity"][i]
```

`x_surface` 是汽车 wall 边界面的面心。汽车壁面压力通常采用
`zeroGradient`，因此 `y_spressure` 使用对应边界面的 owner cell 压力。

默认选择所有 `type wall`，但排除名称以 `domain_` 和 `CTRL_SURFACE_` 开头的
道路、计算域及控制面。`symmetry` 和 `symmetryPlane` 可以解析，但不会被当成
汽车表面。

如需改变汽车部件范围：

```bash
--include-patch-regex '^(Body_|Mirrors_|Notchback_)' \
--exclude-patch-regex '^(domain_|CTRL_SURFACE_)'
```

## 3. 归一化

所有时间步共同统计一套参数。

面场与体场坐标统一使用汽车表面尺度：

```text
x_norm = (x - surface_coords_min) / surface_coords_range
```

`x_surface` 位于 `[0,1]`。远场 `x_volume` 可以小于 0 或大于 1。

物理量采用全时间 mean/std：

```text
p_norm = (p - y_spressure_mean) / y_spressure_std
U_norm = (U - y_velocity_mean) / y_velocity_std
```

速度三个分量分别统计。反归一化为：

```text
p = p_norm * y_spressure_std + y_spressure_mean
U = U_norm * y_velocity_std + y_velocity_mean
```

## 4. OpenFOAM 格式

时间场支持：

- ASCII `List<scalar>` / `List<vector>`；
- Binary `List<scalar>` / `List<vector>`；
- little-endian / big-endian；
- scalar32 / scalar64。

字段数量严格从 `internalField nonuniform List<...>` 后解析，不会把 Binary
payload 中的随机数字和括号误认为列表头。

`internalField uniform` 没有逐单元结果，会被跳过。并行计算的
`processor*/时间/p、U` 需要先执行 `reconstructPar`。

## 5. 检查与运行

检查时间目录 `1` 和 `1.005`：

```bash
python preprocess/process_DrivAerML_transient.py \
  --case-dir preprocess/run_3 \
  --time 1 1.005 \
  --check-only
```

正式处理指定时间：

```bash
python preprocess/process_DrivAerML_transient.py \
  --case-dir preprocess/run_3 \
  --output-dir preprocess/run_3/pklFile_transient \
  --time 1 1.005
```

不传 `--time` 时处理所有包含非均匀 `p/U internalField` 的时间目录。已有输出
时使用 `--overwrite`。

Windows PowerShell：

```powershell
python preprocess\process_DrivAerML_transient.py `
  --case-dir preprocess\run_3 `
  --output-dir preprocess\run_3\pklFile_transient `
  --time 1 1.005
```

## 6. 训练时读取

```python
import pickle
from pathlib import Path

root = Path("pklFile_transient")

with (root / "geometry.pkl").open("rb") as stream:
    geometry = pickle.load(stream)

with (root / "1.pkl").open("rb") as stream:
    timestep = pickle.load(stream)

x_surface = geometry["x_surface"]
x_volume = geometry["x_volume"]
y_spressure = timestep["y_spressure"]
y_velocity = timestep["y_velocity"]
```

训练开始时只加载一次 `geometry.pkl`；DataLoader 每次只加载需要的时间步。

## 7. 大网格缓存

脚本使用 `numpy.memmap` 分块读取 polyMesh、解析 ASCII/Binary 场、统计
mean/std、归一化及校验。默认缓存目录位于 case 下：

```text
run_3/.process_DrivAerML_transient_cache
```

缓存与输出目录分开。只要 polyMesh 未变化，新增时间步可复用几何缓存。更换
网格后会自动重建，也可使用 `--rebuild-geometry` 强制重建。缓存不是训练数据，
全部转换完成并确认 PKL 正常后可以删除。

## 8. 磁盘空间

约 1.47 亿体单元时，`geometry.pkl` 中的 `x_volume` 约 1.77 GB，每个时间步
的 `y_velocity` 也约 1.77 GB。拆分后不会在每个时间步重复存储体场坐标，能
显著节省磁盘空间；处理期间仍需为原始及归一化 memmap 预留额外空间。
