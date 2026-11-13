# DrivAerML 瞬态 OpenFOAM 数据前处理

对应脚本：

```text
preprocess/process_DrivAerML_transient.py
```

该脚本参考 `ML_preprocess.py`，直接读取 `run_3` 的 OpenFOAM
`constant/polyMesh` 和各时间目录中的 `p/U`，生成适合神经网络读取的编号 PKL。
源 OpenFOAM 文件始终只读，不会删除或修改。

## 1. 输出内容

每个有效时间步生成一个编号文件：

```text
pklFile_transient/
├── 1.pkl
├── 2.pkl
├── ...
└── normalization_scalars.pkl
```

编号按照时间值从小到大排序。真实时间保存在 PKL 的 `time_name` 和
`time_value` 中。例如当前测试目录实际是 `1` 和 `1.005`，因此：

```text
1.pkl -> time_name = "1"
2.pkl -> time_name = "1.005"
```

每个样本包含：

```python
{
    "x_surface":   ...,  # (Ns, 3)，汽车表面边界面面心
    "y_spressure": ...,  # (Ns, 1)，对应表面压力
    "x_volume":    ...,  # (Nv, 3)，体单元中心
    "y_velocity":  ...,  # (Nv, 3)，对应体场速度
    "time_name":   ...,
    "time_value":  ...,
}
```

表面和体场分别严格对齐：

```text
x_surface[i] <-> y_spressure[i]
x_volume[i]  <-> y_velocity[i]
```

## 2. 表面点和压力如何获得

脚本读取 `constant/polyMesh/boundary`，默认选择：

- `type wall`；
- 不以 `domain_` 开头；
- 不以 `CTRL_SURFACE_` 开头。

因此保留车身、车轮、底盘、后视镜等汽车壁面，排除道路、远场和控制面。

`x_surface` 使用所选边界面的面心。当前汽车 wall 上的 `p` 通常采用
`zeroGradient`，所以 `y_spressure` 使用该边界面的 owner cell 压力。这保证
表面坐标和压力具有完全一致的面顺序。

如需改变汽车部件范围，可以修改正则表达式：

```bash
--include-patch-regex '^(Body_|Mirrors_|Notchback_)' \
--exclude-patch-regex '^(domain_|CTRL_SURFACE_)'
```

## 3. 归一化

所有时间步共同统计一套参数。

### 坐标

面场和体场统一使用汽车表面尺度：

```text
x_norm = (x - surface_coords_min) / surface_coords_range
```

所以 `x_surface` 位于 `[0,1]`；远场体单元的 `x_volume` 可以小于 0 或大于
1，这是使用汽车尺度描述外流场位置的正常结果。

### 物理量

压力和速度分别使用所有时间步的全局 mean/std：

```text
p_norm = (p - y_spressure_mean) / y_spressure_std
U_norm = (U - y_velocity_mean) / y_velocity_std
```

速度的三个分量分别统计。反归一化为：

```text
p = p_norm * y_spressure_std + y_spressure_mean
U = U_norm * y_velocity_std + y_velocity_mean
```

## 4. 先执行检查

当前本地测试时间目录名是 `1` 和 `1.005`，不是 `1.0005`：

```bash
python preprocess/process_DrivAerML_transient.py \
  --case-dir preprocess/run_3 \
  --time 1 1.005 \
  --check-only
```

检查只读取文件头、数量和 boundary，不生成点云。当前测试 case 的检查结果为：

```text
cells             = 143,870,657
faces             = 442,300,549
vehicle wall faces= 8,808,998
times             = 1, 1.005
```

## 5. 正式处理

macOS/Linux：

```bash
python preprocess/process_DrivAerML_transient.py \
  --case-dir preprocess/run_3 \
  --output-dir preprocess/run_3/pklFile_transient \
  --time 1 1.005
```

处理所有具有非均匀 `p/U internalField` 的时间目录：

```bash
python preprocess/process_DrivAerML_transient.py \
  --case-dir preprocess/run_3 \
  --output-dir preprocess/run_3/pklFile_transient
```

Windows PowerShell：

```powershell
python preprocess\process_DrivAerML_transient.py `
  --case-dir preprocess\run_3 `
  --output-dir preprocess\run_3\pklFile_transient `
  --time 1 1.005
```

已有输出时需要显式覆盖：

```bash
--overwrite
```

## 6. 大网格缓存

当前网格约 1.44 亿体单元、4.42 亿个面，无法整体装入普通内存。脚本使用：

- `numpy.memmap` 分块读取二进制网格；
- 磁盘上的 cell-center 累加缓存；
- 分块解析超大的 ASCII `p/U`；
- 分块统计 mean/std；
- 分块归一化和校验。

默认缓存目录：

```text
pklFile_transient/.cache
```

几何与时间无关。以后新增时间步时，只要 polyMesh 未变化，脚本会复用
`x_surface_raw.npy`、`x_volume_raw.npy` 和表面 owner 映射，不再重新计算
4.42 亿个面的几何信息。

如果 polyMesh 已更换，脚本会根据文件大小和修改时间自动重建。也可强制重建：

```bash
--rebuild-geometry
```

## 7. 磁盘空间

该 case 单个时间步大约包含：

- `x_volume`：约 1.7 GB（float32）；
- `y_velocity`：约 1.7 GB；
- 表面坐标和压力：约 140 MB；
- 最终单个 PKL：约 3.5 GB 以上。

处理过程中还要保存几何和原始物理量缓存。正式运行前应预留远大于最终 PKL
总量的磁盘空间。缓存用于避免重复计算，不建议在全部时间步处理完成前删除。

## 8. normalization_scalars.pkl

关键字段包括：

```text
coords_min / coords_max / coords_range
surface_coords_min / surface_coords_max / surface_coords_range
volume_coords_min / volume_coords_max / volume_coords_range
volume_physical_coords_min / volume_physical_coords_max
y_spressure_mean / y_spressure_std
y_velocity_mean / y_velocity_std
samples
surface_patches
```

为兼容现有 DrivAerML/DrivAerStar dataloader，`volume_coords_*` 也保存面场尺度；
体场自身的物理边界另外保存在 `volume_physical_coords_*`。

## 9. 与 ML_preprocess.py 的区别

`ML_preprocess.py` 输出三个文本文件：位置、压力和速度，并且压力也是完整
cell field。新脚本直接生成 PKL，并把任务拆分为：

- 汽车表面面心 -> 表面压力；
- 全部体单元中心 -> 体场速度。

新脚本还跨时间统一统计归一化参数，并保存时间到编号 PKL 的映射。
