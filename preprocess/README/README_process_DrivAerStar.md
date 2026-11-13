# process_DrivAerStar.py 使用说明

该脚本把**一种车型目录**下的 DrivAerStar EnSight Gold 案例转换成逐案例 PKL。核心原则是：

- 汽车表面场与体场统一使用汽车表面尺度的坐标 min-max 统计量；
- 压力、壁面剪切应力和速度使用全数据集 mean-std；
- 一个 PKL 同时保存表面场和体场；
- 坐标与物理量均按单元中心一一对应。
- 原始 EnSight 文件只读取，不删除、不移动。

## 1. 字段和归一化

| 字段 | 形状 | 归一化 |
|---|---:|---|
| `x_surface` | `(Ns, 3)` | 全数据集汽车表面 min-max |
| `y_spressure` | `(Ns, 1)` | 全数据集 mean-std |
| `y_wss` | `(Ns, 3)` | 全数据集逐分量 mean-std |
| `x_volume` | `(Nv, 3)` | 与 `x_surface` 共用汽车表面 min-max，可超出 `[0,1]` |
| `y_velocity` | `(Nv, 3)` | 全数据集逐分量 mean-std |

两套坐标变换分别为：

```python
x_surface_norm = (x_surface_raw - surface_coords_min) / surface_coords_range
x_volume_norm = (x_volume_raw - surface_coords_min) / surface_coords_range
```

这里的表面统计范围是当前车型目录中所有案例的汽车表面，不是每辆车单独统计。因此保留了该车型案例之间的绝对尺寸差异。体场远场点使用该尺度后可以自然超出 `[0,1]`，脚本不会裁剪或报错。

`normalization_scalars.pkl` 中会分别保存：

```text
surface_coords_min
surface_coords_max
surface_coords_range
volume_coords_min
volume_coords_max
volume_coords_range
y_spressure_mean / y_spressure_std
y_wss_mean / y_wss_std
y_velocity_mean / y_velocity_std
```

其中 `volume_coords_min/max/range` 是兼容字段，内容与 `surface_coords_min/max/range` 相同；实际物理体场范围另存为 `volume_domain_min/max`。脚本写完后会验证上述坐标统计字段是否齐全。

## 2. 从原始 EnSight 数据处理

在项目根目录运行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir /path/to/raw/DrivAerStar \
  --output-dir /path/to/DrivAerStar_surface_scale
```

当前目录按车型组织时，应一次只传入一个车型目录。例如：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir preprocess/DrivAerStar/case_E \
  --output-dir preprocess/DrivAerStar/case_E_pkl \
  --overwrite
```

不要把包含 `case_E`、`case_F`、`case_N` 的共同父目录作为一次处理输入，因为不同车型可能存在相同的数字案例编号。

覆盖已有输出：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir /path/to/raw/DrivAerStar \
  --output-dir /path/to/DrivAerStar_surface_scale \
  --overwrite
```

脚本采用两遍读取：第一遍统计 surface/volume 各自的坐标范围和物理量 mean-std，第二遍归一化并写 PKL。
它不会读取已生成的 PKL 再次归一化。

## 3. 从旧版 PKL 迁移

旧版 `normalization_scalars.pkl` 只包含：

```text
coords_min / coords_max / coords_range
```

如果原始 EnSight 文件已经删除，可用旧统计量恢复物理坐标，再生成面场和体场统一采用表面尺度的 v3：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --source-pkl-dir /path/to/DrivAerStar_v1 \
  --output-dir /path/to/DrivAerStar_surface_scale \
  --overwrite
```

迁移时只重新计算 `x_surface` 和 `x_volume`。已经 mean-std 归一化的
`y_spressure`、`y_wss`、`y_velocity` 原样复制，不会二次归一化。

源目录与输出目录必须不同，以免覆盖唯一可恢复的旧数据。

## 4. 反归一化

```python
x_surface_raw = x_surface * scalars["surface_coords_range"] \
                + scalars["surface_coords_min"]
x_volume_raw = x_volume * scalars["surface_coords_range"] \
               + scalars["surface_coords_min"]

pressure_raw = y_spressure * scalars["y_spressure_std"] \
               + scalars["y_spressure_mean"]
velocity_raw = y_velocity * scalars["y_velocity_std"] \
               + scalars["y_velocity_mean"]
wss_raw = y_wss * scalars["y_wss_std"] + scalars["y_wss_mean"]
```

不要用 surface 的统计量反归一化 volume，也不要反过来使用。

## 5. 原始文件安全策略

当前脚本已经取消删除原始文件的功能。原始 `.case`、`.geo`、`.Pressure`、`.Velocity` 和 `.WallShearStress` 文件只会被读取，不会被删除或移动。这样可以直接使用原始 EnSight 拓扑进行后处理可视化。

`--overwrite` 只影响 `--output-dir` 中已有的 PKL 和 `normalization_scalars.pkl`，不会影响输入目录。
