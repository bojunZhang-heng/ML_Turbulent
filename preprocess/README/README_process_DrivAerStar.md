# process_DrivAerStar.py 使用说明

该脚本把 DrivAerStar 的 EnSight Gold 案例转换成逐案例 PKL。核心原则是：

- 汽车表面场与体场使用两套独立的坐标 min-max 统计量；
- 压力、壁面剪切应力和速度使用全数据集 mean-std；
- 一个 PKL 同时保存表面场和体场；
- 坐标与物理量均按单元中心一一对应。

## 1. 字段和归一化

| 字段 | 形状 | 归一化 |
|---|---:|---|
| `x_surface` | `(Ns, 3)` | surface 专用 min-max |
| `y_spressure` | `(Ns, 1)` | 全数据集 mean-std |
| `y_wss` | `(Ns, 3)` | 全数据集逐分量 mean-std |
| `x_volume` | `(Nv, 3)` | volume 专用 min-max |
| `y_velocity` | `(Nv, 3)` | 全数据集逐分量 mean-std |

两套坐标变换分别为：

```python
x_surface_norm = (x_surface_raw - surface_coords_min) / surface_coords_range
x_volume_norm = (x_volume_raw - volume_coords_min) / volume_coords_range
```

这里的统计范围是整个数据集，不是每辆车单独统计。因此保留了不同车型之间的绝对尺寸差异。

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

脚本写完后会验证上述六个独立坐标统计字段是否齐全。

## 2. 从原始 EnSight 数据处理

在项目根目录运行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir /path/to/raw/DrivAerStar \
  --output-dir /path/to/DrivAerStar_full_v2
```

覆盖已有输出：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir /path/to/raw/DrivAerStar \
  --output-dir /path/to/DrivAerStar_full_v2 \
  --overwrite
```

脚本采用两遍读取：第一遍统计 surface/volume 各自的坐标范围和物理量 mean-std，第二遍归一化并写 PKL。
它不会读取已生成的 PKL 再次归一化。

## 3. 从旧版共享坐标尺度的 PKL 迁移

旧版 `normalization_scalars.pkl` 只包含：

```text
coords_min / coords_max / coords_range
```

如果原始 EnSight 文件已经删除，可用旧统计量恢复物理坐标，再生成独立尺度的 v2：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --source-pkl-dir /path/to/DrivAerStar_v1 \
  --output-dir /path/to/DrivAerStar_full_v2 \
  --overwrite
```

迁移时只重新计算 `x_surface` 和 `x_volume`。已经 mean-std 归一化的
`y_spressure`、`y_wss`、`y_velocity` 原样复制，不会二次归一化。

源目录与输出目录必须不同，以免覆盖唯一可恢复的旧数据。

## 4. 反归一化

```python
x_surface_raw = x_surface * scalars["surface_coords_range"] \
                + scalars["surface_coords_min"]
x_volume_raw = x_volume * scalars["volume_coords_range"] \
               + scalars["volume_coords_min"]

pressure_raw = y_spressure * scalars["y_spressure_std"] \
               + scalars["y_spressure_mean"]
velocity_raw = y_velocity * scalars["y_velocity_std"] \
               + scalars["y_velocity_mean"]
wss_raw = y_wss * scalars["y_wss_std"] + scalars["y_wss_mean"]
```

不要用 surface 的统计量反归一化 volume，也不要反过来使用。

## 5. 原网格保留

如需保留 10 个原始案例用于将预测映射回 EnSight 拓扑：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerStar.py \
  --data-dir /path/to/raw/DrivAerStar \
  --output-dir /path/to/DrivAerStar_full_v2 \
  --remove-source-directories \
  --keep-source-cases 10 \
  --overwrite
```

`--remove-source-directories` 会删除未保留的原始批次，只应在确认输出和备份后使用。
