# DrivAerML / DrivAerStar 表面尺度坐标转换

以下两个脚本把汽车表面和体场坐标统一到同一套、全数据集的表面坐标 min-max 尺度：

```text
process_DrivAerML_surface_scale.py
process_DrivAerStar_surface_scale.py
```

转换公式为：

```text
x_surface_new = (x_surface_physical - surface_min) / surface_range
x_volume_new  = (x_volume_physical  - surface_min) / surface_range
```

由于外流场范围大于汽车表面范围，`x_volume_new` 超出 `[0,1]` 是正确且预期的结果。脚本不会裁剪这些坐标。

脚本读取现有编号 PKL，不需要已经删除的原始 NPY 或 EnSight 文件。它兼容：

- 已经分别使用 `surface_coords_*`、`volume_coords_*` 的数据；
- 只有旧版共享 `coords_*` 统计量的数据。

压力、速度、WSS、总压系数、下采样索引及其他 NumPy 数组均原样复制，并逐文件验证没有发生变化。输出必须使用新目录，脚本不会原地覆盖输入数据。

## DrivAerML

```bash
python preprocess/process_DrivAerML_surface_scale.py \
  --input-dir preprocess/drivaerML_N10k \
  --output-dir preprocess/drivaerML_N10k_surface_scale
```

## DrivAerStar

```bash
python preprocess/process_DrivAerStar_surface_scale.py \
  --input-dir preprocess/DrivAerStar \
  --output-dir preprocess/DrivAerStar_surface_scale
```

目标目录已有文件时，显式加入：

```text
--overwrite
```

新的 `normalization_scalars.pkl` 会把以下三组兼容键全部设为表面尺度：

```text
coords_min/max/range
surface_coords_min/max/range
volume_coords_min/max/range
```

旧坐标统计保存在：

```text
source_coordinate_scalars
```

模型 DataLoader 因此仍能通过 `volume_coords_min/range` 正确恢复体场真实坐标。
