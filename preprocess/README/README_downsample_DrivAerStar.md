# downsample_DrivAerStar.py 使用说明

该脚本读取 `process_DrivAerStar.py` 生成的全车 PKL，并将表面点、体场点下采样后仍写入**同一个 PKL**。
它不会对压力、WSS 或速度再次归一化。

## 1. 对齐关系

表面场共用一组索引：

```text
surface_downsample_indices
x_surface[index] <-> y_spressure[index] <-> y_wss[index]
```

体场共用另一组索引：

```text
volume_downsample_indices
x_volume[index] <-> y_velocity[index]
```

两组索引都会保存到输出 PKL，用于把预测结果映射回原始全量数组。表面和体场不会共用同一组索引。

## 2. 同一个 PKL 分别指定表面和体场点数

推荐用法：表面保留 10,000 点、体场保留 100,000 点：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerStar.py \
  --input-dir /path/to/DrivAerStar_full_v2 \
  --output-dir /path/to/DrivAerStar_Ns10000_Nv100000 \
  --point-set both \
  --surface-num-points 10000 \
  --volume-num-points 100000 \
  --method random \
  --overwrite
```

输出 `1.pkl`、`2.pkl` 等文件中同时包含：

```text
x_surface, y_spressure, y_wss
x_volume, y_velocity
surface_downsample_indices
volume_downsample_indices
```

## 3. 只处理表面场

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerStar.py \
  --input-dir /path/to/DrivAerStar_full_v2 \
  --output-dir /path/to/DrivAerStar_surface_N10000 \
  --point-set surface \
  --surface-num-points 10000 \
  --method voxel \
  --overwrite
```

只处理表面时，输出会移除巨大的体场数组，以减少磁盘占用。

## 4. 只处理体场

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerStar.py \
  --input-dir /path/to/DrivAerStar_full_v2 \
  --output-dir /path/to/DrivAerStar_volume_N100000 \
  --point-set volume \
  --volume-num-points 100000 \
  --method random \
  --overwrite
```

## 5. 兼容旧命令

`--num-points` 仍然支持。当选 `both` 时，它会对表面和体场使用相同点数：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerStar.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --point-set both \
  --num-points 10000
```

如果同时指定通用参数和专用参数，专用参数优先。例如：

```text
--num-points 10000 --volume-num-points 100000
```

表示 surface 为 10,000 点、volume 为 100,000 点。

## 6. random 与 voxel

- `random`：速度快、临时内存低，推荐用于百万级或千万级体场；
- `voxel`：空间覆盖更均匀，适合汽车表面；大规模体场会更慢并占用更多内存。

当前一次运行只能选择一种方法。如果需要表面使用 voxel、体场使用 random，可分两次执行到不同目录；
若必须保存在同一 PKL，建议统一使用 random，或者后续扩展为每组单独的 method 参数。

## 7. 坐标归一化行为

- 输入为 v2：直接继承 `surface_coords_*` 和 `volume_coords_*`，不会重新归一化；
- 输入为旧 v1：先用共享 `coords_min/coords_range` 恢复物理坐标，再分别统计 surface/volume 范围并生成 v2；
- 物理量在两种情况下都不会再次 mean-std。

输出 `normalization_scalars.pkl` 会保存两套独立坐标尺度，以及：

```python
scalars["downsampling"] = {
    "source_directory": "...",
    "point_sets": ["surface", "volume"],
    "num_points": {"surface": 10000, "volume": 100000},
    "method": "random",
    "seed": 42,
}
```

## 8. 可重复性与映射

同一 `seed`、sample ID、采样方法和目标点数会产生相同索引。模型输出可按以下方式放回全量数组：

```python
full_prediction = np.full((original_num_points, channels), np.nan)
full_prediction[sample["surface_downsample_indices"]] = surface_prediction
```

未采样位置保持 `NaN`，可在原网格后处理中显示采样预测，或另外执行插值。
