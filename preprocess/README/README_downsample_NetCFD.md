# downsample_NetCFD.py 使用说明

对应脚本：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/downsample_NetCFD.py
```

该脚本用于下采样 `process_NetCFD.py` 生成的 DrivAerNet-CFD 完整体场速度 PKL，不会再次归一化坐标或速度。

## 1. 输入格式

默认输入目录：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerNet-CFD/volume_velocity_pklFile
```

单个输入样本：

```python
{
    "x_volume": np.ndarray,         # (1, N, 3)
    "y_velocity": np.ndarray,       # (1, N, 3)
    "source_point_ids": np.ndarray  # 可选，(1, N)
}
```

当前本地样本规模为：

```text
109.pkl: N = 9,409,089
110.pkl: N = 10,109,141
```

## 2. 输出格式

如果设置 `--num-points 100000`，输出为：

```python
{
    "x_volume": ...,                 # (1, 100000, 3)
    "y_velocity": ...,               # (1, 100000, 3)
    "source_point_ids": ...,          # 输入存在时输出，(1, 100000)
    "volume_downsample_indices": ..., # (1, 100000)
    "downsampling": {...},
}
```

同一组索引同时作用于坐标、速度和原始点编号：

```text
x_volume[0, i]
<-> y_velocity[0, i]
<-> source_point_ids[0, i]
```

`volume_downsample_indices` 是采样点在输入全量 PKL 中的编号。即使输入没有 `source_point_ids`，该字段也一定会保存。

## 3. 推荐运行方法

从约一千万点下采样到十万点：

```bash
cd ~/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_NetCFD.py \
  --input-dir preprocess/DrivAerNet-CFD/volume_velocity_pklFile \
  --output-dir preprocess/DrivAerNet-CFD/volume_velocity_pklFile_100k \
  --num-points 100000 \
  --method random \
  --seed 42 \
  --overwrite
```

下采样到一万点：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_NetCFD.py \
  --output-dir preprocess/DrivAerNet-CFD/volume_velocity_pklFile_10k \
  --num-points 10000 \
  --method random \
  --overwrite
```

省略 `--input-dir` 时使用默认输入目录。

## 4. random 与 voxel

### random

- 速度快；
- 临时内存占用低；
- 推荐用于千万级体场；
- 相同 `seed`、样本编号和目标点数会得到相同结果。

### voxel

- 空间覆盖通常更均匀；
- 对千万级点云需要较大的临时内存；
- 处理速度明显慢于 random。

体场训练优先使用：

```text
--method random
```

## 5. normalization_scalars.pkl

输出目录会复制原有归一化统计信息，并增加：

```python
scalars["downsampling"] = {
    "source_directory": "...",
    "point_set": "volume",
    "num_points": 100000,
    "method": "random",
    "seed": 42,
    "physical_fields_renormalized": False,
}
```

以下统计量不会重新计算，因为下采样只是从已经归一化的数据中选点：

```text
coords_min / coords_max / coords_range
surface_coords_min / surface_coords_max / surface_coords_range
y_velocity_mean / y_velocity_std
```

## 6. 映射回完整网格

模型预测形状通常为：

```text
(1, N_sampled, 3)
```

可以先建立完整数组：

```python
full_prediction = np.full((original_num_points, 3), np.nan, dtype=np.float32)
indices = sample["volume_downsample_indices"][0]
full_prediction[indices] = sampled_prediction[0]
```

未采样点保持 `NaN`。如果输入 PKL 包含 `source_point_ids`，应优先使用采样后的 `source_point_ids[0]` 映射回原 VTK 点编号。

## 7. 安全行为

- 输入 PKL 不会修改或删除；
- 输出目录不能与输入目录相同；
- `--overwrite` 只清理输出目录中的编号 PKL、统计文件和临时 PKL；
- 若目标点数大于任意输入样本的点数，程序会停止并报错；
- 坐标允许超出 `[0,1]`，不会裁剪；
- `x_volume` 与 `y_velocity` 必须严格对齐。
