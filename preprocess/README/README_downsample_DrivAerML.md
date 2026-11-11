# downsample_DrivAerML.py 使用说明

该脚本专门处理 `process_DrivAerML.py` 生成的编号 PKL。表面场和体场可以采用不同采样数量，采样后仍保存在同一个 PKL 中。

脚本不重新计算归一化，也不会再次修改压力、速度、WSS 或总压系数。

## 1. 输入要求

输入目录必须包含：

```text
1.pkl
2.pkl
...
normalization_scalars.pkl
```

并且 `normalization_scalars.pkl` 必须包含表面和体场坐标统计键：

```text
surface_coords_min / surface_coords_max / surface_coords_range
volume_coords_min  / volume_coords_max  / volume_coords_range
```

如果输入仍是只有 `coords_min/coords_range` 的 v1 数据，应先执行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerML.py \
  --source-pkl-dir /path/to/drivaerML_v1 \
  --output-dir /path/to/drivaerML_v2 \
  --overwrite
```

下采样脚本会拒绝直接处理共享坐标尺度的 v1 数据，避免继续传播错误尺度。

脚本同时支持 surface-scale v3 数据，即 `x_surface` 和 `x_volume` 都使用
汽车表面 `surface_coords_min/range`。这种格式下外流场坐标超出 `[0,1]`
属于正常情况，不会被裁剪或拒绝。脚本会检查表面和体场统计量确实相同，
并把输出 schema 统一修正为 `drivaerml-normalized-surface-scale-v3`。

## 2. 推荐命令：表面和体场保存在同一 PKL

例如表面保留 10,000 点、体场保留 50,000 点：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerML.py \
  --input-dir /Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000_v2 \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N10k \
  --point-set auto \
  --surface-num-points 10000 \
  --volume-num-points 10000 \
  --method random \
  --overwrite
```

`auto` 和 `both` 都适合 DrivAerML，因为数据中既有完整样本，也有只有表面场的部分样本：

- 完整样本同时采样 surface 和 volume；
- 只有表面场的样本只采样 surface；
- 不会为缺失的体场生成零数组。

因此，`both` 表示“尽可能保留两类场”，而不是要求每个源样本必须同时
具有两类场。速度训练的数据发现逻辑会自动跳过没有 `x_volume/y_velocity`
的部分样本，压力和 WSS 训练仍可使用这些样本。

### 2.1 中断后继续

如果任务中途退出，可使用完全相同的参数并增加 `--resume`：

```bash
python preprocess/downsample_DrivAerML.py \
  --input-dir /data/zhangbojun/DrivAerML/pklFile_100k \
  --output-dir /data/zhangbojun/DrivAerML/pklFile_10k \
  --point-set both \
  --surface-num-points 10000 \
  --volume-num-points 10000 \
  --method random \
  --resume
```

已有编号 PKL 会先通过字段、点数、有限值和坐标范围校验，然后跳过；只生成缺失文件。`--resume` 与 `--overwrite` 不能同时使用。

## 3. 输出字段

完整样本仍保存：

```text
x_surface
y_spressure
y_wss
x_volume
y_velocity
volume_cell_totalpcoeff（如果源样本存在）
surface_downsample_indices
volume_downsample_indices
```

表面字段共用 `surface_downsample_indices`：

```text
x_surface[i] <-> y_spressure[i] <-> y_wss[i]
```

体场字段共用 `volume_downsample_indices`：

```text
x_volume[i] <-> y_velocity[i] <-> volume_cell_totalpcoeff[i]
```

因此坐标和物理量仍保持严格的一一对应。

## 4. 只保留表面场

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerML.py \
  --input-dir /path/to/drivaerML_v2 \
  --output-dir /path/to/drivaerML_surface10000 \
  --point-set surface \
  --surface-num-points 10000 \
  --method voxel \
  --overwrite
```

输出将移除 `x_volume`、`y_velocity` 和 `volume_cell_totalpcoeff`，用于减少磁盘占用。

## 5. 只保留体场

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerML.py \
  --input-dir /path/to/drivaerML_v2 \
  --output-dir /path/to/drivaerML_volume50000 \
  --point-set volume \
  --volume-num-points 50000 \
  --method random \
  --overwrite
```

只有包含体场的完整样本能够用于该模式。遇到缺失 `x_volume` 的部分样本时，程序会报错；如需自动兼容部分样本，请使用 `--point-set auto`。

## 6. 兼容统一点数参数

旧式 `--num-points` 仍可使用：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerML.py \
  --input-dir /path/to/drivaerML_v2 \
  --output-dir /path/to/output \
  --point-set auto \
  --num-points 10000
```

这会对所有可用点集使用 10,000 点。专用参数优先级更高，例如：

```text
--num-points 10000 --volume-num-points 50000
```

表示 surface 使用 10,000，volume 使用 50,000。

## 7. random 与 voxel

- `random`：速度快、内存低，推荐体场和批量处理；
- `voxel`：空间覆盖更均匀，适合表面点，但计算和临时内存开销更高。

一次运行使用一种方法。若同时保存 surface 和 volume，通常推荐 `random` 以平衡速度和内存。

## 8. normalization_scalars.pkl

输出目录中的统计文件会继承原始归一化参数，并增加：

```python
scalars["downsampling"] = {
    "source_directory": "...",
    "point_set": "auto",
    "num_points": {
        "surface": 10000,
        "volume": 50000,
    },
    "method": "random",
    "seed": 42,
}
```

`samples` 清单中的 `surface_points`、`volume_points` 会更新为采样后的数量，同时通过
`original_surface_points`、`original_volume_points` 保留原始数量。

## 9. 映射回全量点集

```python
full_prediction = np.full((original_surface_points, channels), np.nan)
full_prediction[sample["surface_downsample_indices"]] = prediction
```

体场使用 `volume_downsample_indices`。未采样位置保持 `NaN`，如需完整连续场，还需要最近邻或其他插值方法。
