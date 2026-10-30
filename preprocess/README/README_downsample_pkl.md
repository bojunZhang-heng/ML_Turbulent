# PKL 点云离线下采样

脚本：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/downsample_pkl.py
```

该脚本同时支持：

- DrivAerStar：`x_surface/y_spressure/y_wss` 和 `x_volume/y_velocity`；
- 旧格式数据：`coor/normals/pressure/seg_matrix` 等按节点对齐字段。

脚本只选择已有节点，不重新归一化。`normalization_scalars.pkl` 会原样复制到输出目录。

## DrivAerStar 对齐规则

DrivAerStar 有两套拓扑不同的点集，必须分别采样：

| 点集 | 坐标 | 同步物理量 | 保存的原始索引 |
|---|---|---|---|
| surface | `x_surface` | `y_spressure`、`y_wss` | `surface_downsample_indices` |
| volume | `x_volume` | `y_velocity` | `volume_downsample_indices` |

一个点集内的坐标和物理量严格使用同一组索引。

## 推荐：生成表面训练集

```bash
cd /Users/zhangbojun/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python preprocess/downsample_pkl.py \
  --input-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar_surface_N100000 \
  --point-set surface \
  --num-points 100000 \
  --method random
```

输出 PKL 只包含：

```text
x_surface
y_spressure
y_wss
surface_downsample_indices
metadata
normalization
```

原始千万级体场数组会被移除，因此表面数据集不会继续占用体场存储空间。

## 生成体速度训练集

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/downsample_pkl.py \
  --input-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar_volume_N100000 \
  --point-set volume \
  --num-points 100000 \
  --method random
```

输出 PKL 只包含：

```text
x_volume
y_velocity
volume_downsample_indices
metadata
normalization
```

## 同时保留表面和体场

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/downsample_pkl.py \
  --input-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar_both_N100000 \
  --point-set both \
  --num-points 100000 \
  --method random
```

表面和体场会各自独立选择 100000 个节点，不会混用索引。

## 参数

| 参数 | 说明 |
|---|---|
| `--input-dir` | 原始 PKL 目录 |
| `--output-dir` | 下采样结果目录 |
| `--num-points` | 每个选中点集保留的节点数 |
| `--point-set` | `surface`、`volume`、`both` 或 `auto` |
| `--method` | `random` 或 `voxel` |
| `--seed` | 可复现随机种子 |
| `--overwrite` | 覆盖已有输出文件 |

`auto` 在 DrivAerStar 中等价于对当前 PKL 内存在的表面和体场分别采样；在旧格式中自动处理 `coor`。

对于千万级体场，推荐 `random`。体素采样需要建立较大的临时体素数组，内存占用明显更高。

## 覆盖已有结果

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/downsample_pkl.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --point-set surface \
  --num-points 100000 \
  --method random \
  --overwrite
```

脚本先写入临时文件，再原子替换目标 PKL，降低写入中断造成损坏的风险。

## 映射回原始网格

假设模型输出对应表面下采样点：

```python
with open("1301.pkl", "rb") as f:
    sample = pickle.load(f)

indices = sample["surface_downsample_indices"]
full_prediction = np.full((original_surface_count, 1), np.nan)
full_prediction[indices] = prediction
```

体速度使用：

```python
indices = sample["volume_downsample_indices"]
full_velocity = np.full((original_volume_count, 3), np.nan)
full_velocity[indices] = prediction
```

下采样索引只能恢复被选中节点的位置。未参与训练的节点仍需最近邻、反距离权重或网格插值才能获得连续预测场。

## 注意

1. 输出目录不能与输入目录相同。
2. 默认不覆盖已有结果。
3. 请求点数不能超过对应原始点数。
4. 下采样后继续使用原始全数据集归一化统计量。
5. 正式比较不同模型时，应使用相同输出目录和相同下采样索引。
