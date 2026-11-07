# MLCFD 面场/体场独立归一化

脚本：`preprocess/process_mlcfd_split.py`

该脚本读取 `process_mlcfd.py` 已生成的旧版逐样本 PKL，将拼接在一起的汽车表面节点和外流场节点拆成与 DrivAerStar 一致的字段，并重新生成统一的归一化统计。

## 为什么不能直接再次归一化

源 PKL 中的 `input` 和 `output` 已归一化。脚本会先使用源目录中的 `normalization_scalars.pkl` 恢复真实坐标与物理量，然后基于全部待处理样本重新统计，最后只归一化一次。它不会直接把已归一化数组再做 min-max 或 mean-std。

## 输出字段

每个新 PKL 同时提供三种训练方式：

| 任务 | 输入字段 | 输出字段 | 输入/输出维度 |
|---|---|---|---|
| 七维推四维 | `input` | `output` | 7 -> 4 |
| 面场推压力 | `x_surface` | `y_spressure` | 3 -> 1 |
| 体场推速度 | `x_volume` | `y_velocity` | 3 -> 3 |

另外保留：

- `surface_mask` / `surf`：旧版拼接节点中面场节点的位置；
- `x` / `y`：`input` / `output` 的兼容别名；
- `mapping`、`source`：原有网格回映射信息；
- `metadata`：源格式、面场点数、体场点数和节点顺序说明。

MLCFD 源数据不包含壁面剪切应力，因此新 PKL 不会伪造 DrivAerStar 的 `y_wss` 字段。

## 归一化规则

- `x_surface`：所有样本汽车表面坐标的全局 min-max；
- `x_volume`：所有样本外流场坐标的全局 min-max；
- `y_spressure`：所有表面压力的逐分量 mean-std；
- `y_velocity`：所有体场速度的逐分量 mean-std；
- SDF：沿用源 PKL 中已经完成的 min-max；
- 单位法向量：保持不变。

新 `normalization_scalars.pkl` 的关键字段为：

```text
surface_coords_min / max / range
volume_coords_min / max / range
y_spressure_mean / std / count
y_velocity_mean / std / count
output_mean / output_std
samples / num_samples
```

其中 `output_mean` 与 `output_std` 按 `[velocity_x, velocity_y, velocity_z, pressure]` 拼接，专门用于兼容七维推四维训练和反归一化。

## macOS 运行方法

在仓库根目录运行：

```bash
cd ~/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_mlcfd_split.py
```

默认输入与输出：

```text
输入：preprocess/mlcfd_data/pkl_test
输出：preprocess/mlcfd_data/pkl_test_v2
```

建议先用两个样本做冒烟测试：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_mlcfd_split.py \
  --limit 2 \
  --output-dir preprocess/mlcfd_data/pkl_test_smoke
```

确认输出后处理全部样本：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_mlcfd_split.py \
  --input-dir preprocess/mlcfd_data/pkl_test \
  --output-dir preprocess/mlcfd_data/pkl_test_v2
```

如果输出目录已存在，明确允许覆盖时加入：

```bash
--overwrite
```

脚本禁止输入目录与输出目录相同，并会拒绝把已经是 `mlcfd-split-fields-v3` 的数据再次处理。

## ShapeNet 训练任务配置

在 `RETO++/ShapeNet` 的 YAML 中使用以下三组配置之一。

七维推四维：

```yaml
experiment:
  predicted_feature_name: p_and_v
model:
  space_dim: 7
  out_dim: 4
data:
  data_path: ../../preprocess/mlcfd_data/pkl_test_v2
```

面场坐标推压力：

```yaml
experiment:
  predicted_feature_name: surface_pressure
model:
  space_dim: 3
  out_dim: 1
data:
  data_path: ../../preprocess/mlcfd_data/pkl_test_v2
```

体场坐标推速度：

```yaml
experiment:
  predicted_feature_name: volume_velocity
model:
  space_dim: 3
  out_dim: 3
data:
  data_path: ../../preprocess/mlcfd_data/pkl_test_v2
```

相对路径按运行命令的当前工作目录解析。远端服务器建议使用绝对路径，避免从不同目录启动时读错数据。
