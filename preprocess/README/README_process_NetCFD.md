# DrivAerNet-CFD 体场速度预处理

对应脚本：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/process_NetCFD.py
```

该脚本专门处理 DrivAerNet-CFD 非结构网格的**完整体场速度**，不再提供压力、WSS 或汽车表面场输出模式。

## 1. 输入数据

已使用以下文件验证：

```text
preprocess/DrivAerNet-CFD/E_S_WWC_WM_109.vtk
```

脚本使用两个 VTK 字段：

| 字段 | 用途 | 是否写入样本 PKL |
|---|---|---|
| `U` | 三分量体场速度 | 是，保存为 `y_velocity` |
| `wallShearStress` | 从外边界识别汽车表面，计算坐标尺度 | 否 |

汽车表面和 WSS 只参与坐标统计，不作为模型训练数据输出。

## 2. 归一化

### 2.1 体场坐标

体场坐标采用整个数据集汽车表面的 global min-max 尺度：

\[
\hat{\boldsymbol x}=
\frac{\boldsymbol x_{volume}-\boldsymbol x_{surface,min}}
{\boldsymbol x_{surface,max}-\boldsymbol x_{surface,min}}.
\]

汽车以外的远场和尾流坐标允许超过 `[0,1]`，脚本不会裁剪。

### 2.2 速度

速度的三个分量分别使用全数据集 mean-std：

\[
\hat{\boldsymbol u}=
\frac{\boldsymbol u-\boldsymbol\mu_u}{\boldsymbol\sigma_u}.
\]

反归一化：

```python
y_velocity_physical = y_velocity_normalized * y_velocity_std + y_velocity_mean
```

## 3. 输出格式

每个 VTK 输出一个同编号的 PKL：

```python
{
    "x_volume": np.ndarray,         # (1, N, 3), float32
    "y_velocity": np.ndarray,       # (1, N, 3), float32
    "source_point_ids": np.ndarray  # 可选，形状 (1, N)
}
```

`x_volume[0, i]`、`y_velocity[0, i]` 和 `source_point_ids[0, i]` 始终对应同一个原始 VTK 点。字段命名与 DrivAerML、DrivAerStar 数据集保持一致。

PKL 中的首维是显式样本维。dataloader 读取单个 PKL 时会先将 `(1,N,3)` 转为 `(N,3)`，之后 PyTorch DataLoader 再组合为 `(B,N,3)`，不会产生 `(B,1,N,3)`。

输出目录还包含 `normalization_scalars.pkl`，主要字段为：

```text
coords_min
coords_max
coords_range
surface_coords_min
surface_coords_max
surface_coords_range
region_coords_min
region_coords_max
y_velocity_mean
y_velocity_std
y_velocity_point_count
coordinate_scale_policy
volume_coordinates_may_exceed_unit_interval
samples
```

兼容字段 `coords_min/max/range` 与 `surface_coords_min/max/range` 相同。

## 4. macOS 运行

```bash
cd ~/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_NetCFD.py \
  --output-dir preprocess/DrivAerNet-CFD/volume_velocity_pklFile \
  --save-point-ids \
  --overwrite
```

只测试 `109.vtk`：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_NetCFD.py \
  --pattern 'E_S_WWC_WM_109.vtk' \
  --output-dir preprocess/DrivAerNet-CFD/test_volume_velocity \
  --save-point-ids \
  --overwrite
```

## 5. Windows PowerShell 运行

```powershell
cd /Users/zhangbojun/ML_Turbulent

python preprocess\process_NetCFD.py `
  --output-dir preprocess\DrivAerNet-CFD\volume_velocity_pklFile `
  --save-point-ids `
  --overwrite
```

## 6. 映射回原网格

建议始终添加 `--save-point-ids`。体场点保持原始 VTK 点顺序，预测速度反归一化后，可根据 `source_point_ids` 写回原始网格的 point data，用 ParaView 等工具进行后处理。

## 7. 注意事项

1. 脚本固定提取完整体场和速度，不再接受 `--region`、`--fields` 参数。
2. 单个 VTK 可能包含约一千万个点，需要预留足够内存和磁盘空间。
3. `--overwrite` 会删除输出目录中已有的 PKL 文件。
4. dataloader 不应再次归一化，也不能把 `x_volume` 中超出 `[0,1]` 的体场坐标裁剪掉。
5. 原始 VTK 不包含现成 SDF；当前脚本不会计算或保存 SDF。
