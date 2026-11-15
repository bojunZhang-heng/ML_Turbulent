# 模型记录
## DrivAerStar

### E / F / N 多车型训练

所有模型配置均支持同时从 `case_E`、`case_F`、`case_N` 三类车型读取
PKL。配置使用复合样本编号 `(vehicle, case_id)`，因此不同目录中相同的
数字文件名不会互相覆盖；每类车型也会使用本目录自己的
`normalization_scalars.pkl` 完成反归一化和评价。

```yaml
data:
  dataset_root: /data/zhangbojun/DrivAerStar
  selection_mode: random
  selection_seed: 42
  train_ratio: 0.6
  val_ratio: 0.3
  test_ratio: 0.1
  vehicle_datasets:
    case_E:
      path: case_E_pklFile_10k
      count: all       # 可改为整数，例如 200
      enabled: true
    case_F:
      path: case_F_pklFile_10k
      count: all
      enabled: true
    case_N:
      path: case_N_pklFile_10k
      count: all
      enabled: true
```

每种车型先独立随机选取 `count` 个案例，再分别按相同比例划分训练、验证、
测试集。若暂时不训练某类车型，将对应的 `enabled` 改为 `false`。每个启用
目录的样本数应足以产生非空划分；当前 `0.6/0.3/0.1` 比例建议每类至少
选择 10 个案例。

重新生成 DrivAerStar 的任务配置：

```bash
python ../generate_task_configs.py --dataset DrivAerStar
```

### 双 checkpoint 与最终评价

所有模型都会同时保存两份最优权重：

- `*_val_loss.pth`：验证集归一化 MSE 最低；
- `*_relative_error.pth`：验证集反归一化相对 L2 最低。

训练结束后程序会在同一个测试集上依次加载两份权重，最终用反归一化物理量
的相对 L2 和 R² 进行比较：

```text
Checkpoint test comparison:
  Selected by              Relative L2             R2
  val_loss                     0.xxxxxx       0.xxxxxx
  relative_error               0.xxxxxx       0.xxxxxx
```

矢量任务还会分别打印三个分量的相对 L2 和 R²。旧版单一
`best_model_*.pth` 不会自动拆分成两份，需要重新训练。

### "cuda 3" 1115-pressure
3290772
3290773
3290774
3290775
3290776
3290777



### 900组-10k-pressure

### 569组-10k-pressure
#### RETO_plus_v2
0.205    best_model_RETO_plus_v2_surface_pressure_20261105_0915.pth

#### RETO_plus
0.222    best_model_RETO_plus_surface_pressure_20261101_1301.pth
0.218    best_model_RETO_plus_surface_pressure_20261102_0922.pth

#### Transolver
0.224    best_model_Transolver_surface_pressure_20261101_1301.pth

#### CANO
0.228    best_model_CANO_surface_pressure_20261101_1301.pth



### 569组-10k-velocity
#### RETO_plus_v2
0.127057 best_model_RETO_plus_v2_volume_velocity_20261113_1119.pth
0.130969 best_model_RETO_plus_v2_volume_velocity_20261113_1117.pth

#### RETO_plus
0.128682 best_model_RETO_plus_volume_velocity_20261111_1433.pth
0.128354 best_model_RETO_plus_volume_velocity_20261113_1117.pth

#### CANO
0.127763 best_model_CANO_volume_velocity_20261111_1433.pth
0.129372 best_model_CANO_volume_velocity_20261113_1117.pth

#### LinearNO
0.127827 best_model_LinearNO_volume_velocity_20261111_1433.pth
0.128228 best_model_LinearNO_volume_velocity_20261113_1117.pth

#### Transolver
0.133392 best_model_Transolver_volume_velocity_20261111_1433.pth
0.129814 best_model_Transolver_volume_velocity_20261113_1117.pth

#### SATO
0.130762 best_model_SATO_volume_velocity_20261111_1433.pth
0.130746 best_model_SATO_volume_velocity_20261113_1117.pth
