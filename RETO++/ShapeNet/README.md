# 模型记录

SATO 已适配 MLCFD 的 7 推 4 联合任务。使用 `config_SATO.yml`，模型结构和
运行说明见 `models/README_SATO.md`。

## cuda 1 1113
3208779
3208780
3208781
3208782
3208783
3210160

## ShapeNet
### RETO_plus_v2
#### best_model_RETO_plus_v2_p_and_v_20261105_0918.pth
  Test velocity relative L2: 0.026872
  Test pressure relative L2: 0.078786
  Test Cd relative error: 0.014630
#### best_model_RETO_plus_v2_p_and_v_20261106_0956.pth
  Test velocity relative L2: 0.026422
  Test pressure relative L2: 0.076814
  Test Cd relative error: 0.014836
#### best_model_RETO_plus_v2_surface_pressure_20261107_1652.pth
  Test pressure relative L2: 0.090393
  Test Cd relative error: 0.020546




### Transolver
#### best_model_Transolver_p_and_v_2026-1101.pth
  Test velocity relative L2: 0.025799
  Test pressure relative L2: 0.078387
  Test Cd relative error: 0.013380
#### best_model_Transolver_surface_pressure_20261107_1652.pth
  Test pressure relative L2: 0.091752
  Test Cd relative error: 0.018949

#### best_model_Transolver_volume_velocity_20261109_1327.pth
  Test velocity relative L2: 0.033376


### RETO++
#### best_model_RETO_plus_p_and_v_2026-1101.pth
  Test velocity relative L2: 0.024785
  Test pressure relative L2: 0.075845
  Test Cd relative error: 0.015862
#### best_model_RETO_plus_surface_pressure_20261107_1653.pth
  Test pressure relative L2: 0.082857
  Test Cd relative error: 0.020483

#### best_model_RETO_plus_v2_surface_pressure_20261109_1012.pth
  Test pressure relative L2: 0.136628
  Test Cd relative error: 0.034242

#### best_model_RETO_plus_v2_surface_pressure_20261109_1324.pth
  Test pressure relative L2: 0.093887
  Test Cd relative error: 0.020106


### CANO
#### best_model_CANO_p_and_v_2026-1101.pth
  Test velocity relative L2: 0.025350
  Test pressure relative L2: 0.078819
  Test Cd relative error: 0.014891

#### best_model_CANO_surface_pressure_20261107_1652.pth
  Test pressure relative L2: 0.092158
  Test Cd relative error: 0.017829
  
#### best_model_CANO_volume_velocity_20261109_1327.pth
  Test velocity relative L2: 0.029975


### LinearNO
#### best_model_LinearNO_surface_pressure_20261107_1653.pth
  Test pressure relative L2: 0.088630
  Test Cd relative error: 0.017017




## MLCFD 三种训练任务

`preprocess/process_mlcfd_split.py` 生成的数据可由同一套 `main.py`、
`dataloader.py` 和 `utils/ml.py` 训练三种任务：

| 配置文件 | `predicted_feature_name` | 输入 | 输出 |
|---|---|---|---|
| `config_mlcfd_combined.yml` | `p_and_v` | 7维坐标/SDF/法向量 | 3维速度+压力 |
| `config_mlcfd_surface_pressure.yml` | `surface_pressure` | 面场3维坐标 | 面场压力 |
| `config_mlcfd_volume_velocity.yml` | `volume_velocity` | 体场3维坐标 | 体场3维速度 |

从 `RETO++/ShapeNet` 目录运行：

```bash
python main.py --config config_mlcfd_combined.yml
python main.py --config config_mlcfd_surface_pressure.yml
python main.py --config config_mlcfd_volume_velocity.yml
```

三个任务共用同一批 PKL，但模型的 `space_dim` 和 `out_dim` 必须与表中一致。
验证和测试指标均先反归一化到真实物理尺度，再计算 relative L2 与 RMSE。
压力任务在提供原始压力 VTK 时还会计算压力主导的 Cd；速度任务不计算 Cd。

