# -----------------------------------------------------------------------
# Author: Bojun Zhang
# Date  : 2026-10-29 15:59:05
# File  : READEME
# -----------------------------------------------------------------------


# 模型记录
## DrivAerNet

## 体场速度预测

`preprocess/process_NetCFD.py` 生成的样本采用：

```text
x_volume: 归一化体场坐标，PKL 中形状 (1, N, 3)
y_velocity: 归一化速度，PKL 中形状 (1, N, 3)
```

训练目录支持任务名 `volume_velocity`。对应 YAML 必须显式配置 `out_dim: 3`；`main.py` 只负责校验，不会自动覆盖 YAML。可用配置：

```text
config_LinearNO_volume_velocity.yml
config_Transolver_volume_velocity.yml
config_CANO_volume_velocity.yml
config_RETO_plus_volume_velocity.yml
config_RETO_plus_v2_volume_velocity.yml
config_RETO_volume_velocity.yml
config_SATO_volume_velocity.yml
```

运行示例：

```bash
cd ~/ML_Turbulent/RETO++/DrivAerNet
python main.py --config config_RETO_plus_volume_velocity.yml
```

根据服务器上的实际位置修改配置中的 `data.data_path`。验证与测试阶段会先反归一化到物理速度，再报告总体相对 L2 误差以及 `u/v/w` 三个速度分量的相对误差。

### "cuda 5"  1112
3204421
3204422
3204748
3204749

### 8种车型-800组-10k-pressure
#### RETO_plus_v2
0.177    best_model_RETO_plus_v2_pressure_20261105_0910.pth
0.176    best_model_RETO_plus_v2_pressure_20261106_0951.pth 
0.181    best_model_RETO_plus_v2_pressure_20261106_0953.pth

#### CANO
0.178    best_model_CANO_pressure_20261029_1231.pth

#### RETO++
0.175    best_model_RETO_plus_pressure_20261029_1231.pth

#### Transolver
0.180    best_model_Transolver_pressure_20261029_1231.pth


### 8种车型-1600组-10k-pressure
#### RETO_plus
0.176748 best_model_RETO_plus_pressure_20261109_1348.pth

#### RETO_plus_v2
0.164616 best_model_RETO_plus_v2_pressure_20261109_1345.pth
0.164769 best_model_RETO_plus_v2_pressure_20261109_1348.pth

#### Transolver
0.167438 best_model_Transolver_pressure_20261107_1036.pth
0.167230 best_model_Transolver_pressure_20261109_1345.pth

#### CANO
0.172074 best_model_CANO_pressure_20261107_1036.pth
0.170248 best_model_CANO_pressure_20261109_1345.pth
0.172148 best_model_CANO_pressure_20261109_1348.pth


### 8种车型-全车-pressure
#### PID
2223406 RETO_plus   RETO_plus_DrivAerNet_2026_1029_1557
2114390 Transolver  Transolver_DrivAerNet_2026_1030_1249
2352559 CANO        CANO_DrivAerNet_2026_1101_1047
