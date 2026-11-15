# DrivAerML 瞬态点云预测

该目录只用于 DrivAerML 时间序列训练。原先复制进来的稳态数据加载、多车型
采样、稳态模型、稳态配置、旧 checkpoint 和稳态后处理已经移除。

## 1. 任务定义

数据目录：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/run_1_pklFile_10k
```

输入数据采用共享几何和分时间步物理场：

```text
run_1_pklFile_10k/
├── geometry.pkl
├── normalization_scalars.pkl
├── 10.pkl
├── 11.pkl
└── ...
```

程序从 `10.pkl` 开始，忽略 `1.pkl-9.pkl`。当前完整目录包含
`10.pkl-233.pkl` 共 224 帧；剔除跨缺失时间的配对后，有 174 个固定步长预测
样本，按默认比例得到训练 121、验证 27、测试 26 个时间对。

支持两个独立任务：

```text
surface_pressure: x_surface + p(t) -> p(t + delta_t)
volume_velocity : x_volume  + U(t) -> U(t + delta_t)
```

压力和速度不会混在一个网络中训练，因为它们分别定义在汽车表面点和体单元
中心上，空间节点并不对应。

## 2. 时间序列训练方式

训练逻辑参考 P3D 的 next-frame prediction 和 autoregressive rollout：

```text
训练：真实 state(t)       -> 预测 state(t+1)
测试：真实初始 state(t0)  -> prediction(t1)
      prediction(t1)      -> prediction(t2)
      prediction(t2)      -> prediction(t3) ...
```

默认 `context_steps: 1`。设置为 2 时，输入最近两帧：

```text
[state(t-1), state(t)] -> state(t+1)
```

数据按照时间顺序切分，不能在切分前随机打乱。训练区间内部可以 shuffle；验证
和测试保持时间顺序。测试指标来自自回归 rollout，而不是每一步都使用真实上一
帧的 teacher forcing。

预期保存间隔固定为 `0.005 s`。程序使用 `expected_delta_t: 0.005` 严格检查
相邻时间；如果缺少中间 PKL，跨越 `0.010 s` 或 `0.015 s` 的时间对会被跳过，
不会混入固定步长训练。模型仍显式接收归一化的当前时间和 `delta_t`，便于检查
数据并保留将来扩展能力。

从 `1.55 s` 到 `3.00 s` 若严格每 `0.005 s` 保存，理论上应有 291 帧；当前有
224 帧，共缺少 67 个时间点并形成 49 处断档。程序会打印并跳过这些断档；如能
补齐缺失帧，可获得更长的连续自回归训练和测试区间。

## 3. 多模型对比

P3D 原模型输入是规则三维体素：

```text
(batch, channels, depth, height, width)
```

本数据是从非结构网格随机下采样得到的固定点云：

```text
(batch, points, channels)
```

把随机点编号直接 reshape 成立方体会制造不存在的空间邻接关系。因此本工程
沿用 P3D 的时间预测和 rollout 方法，并使用与稳态目录一致的非结构点云模型：

```text
Transolver
LinearNO
CANO
RETO
RETO_plus
RETO_plus_v2
SATO
```

七种模型共用完全相同的数据配对、时间切分、loss、反归一化与自回归测试逻辑。
模型差异只保留在空间算子内部。需要注意，`RETO` 使用完整点对点 attention，
复杂度为 `O(N²)`；其余模型使用切片、线性、聚类或局部序列机制，更适合大量
点。`N=10000` 时 RETO 仍可测试，但显存和运行时间会明显高于其他模型。

## 4. 模型输入与输出

所有模型的统一输入包括：

```text
归一化坐标
最近 context_steps 帧的归一化物理场
归一化当前时间
归一化 delta_t
```

模型默认预测物理场增量：

```text
state(t + delta_t) = state(t) + learned_update
```

增量预测适合相邻瞬态场。`RETO、RETO_plus、RETO_plus_v2` 的位置编码严格只
作用于前三维 XYZ；`SATO` 的序列化顺序也只由 XYZ 计算。物理场与时间条件作为
额外特征进入空间主干，不会被错误解释为坐标。

## 5. 数据切分

程序先剔除不满足 `delta_t=0.005 s` 的时间对，再按时间顺序分配：

```text
train: 最早的连续时间对
val  : 中间的连续时间对
test : 最晚的连续时间对
```

当前数据来自同一条 CFD 轨迹，因此测试结果衡量的是同一算例后段的时间外推，
不是跨车型泛化能力。

## 6. 运行方法

先检查数据和模型前向传播：

```bash
cd /Users/zhangbojun/ML_Turbulent/RETO++/DrivAerML_transient

/opt/anaconda3/envs/pgd-no/bin/python main.py \
  config_Transolver_surface_pressure.yml \
  --phase inspect
```

训练表面压力：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py config_Transolver_surface_pressure.yml
```

训练体场速度：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py config_Transolver_volume_velocity.yml
```

断点续训：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py \
  config_LinearNO_volume_velocity.yml \
  --phase resume
```

只进行测试 rollout：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py \
  config_CANO_volume_velocity.yml \
  --phase test
```

也兼容：

```text
python main.py --config config_RETO_plus_v2_volume_velocity.yml
```

配置命名统一为：

```text
config_<MODEL>_surface_pressure.yml
config_<MODEL>_volume_velocity.yml
```

例如比较全部压力模型时，依次运行 7 个 `*_surface_pressure.yml`；每个配置使用
独立实验目录，不会覆盖其他模型的 checkpoint。

训练和验证均使用批次进度条：

```text
Epoch 12/200 - Train: 68%|...| loss=..., mse=..., relL2=..., lr=...
Epoch 12/200 - Val:   44%|...| mse=..., relL2=...
```

使用 `nohup` 重定向到日志时也会保留这些进度信息。

## 7. 时间戳与断点续训

程序启动训练时生成和稳态目录相同形式的时间戳：

```text
YYYYMMDD_HHMM
```

每个模型的压力任务和速度任务也分别使用独立实验目录：

```text
experiment_DrivAerML_transient_RETO_plus_v2_surface_pressure/
├── best_model_RETO_plus_v2_surface_pressure_20261115_1030.pth
├── history_RETO_plus_v2_surface_pressure_20261115_1030.json
└── test_RETO_plus_v2_surface_pressure_20261115_1030/
```

断点续训指定原时间戳：

```bash
python main.py config_RETO_plus_v2_surface_pressure.yml \
  --phase resume \
  --run-timestamp 20261115_1030
```

单独测试指定原时间戳：

```bash
python main.py config_RETO_plus_v2_surface_pressure.yml \
  --phase test \
  --run-timestamp 20261115_1030
```

对于 `resume/test`，省略 `--run-timestamp` 时，会在该模型独立实验目录中自动
选择当前任务最新的 `best_model_*.pth`。训练时也可以显式传入时间戳，但通常
让程序自动生成即可。

## 8. AMP 设置

```text
V100:      amp_type: fp16
A100/H100: amp_type: bf16
macOS MPS: amp_type: none
CPU:       amp_type: none
```

V100 不支持原生 BF16，远端使用 V100 时必须把 YAML 中的 `bf16` 改为 `fp16`。

## 9. 输出

每个模型、每种任务拥有独立实验目录，文件名再带时间戳：

```text
experiment_DrivAerML_transient_<model>_<task>/
├── best_model_<model>_<task>_<timestamp>.pth
├── history_<model>_<task>_<timestamp>.json
└── test_<model>_<task>_<timestamp>/
    ├── test_metrics.json
    └── rollout_predictions.npz
```

测试同时报告归一化空间和反归一化物理空间的指标：

```text
normalized_mse
normalized_relative_l2
physical_mae
physical_rmse
physical_relative_l2
```

`rollout_predictions.npz` 保存归一化预测、反归一化预测、真实值和对应样本编号。

本工程不再生成 `run_metadata.json`。保留的 JSON 文件用途如下：

```text
history_*.json      每个 epoch 的 train loss、val MSE、val relative L2 和学习率
test_metrics.json   最佳模型自回归测试的误差汇总
```

它们只是轻量结果记录，便于画训练曲线和比较模型，不参与模型前向计算，也不是
恢复训练所必需的文件。恢复训练使用带时间戳的 `.pth` checkpoint。

## 10. 文件职责

```text
main.py                         配置解析、数据/模型构建、流程入口
dataloader.py                   时间排序、相邻帧配对、按时间切分
models/temporal_adapter.py       统一瞬态输入、模型注册和残差预测
models/*.py                     七种原始空间模型的瞬态适配版本
SATO_dataset/                   SATO 的 XYZ 序列化工具
utils/training.py               训练、验证、checkpoint、自回归测试
config_<MODEL>_<TASK>.yml       模型与任务专用配置
```
