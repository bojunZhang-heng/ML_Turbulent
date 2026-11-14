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

## 3. 为什么没有直接使用 P3D 的 Conv3D

P3D 原模型输入是规则三维体素：

```text
(batch, channels, depth, height, width)
```

本数据是从非结构网格随机下采样得到的固定点云：

```text
(batch, points, channels)
```

把随机点编号直接 reshape 成立方体会制造不存在的空间邻接关系。因此本工程
沿用 P3D 的时间预测和 rollout 方法，但空间模型采用适合点云的线性注意力算子。
其复杂度随点数近似线性增长，不构造 `N x N` 注意力矩阵。

## 4. 模型输入与输出

模型输入包括：

```text
归一化坐标
连续正余弦坐标编码
最近 context_steps 帧的归一化物理场
归一化当前时间
归一化 delta_t
```

模型默认预测物理场增量：

```text
state(t + delta_t) = state(t) + learned_update
```

增量预测适合相邻瞬态场。模型包含独立 Q/K/V 的线性注意力、MLP、残差连接
和 block checkpoint，不会生成点数平方级的注意力矩阵。

## 5. 数据切分

程序先剔除不满足 `delta_t=0.005 s` 的时间对，再按时间顺序分配：

```text
train: 最早的连续时间对
val  : 中间的连续时间对
test : 最晚的连续时间对
```

数据量仍然很小，只适合打通流程和初步过拟合测试。要评估时间泛化能力，建议
准备几十到几百个连续时间步，并保证输出时间间隔满足研究需求。

## 6. 运行方法

先检查数据和模型前向传播：

```bash
cd /Users/zhangbojun/ML_Turbulent/RETO++/DrivAerML_transient

/opt/anaconda3/envs/pgd-no/bin/python main.py \
  config_surface_pressure.yml \
  --phase inspect
```

训练表面压力：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py config_surface_pressure.yml
```

训练体场速度：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py config_volume_velocity.yml
```

断点续训：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py \
  config_volume_velocity.yml \
  --phase resume
```

只进行测试 rollout：

```bash
/opt/anaconda3/envs/pgd-no/bin/python main.py \
  config_volume_velocity.yml \
  --phase test
```

也兼容：

```text
python main.py --config config_volume_velocity.yml
```

## 7. AMP 设置

```text
V100:      amp_type: fp16
A100/H100: amp_type: bf16
macOS MPS: amp_type: none
CPU:       amp_type: none
```

V100 不支持原生 BF16，远端使用 V100 时必须把 YAML 中的 `bf16` 改为 `fp16`。

## 8. 输出

每项任务默认输出：

```text
experiments/<experiment.name>/
├── best_model.pth
├── history.json
└── test/
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

## 9. 文件职责

```text
main.py                         配置解析、数据/模型构建、流程入口
dataloader.py                   时间排序、相邻帧配对、按时间切分
models/temporal_point_operator.py  瞬态点云算子
utils/training.py               训练、验证、checkpoint、自回归测试
config_surface_pressure.yml     表面压力配置
config_volume_velocity.yml      体场速度配置
```
