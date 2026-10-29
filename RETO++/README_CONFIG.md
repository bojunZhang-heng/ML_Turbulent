# RETO++ 分组 YAML 超参数说明

`DrivAerNet/main.py` 和 `ShapeNet/main.py` 中的模型结构不再使用写死的
`n_layers=8`、`n_hidden=256` 等数值。模型初始化参数统一由对应 YAML
提供。

## 配置分类

```yaml
experiment:
  exp: ShapeNet_RETO_plus
  model_name: RETO_plus
  predicted_feature_name: p_and_v
  phase: restart_train

model:
  space_dim: 7
  out_dim: 4
  n_layers: 8
  n_hidden: 256
  dropout: 0.0
  act: gelu
  fun_dim: 0
  ref: 8
  unified_pos: false
  isregular: false

attention:
  n_head: 8
  mlp_ratio: 2
  key_ratio: 1
  cluster_num: 64
  cluster_temperature_init: 1.0
  layer_scale_init: 0.1

position_encoding:
  num_position_frequencies: 6

serialization:
  patch_size: 20
  shift: 2
  n_iter: 1

checkpoint:
  checkpoint_segment_size: 2
  checkpoint_num_blocks: 6

training:
  num_epochs: 200
  eval_freq: 1
  learning_rate: 5.0e-4
  pressure_weight: 0.5
  batch_size: 1
  shuffle: true

data:
  data_path: /path/to/pkl_data
  vtk_data_path: /path/to/raw_data
```

每个模型只需保留它实际使用的分类和字段。例如 Transolver 使用
`slice_num`，RETO+ 使用 `cluster_num` 和 checkpoint 参数，SATO 使用
`patch_size/shift/n_iter`。

## 已适配的训练入口模型

DrivAerNet：

```text
Transolver
LinearNO
RopeLinearNO_v3
CANO
RETO_plus
CANO_GatedRope
LinearNO_se
SATO
```

ShapeNet：

```text
Transolver
LinearNO
CANO
RETO_plus
```

相应的 `config_<model>.yml` 均包含完整模型结构参数。旧的扁平 YAML
仍可被读取，但建议新实验全部使用分组格式。

程序启动时会原样打印分组 YAML，训练日志因此会保留本次实验使用的
结构、注意力、checkpoint、训练和数据配置。

## 运行方法

DrivAerNet：

```bash
cd /Users/zhangbojun/ML_Turbulent/RETO++/DrivAerNet
/opt/anaconda3/envs/pgd-no/bin/python main.py --config config_RETO_plus.yml
```

ShapeNet：

```bash
cd /Users/zhangbojun/ML_Turbulent/RETO++/ShapeNet
/opt/anaconda3/envs/pgd-no/bin/python main.py --config config_RETO_plus.yml
```

命令行参数仍可以临时覆盖 YAML。例如：

```bash
python main.py --config config_RETO_plus.yml --n_hidden 192 --n_layers 6
```

但正式实验应把最终参数写回 YAML，以保证结果可复现。

## DrivAerNet 多车型抽样

`DrivAerNet/main.py` 支持在一次实验中选择多个车型，并为每种车型指定
不同的样本数。各车型的 PKL 都可以从 `1.pkl` 开始编号，dataloader
内部使用 `(车型名, 数字ID)` 作为唯一键，不会发生同名文件冲突。

```yaml
data:
  dataset_root: /data/zhangbojun/DrivAerNet_dataset
  selection_mode: random
  selection_seed: 42
  train_ratio: 0.6
  val_ratio: 0.3
  test_ratio: 0.1

  vehicle_datasets:
    E_S_WWC_WM:
      path: E_S_WWC_WMpklFile
      count: 200
      enabled: true
    E_S_WW_WM:
      path: E_S_WW_WMpklFile
      count: 100
      enabled: true
    F_S_WWC_WM:
      path: F_S_WWC_WMpklFile
      count: 150
      enabled: false
```

字段含义：

- `dataset_root`：所有车型目录的共同父目录；
- `path`：相对于 `dataset_root` 的目录，也可以使用绝对路径；
- `count`：本次实验从该车型选择的总样本数，包括 train/val/test；
- `enabled`：是否加入本次实验；
- `selection_mode: random`：先按固定种子随机排列，再取前 `count` 个；
- `selection_mode: first`：直接选择数字 ID 最小的前 `count` 个；
- `selection_seed`：控制随机抽样，参数相同即可复现实验；
- `train_ratio/val_ratio/test_ratio`：在每一种车型内部独立划分。

例如 `count: 100` 和 `0.6/0.3/0.1` 会为该车型生成 60 个训练、
30 个验证和 10 个测试样本。设置 `count: all` 可使用该目录中的全部
编号 PKL。请求数量超过目录实际样本数时程序会直接报错，避免静默地
少用数据。

每个车型目录必须包含自己的 `normalization_scalars.pkl`。训练目标仍
使用各 PKL 中已经归一化的数据；验证和测试计算物理尺度相对误差时，
程序会根据复合 ID 自动选取该车型对应的 mean/std，不会错误地使用第
一个目录的统计量。

不同车型的点数通常不同，因此多车型训练建议使用：

```yaml
training:
  batch_size: 1
```

当前 `config_RETO_plus.yml` 已列出以下八种车型：

```text
E_S_WWC_WM
E_S_WW_WM
F_S_WWC_WM
F_S_WWS_WM
N_S_WWS_WM
N_S_WWC_WM
N_S_WW_WM
F_D_WM_WW
```
