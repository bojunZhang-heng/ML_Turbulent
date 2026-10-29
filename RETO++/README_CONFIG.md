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
