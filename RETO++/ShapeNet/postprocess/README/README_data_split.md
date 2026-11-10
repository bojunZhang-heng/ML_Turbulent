# `data_split.py`

## 作用

读取训练 YAML，并用与训练阶段相同的规则重建测试集，保证后处理没有随意选择测试样本。

## 核心对象

### `CONFIG_SECTIONS`

列出需要展平的 YAML 分组：`experiment`、`model`、`attention`、`position_encoding`、`serialization`、`checkpoint`、`training` 和 `data`。

### `load_config(path)`

返回：

```python
raw_config, flat_config
```

- `raw_config` 保留 YAML 层级，用于读取 `vehicle_datasets`；
- `flat_config` 把各分组键合并，便于按 `cfg["n_hidden"]` 取值。

加载前会检查 Tab，并在 YAML 缩进或语法错误时打印出错行前后三行。YAML 层级统一使用两个空格，不能使用 Tab。

### `resolve_test_data(raw_cfg, cfg, override=None)`

支持：

1. ShapeNet 单数据目录：默认严格复现 `main.py` 的模十划分，即排序后位置 `0-6` 为训练、`7` 为验证、`8-9` 为测试；
2. 多车型目录：调用 `discover_multi_vehicle_samples()`，返回车型到目录的映射和组合测试 ID；
3. `override`：用命令行 `--data-path` 覆盖 YAML 路径。

如需对历史实验显式使用通用随机比例划分，可在 YAML 中设置：

```yaml
postprocess:
  postprocess_split_mode: random_ratio
```

默认值 `shapenet_modulo_7_1_2` 与当前 ShapeNet 训练主函数一致。任务 YAML 中即使存在 `train_ratio` 等字段，也不会覆盖这个默认训练划分。

### `display_case_id(sim_id)`

普通 ID 转为字符串；多车型 `(vehicle, sample_id)` 转为 `vehicle/sample_id`。

## 关键保证

模十划分只依赖排序后的有效样本 ID，结果稳定且与训练一致。选择 `random_ratio` 时，随机划分使用固定 seed，不依赖 NumPy/PyTorch 的全局随机状态；比例必须非负且总和为 1。
