# `model_registry.py`

## 作用

根据 YAML 中的 `model_name` 和超参数构建正确模型，并严格加载 checkpoint。显式文件兼容
旧实验；实验目录可以按照五种训练协议及两种选模指标自动选择最新 checkpoint。

## 支持模型

```text
Transolver
LinearNO
CANO
RETO
RETO_plus
RETO_plus_v2
SATO
```

## 核心函数

### `checkpoint_model_hint(state_dict)`

通过明确的权重键识别 RETO+ 编码器版本：

- `preprocess.omega`：`RETO_plus_v2` 连续正余弦编码；
- `preprocess.linear_pre/linear_post`：`RETO_plus` MLP 编码。

### `value(config, name, default)`

统一处理 YAML 中键缺失或值为 `None` 的情况。

### `build_model(model_name, cfg)`

为每种模型建立独立参数分支。公共参数包括 `space_dim`、`n_layers`、`n_hidden`、`n_head`、`mlp_ratio` 和 `out_dim`；模型专有参数只传给对应类。

特别地：

```text
RETO_plus     -> MLP preprocess
RETO_plus_v2  -> ContinuousSincosEmbed，额外使用 sincos_max_wavelength
```

### `load_checkpoint(model, path, expected_model_name)`

支持纯 state dict，以及包含 `model_state_dict`、`state_dict`、`model`、`model_state`、
`net` 或 `network` 的包装格式；支持去除 DataParallel 的 `module.`、compile 的
`_orig_mod.` 和统一的 `model.` 前缀。旧版完整 `nn.Module` 也可加载，但 pickle 文件必须可信。

### `resolve_checkpoint_path(...)`

显式 `.pth` 文件直接使用。传入实验目录时，按 `training_protocol`（五种协议之一）与
`selection_metric`（`val_loss` 或 `relative_error`）过滤，并选择时间戳最新的文件。

加载始终使用 `strict=True`。模型版本错配时给出明确错误，不会静默忽略 missing/unexpected keys。

## 注意事项

YAML 不只决定模型名称，还决定层数、隐藏维度、head 数和 cluster 数。因此即使模型名称相同，超参数不同也不能加载同一 checkpoint。
