# `inference.py`

## ShapeNet 前向一致性

ShapeNet 的训练函数对 RETO_plus 和 RETO_plus_v2 使用
`model(normalized_input)`。后处理默认采用同一调用方式，让模型内部从归一化输入取得位置坐标；不会额外把反归一化真实坐标传入 `rope_coords`。

只有训练阶段也显式使用真实坐标的未来实验，才应在 YAML 中设置：

```yaml
postprocess:
  forward_mode: raw_coords
```

## 作用

执行一个模型的一次完整测试：重建 DataLoader、加载模型、推理、反归一化、累计指标、测量效率并返回绘图所需案例。

## 核心对象

### `OUTPUT_COMPONENTS`

规定任务输出维度：压力为 1，WSS 和速度为 3。

### `select_device(name)`

支持 `auto`、`cpu`、`cuda`、`mps`。`auto` 的优先级为 CUDA、MPS、CPU。

### `synchronize(device)`

CUDA/MPS 异步执行会导致直接计时不准确，因此推理计时前后同步设备。

### `_sample_scalars(all_scalars, sim_id)`

单数据集直接返回统一统计量；多车型数据从 `by_vehicle` 中选择当前车型的归一化参数。

### `_component_names(task, count)`

为输出分量生成可读名称，例如 `velocity_x/y/z`。

### `evaluate_run(...)`

主要流程：

1. 根据 YAML 重建测试 ID；
2. 创建 `batch_size=1` 的测试 DataLoader；
3. 构建模型并严格加载 checkpoint；
4. 配置 AMP，CUDA 下重置峰值显存；
5. 首批执行 warm-up；
6. 统一调用 `forward_model()`，必要时生成 SATO serialization order；
7. 将预测和目标从 mean-std 恢复到物理单位；
8. 计算全局、逐分量、逐案例、矢量方向与 bootstrap 指标；
9. 统计参数量、耗时、吞吐率和 CUDA 峰值显存；
10. 返回 summary、逐案例 rows 和一个场可视化案例。

运行时还会打印模型、任务、checkpoint、数据源、设备、AMP、参数量、测试 case ID，以及每个案例的全部指标。逐案例打印可由主程序的 `--no-print-per-case` 关闭。

## 返回值

```python
summary, case_rows, selected_case
```

`selected_case` 包含物理坐标、真实场和预测场。使用 `--save-predictions` 时还会保存每个案例的压缩 NPZ。

## 指标尺度

`normalized_mse` 仅用于训练尺度诊断；正式场指标全部使用 `denormalize_field()` 后的物理值。
