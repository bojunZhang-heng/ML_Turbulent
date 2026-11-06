# `metrics/field_metrics.py`

## 作用

计算反归一化物理场的回归指标，并支持不保存全部节点预测的流式累计。

## 辅助函数

### `_finite(value)`

把有限数值转成 Python `float`；NaN/Inf 转成 `None`，保证结果能安全写入 JSON。

### `_flatten(values)`

把标量场 `(N,)` 转成 `(N,1)`，并把任意前置 batch/空间维合并为 `(M,C)`。

## `case_metrics(prediction, target)`

计算单个汽车案例：

- relative L2；
- R²；
- MAE；
- RMSE；
- bias；
- Pearson r；
- maximum absolute error。

relative L2 定义为：

```text
||prediction - target||₂ / ||target||₂
```

## `FieldAccumulator`

流式保存充分统计量：样本数、和、平方和、预测-目标乘积、误差平方和等。每处理一个案例调用：

```python
accumulator.update(prediction, target)
```

最后调用：

```python
accumulator.finish(component_names)
```

返回：

- `overall`：所有节点及所有分量展平后的节点等权结果；
- `components`：每个物理分量独立结果；
- 额外包括 `nrmse_std` 和 `nrmse_range`。

## `summarize_cases(rows)`

先对每辆车计算指标，再汇总 mean、std、median、P95 和 worst。这里每辆车权重相同，不受点数影响。

## 两种聚合的区别

```text
FieldAccumulator overall：节点等权，点数多的案例贡献更大
summarize_cases：案例等权，每辆车贡献相同
```

模型论文对比建议同时报告两者。
