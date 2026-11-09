# `metrics/statistics.py`

## 作用

用非参数 bootstrap 量化测试集有限样本带来的不确定性，并对两个模型做配对比较。

## `bootstrap_mean_ci(values, samples=2000, confidence=0.95, seed=42)`

对案例指标进行有放回重采样，每次计算均值，最后取分位数置信区间。

返回示例：

```python
{
    "mean": 0.081,
    "lower": 0.076,
    "upper": 0.087,
    "confidence": 0.95,
    "bootstrap_samples": 2000,
}
```

当 `samples=0` 时只返回原始均值，不计算上下界。

## `paired_bootstrap_difference(first, second, samples=2000, seed=42)`

输入必须是一一对应的同一批案例指标。先计算：

```text
difference[i] = first[i] - second[i]
```

再对 difference 做 bootstrap。返回平均差、95% CI 和 `probability_first_better`。

对于 relative L2 等越小越好的指标，差值小于 0 表示第一个模型更好。

## 注意事项

- 配对数组顺序必须相同；主程序已通过 case ID 校验保证这一点；
- bootstrap CI 是测试案例抽样不确定性，不等同于多随机种子训练的方差；
- 若要报告训练稳定性，仍需独立训练多次。
