# `metrics/__init__.py`

## 作用

定义 `metrics` 包的公共接口。目前从 `field_metrics.py` 导出：

```python
FieldAccumulator
case_metrics
summarize_cases
```

因此可以使用：

```python
from metrics import FieldAccumulator, case_metrics, summarize_cases
```

`coefficient_metrics`、`statistics` 和 `efficiency` 未在这里统一导出，调用者需要从具体模块显式导入，避免公共命名空间过大。
