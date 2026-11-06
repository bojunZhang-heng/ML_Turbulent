# `plots/__init__.py`

## 作用

定义 `plots` 包最常用的公共绘图接口：

```python
plot_error_raincloud
plot_field_comparison
plot_efficiency_bubble
```

可以直接使用：

```python
from plots import plot_error_raincloud
```

气动系数、attention、scaling 和 ablation 属于可选功能，因此由主程序从具体文件显式导入。
