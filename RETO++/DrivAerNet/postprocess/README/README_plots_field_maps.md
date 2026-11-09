# `plots/field_maps.py`

## 作用

在同一汽车案例上绘制 ground truth、各模型 prediction 和 absolute error 三维点云图。

## `_scalar_view(values)`

- 标量场：直接展平；
- 三分量速度/WSS：转成矢量模长。

因此同一函数可用于压力、速度和 WSS。

## `plot_field_comparison(...)`

关键输入：

```python
case_id
coords                 # (N,3) 原始物理坐标
truth                  # (N,1) 或 (N,3)
predictions            # {label: prediction}
output
point_size
```

## 公平颜色尺度

- GT 和所有模型预测共用 field color limits；
- 所有模型 absolute error 共用 error color limits；
- limits 使用 1%～99% 分位数，降低单个异常点对色阶的破坏。

这样不同模型之间的颜色差异具有可比性，避免每个子图自动缩放造成视觉误导。

## 输出布局

第一行：GT 和各模型预测；第二行：公共说明和各模型绝对误差。坐标轴隐藏，使用固定观察角度。
