# `plots/raincloud.py`

## 作用

绘制 CANO 论文风格的逐案例 relative L2 分布图，将三种统计表达叠加：

- 左半 violin：概率密度；
- 中间 boxplot：中位数和四分位范围；
- 右侧 jitter scatter：每个真实测试案例。

## `plot_error_raincloud(values_by_model, output, seed=42)`

输入格式：

```python
{
    "Transolver": [0.08, 0.09, 0.07],
    "LinearNO": [0.07, 0.08, 0.06],
}
```

`seed` 只控制散点水平方向的视觉抖动，不改变指标。

## 输出

PNG，默认由主程序保存为：

```text
figures/error_raincloud.png
```

## 边界处理

单案例或所有误差完全相同时，代码添加极小的仅绘图扰动，防止 violin 核密度估计失败；原始数据和 box/scatter 数值不变。
