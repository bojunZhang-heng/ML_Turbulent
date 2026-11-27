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

## 论文修图参数

具体绘图文件为：

```text
postprocess/plots/raincloud.py
```

文件顶部的以下标记区集中保存了论文排版相关参数：

```python
# ==================== PAPER FIGURE TUNING: START ====================
...
# ===================== PAPER FIGURE TUNING: END =====================
```

默认字体为 `Times New Roman`。可在该区域直接调整：

- `FONT_SIZE`、`AXIS_LABEL_SIZE`、`TICK_LABEL_SIZE`：正文字号、坐标轴标题和刻度字号；
- `FIGURE_MIN_WIDTH`、`WIDTH_PER_MODEL`、`FIGURE_HEIGHT`：画布尺寸；
- `VIOLIN_WIDTH`、`VIOLIN_ALPHA`：小提琴图宽度和透明度；
- `BOX_WIDTH`、`BOX_LINE_WIDTH`、`MEDIAN_LINE_WIDTH`：箱线图样式；
- `SCATTER_SIZE`、`SCATTER_ALPHA`：逐案例散点样式；
- `X_LABEL_ROTATION`：模型名称旋转角度；
- `GRID_ALPHA`：网格线透明度；
- `OUTPUT_DPI`：PNG 输出分辨率，论文通常使用 300～600 dpi。

字体设置通过 `plt.rc_context()` 仅作用于雨云图，不会改变后处理中的其他图。

## 边界处理

单案例或所有误差完全相同时，代码添加极小的仅绘图扰动，防止 violin 核密度估计失败；原始数据和 box/scatter 数值不变。
