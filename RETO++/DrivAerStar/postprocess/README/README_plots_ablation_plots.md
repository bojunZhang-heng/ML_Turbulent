# `plots/ablation_plots.py`

## 作用

绘制 cluster 数、slice 数、模型层数或其他单一数值超参数与 relative L2 的关系。

## CSV 格式

```csv
model,setting,relative_l2
CANO,32,0.081
CANO,64,0.076
CANO,128,0.077
```

## `plot_ablation_csv(path, output)`

按模型分组，用折线和 marker 展示 `setting -> relative_l2`。

## 注意事项

- `setting` 必须可转换为数值；
- 同一条曲线上除目标超参数外，其余训练设置应保持不变；
- 当前实现保持 CSV 行顺序。如果希望 x 轴严格递增，应在生成 CSV 时按 setting 排序；
- 不同超参数含义不应混在同一张图中，例如不要把 cluster 数和 layer 数作为一条连续曲线。
