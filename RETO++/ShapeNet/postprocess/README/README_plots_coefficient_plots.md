# `plots/coefficient_plots.py`

## 作用

根据 `coefficient_metrics.py` 读取的数据生成两张气动系数图。

## `plot_coefficients(data, metrics, output_dir)`

### `coefficient_scatter.png`

- x：真实 CD/CL；
- y：预测 CD/CL；
- 虚线：理想的 `prediction = truth`；
- 图例：模型名和 Spearman rho。

散点越靠近对角线，系数数值预测越准确。

### `coefficient_rank.png`

先按真实系数升序排列案例，再绘制 GT 和各模型预测曲线。用于观察模型是否能正确保留设计优劣顺序，以及在哪些系数区间偏差最大。

## 输入来源

主程序通过 `--coefficient-csv` 读取外部 CSV。当前 DrivAerStar PKL 缺少直接积分 CD 所需的完整几何和来流信息，因此本文件不从场预测自行计算系数。
