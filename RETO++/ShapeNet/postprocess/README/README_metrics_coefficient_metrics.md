# `metrics/coefficient_metrics.py`

## 作用

评价阻力系数 `CD`、升力系数 `CL` 等每案例标量，并衡量车型排序能力。

## `_rankdata(values)`

将数值转换为秩。相同值使用平均秩，行为适合 Spearman 相关系数。

## `spearman_rho(first, second)`

先分别转成秩，再计算 Pearson correlation。它评价排序一致性，而不是数值误差。

```text
rho = 1：排序完全一致
rho = 0：没有单调排序关系
rho = -1：排序完全相反
```

## `read_coefficient_csv(path, labels)`

CSV 必须包含：

```csv
case_id,true_coefficient,Transolver,LinearNO,CANO
101,0.281,0.286,0.279,0.283
```

其中模型列名必须与命令行 `--run LABEL` 一致。

每个模型输出：

- coefficient relative L2；
- MAE；
- Spearman rho。

函数同时返回绘图所需的 case ID、真实系数和各模型预测数组。

## 限制

本模块不会由压力场自动积分 CD。若要积分，需要表面法向量、单元面积、参考面积、来流密度/速度以及明确的压力定义。
