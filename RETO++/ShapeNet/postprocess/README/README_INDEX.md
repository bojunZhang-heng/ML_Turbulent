# DrivAerStar postprocess 文件索引

`postprocess` 是独立于训练循环的模型评价模块。整体调用链为：

```text
evaluate_models.py
├── data_split.py                 读取 YAML、重建测试集
├── model_registry.py             构建模型、检查并加载 checkpoint
├── inference.py                  推理、反归一化、计时、收集逐案例结果
├── metrics/                      数值指标和统计推断
└── plots/                        论文风格出图
```

## 主目录文件

| 代码文件 | README | 作用 |
|---|---|---|
| `__init__.py` | `README_postprocess_init.md` | 声明 postprocess Python 包 |
| `evaluate_models.py` | `README_evaluate_models.md` | 命令行主入口、多模型比较和输出协调 |
| `data_split.py` | `README_data_split.md` | 展平 YAML 并重建测试样本 ID |
| `model_registry.py` | `README_model_registry.md` | 按 YAML 构建模型并严格加载权重 |
| `inference.py` | `README_inference.md` | 执行单模型完整评价流程 |

## metrics 子目录

| 代码文件 | README | 作用 |
|---|---|---|
| `metrics/__init__.py` | `README_metrics_init.md` | 导出常用场指标接口 |
| `metrics/field_metrics.py` | `README_metrics_field_metrics.md` | 场误差、流式全局统计和案例统计 |
| `metrics/statistics.py` | `README_metrics_statistics.md` | bootstrap 置信区间和配对比较 |
| `metrics/coefficient_metrics.py` | `README_metrics_coefficient_metrics.md` | CD/CL 误差和 Spearman 排序相关性 |
| `metrics/efficiency.py` | `README_metrics_efficiency.md` | 参数量和 checkpoint 大小 |

## plots 子目录

| 代码文件 | README | 作用 |
|---|---|---|
| `plots/__init__.py` | `README_plots_init.md` | 导出三个常用绘图函数 |
| `plots/raincloud.py` | `README_plots_raincloud.md` | 逐案例 relative L2 雨云图 |
| `plots/field_maps.py` | `README_plots_field_maps.md` | GT、预测、绝对误差三维点云图 |
| `plots/efficiency_bubble.py` | `README_plots_efficiency_bubble.md` | 时间、显存、参数量气泡图 |
| `plots/coefficient_plots.py` | `README_plots_coefficient_plots.md` | 气动系数散点图和排序曲线 |
| `plots/attention_maps.py` | `README_plots_attention_maps.md` | 外部 attention/cluster/slice NPZ 可视化 |
| `plots/scaling_curves.py` | `README_plots_scaling_curves.md` | 点数与时间/显存扩展曲线 |
| `plots/ablation_plots.py` | `README_plots_ablation_plots.md` | cluster/slice/layer 消融曲线 |

论文评价方法、完整命令和输入格式总说明仍见上一级 `README_EVALUATION.md`。
