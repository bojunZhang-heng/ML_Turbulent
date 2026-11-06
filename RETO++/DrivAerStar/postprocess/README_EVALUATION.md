# DrivAerStar 模型评价与论文风格出图

每个 Python 子文件的逐文件说明位于 `README/`，总索引为
`README/README_INDEX.md`。

本目录把模型构建、测试集推理、指标计算和出图分开。主入口是
`evaluate_models.py`，可以一次比较多个 checkpoint。所有场误差先通过
`normalization_scalars.pkl` 反归一化，再在真实物理量尺度上计算。

## 三篇论文如何评价模型

### 1. Cluster Attention Neural Operators（CANO）

论文采用的评价方法：

1. **测试场相对 L2 误差**：对每个测试样本计算 relative L2，比较模型平均精度和误差分布。
2. **逐样本稳定性**：不是只报告均值，还展示中位数、四分位数、长尾和离群样本。
3. **物理场案例对比**：同时比较 ground truth、prediction 与 absolute error，观察局部结构是否恢复。
4. **聚类可解释性**：可视化 CANO cluster assignment，并与 Transolver 的 slice assignment 比较。
5. **频域物理一致性**：在湍流任务上比较 turbulent kinetic energy spectrum；预测均值曲线同时给出 `mean ± 3 std` 阴影。
6. **长时 rollout**：按时间步统计 relative L2，评价误差累积和长期稳定性。
7. **数据效率**：改变训练样本数量，比较小数据条件下的误差。
8. **聚类数消融**：改变 cluster number，分析精度、表达能力与成本的关系。
9. **参数量**：与基线比较模型规模。

论文中的主要图：

- relative L2 **雨云图**：半 violin、boxplot 与 jitter scatter 组合；
- ground truth / prediction / absolute error 场图；
- cluster assignment 与 slice assignment 空间激活图；
- TKE spectrum 曲线和 `mean ± 3 std` 阴影；
- rollout 的 time-step error curve、短期/长期误差雨云图；
- 不同时间步的 absolute error 场图；
- training samples、cluster number 等消融表或曲线。

本框架对应实现：`plots/raincloud.py`、`plots/field_maps.py`、
`plots/attention_maps.py` 和 `plots/ablation_plots.py`。

注意：DrivAerStar 的汽车表面是不规则点云，表面压力任务没有规则波数空间，因此不能直接照搬
论文中的 TKE spectrum。只有评价具有合理空间频谱定义的速度/涡量场时才建议加入该图。

### 2. Transolver is a Linear Transformer（LinearNO）

论文采用的评价方法：

1. **物理场 relative L2**：跨 PDE、汽车和翼型任务比较场预测误差。
2. **气动系数 relative L2**：汽车使用阻力系数 `CD`，翼型使用升力系数 `CL`。
3. **排序能力**：使用 Spearman rank correlation，分别报告 `rho_D` 或 `rho_L`；这反映模型能否正确排序设计优劣。
4. **参数量和 GFLOPs**：同时评价预测精度与计算代价。
5. **attention rank**：逐层计算 attention matrix 的平均秩，分析注意力表达能力。
6. **slice number 消融**：改变物理 token/slice 数，比较预测误差。
7. **Softmax 维度消融**：比较 Q、K 在不同维度归一化的效果。
8. **OOD 泛化**：测试训练范围外的 Reynolds number 和 angle of attack。
9. **超分辨率与稀疏采样**：改变输入采样率，以 relative MAE 等指标评价跨分辨率能力。

论文中的主要图：

- Query weight map、Key weight map、Transolver slice map；
- 选定 Q/K pair 的空间权重图；
- prediction/error distribution；
- 每层 attention matrix average rank 曲线；
- slice number 与 prediction error 曲线；
- GT / LinearNO prediction / Transolver prediction / error maps；
- AirfRANS、ShapeNet 等任务的场可视化；
- OOD 和采样率对比表。

本框架用 `metrics/coefficient_metrics.py` 计算 coefficient relative L2 与 Spearman rho，
`plots/coefficient_plots.py` 绘制 coefficient scatter 和排序曲线，
`plots/attention_maps.py` 绘制外部导出的 Q/K/slice 权重，
`plots/ablation_plots.py` 绘制 slice 或其他超参数消融。

attention matrix rank 需要模型在 forward 中额外导出逐层 attention。本框架不修改训练模型，因而没有
凭空重建该矩阵；应先从模型导出权重，再做独立秩分析。

### 3. Transolver

论文采用的评价方法：

1. **物理场 relative L2**：评价 volume field 和 surface field。
2. **气动系数误差与排序**：ShapeNet-Car 报告 `CD` relative L2 和 Spearman `rho_D`；AirfRANS 报告 `CL` relative L2 和 `rho_L`。
3. **效率评价**：参数量、GPU memory、running time 三者联合比较。
4. **注意力有效性**：计算 learned attention 与 uniform distribution 之间的 KL divergence。
5. **切片可解释性**：可视化 learned physical slices 和 attention heatmap。
6. **网格鲁棒性**：在原网格和随机重采样网格上比较 slice 与预测结果。
7. **规模扩展**：改变 mesh resolution、训练数据量和模型参数量，报告精度曲线。
8. **多次实验不确定性**：报告 mean ± std 或 confidence interval，而不是单次最好结果。
9. **OOD 条件**：在训练范围外的 Reynolds number、AoA 等条件测试。
10. **任务专用指标**：例如 WaterDrop 使用 position MSE。

论文中的主要图：

- learned slice visualization；
- 原网格与随机重采样网格的 slice maps；
- attention heatmap；
- **效率气泡图**：x 为 running time、y 为 GPU memory、bubble size 为 parameter count；
- ground truth / prediction / absolute error case study；
- resolution、data、model parameter scalability curves；
- GPU memory/time 随 input mesh points 的变化曲线；
- OOD 表格以及多次实验 mean ± std 表。

本框架对应实现为 `metrics/efficiency.py`、`plots/efficiency_bubble.py`、
`plots/scaling_curves.py`、`metrics/statistics.py`、`plots/field_maps.py` 和
`plots/attention_maps.py`。

## 建议用于 DrivAerStar 的核心评价集合

不能只按一个平均 relative L2 排名。建议固定报告：

1. **global relative L2**：把所有节点合并，节点等权；
2. **case-equal mean / median / P95 / worst relative L2**：每辆车等权，P95 反映困难车型；
3. **R2、MAE、RMSE、bias、Pearson r**：分别描述拟合、物理单位误差、系统偏差和线性相关性；
4. **bootstrap 95% CI**：量化测试样本有限导致的不确定性；
5. **paired bootstrap difference**：同一批汽车上比较两个模型，而不是比较两个无关均值；
6. **参数量、推理时间、峰值 CUDA 显存、points/s**；
7. **误差雨云图、典型场图、效率气泡图**；
8. 如果有 `CD/CL`，再加入 coefficient relative L2 和 Spearman rho。

三分量速度或 WSS 还会报告各分量误差、矢量模长误差、平均 cosine similarity 和平均方向角误差。

## 快速使用

在 `RETO++/DrivAerStar` 目录运行。单模型兼容旧用法：

```bash
python postprocess/evaluate_models.py \
  --config config_RETO.yml \
  --checkpoint experiment_DrivAerStar_RETO/best_model.pth
```

推荐一次比较多个模型：

```bash
python postprocess/evaluate_models.py \
  --run Transolver config_Transolver.yml checkpoints/transolver.pth \
  --run LinearNO config_LinearNO.yml checkpoints/linearno.pth \
  --run CANO config_CANO.yml checkpoints/cano.pth \
  --output-dir postprocess_results
```

Windows PowerShell：

```powershell
python postprocess\evaluate_models.py `
  --run Transolver config_Transolver.yml checkpoints\transolver.pth `
  --run LinearNO config_LinearNO.yml checkpoints\linearno.pth `
  --run CANO config_CANO.yml checkpoints\cano.pth `
  --output-dir postprocess_results
```

快速测试前 3 个测试 case，并覆盖 YAML 中的数据路径：

```powershell
python postprocess\evaluate_models.py `
  --run LinearNO config_LinearNO.yml checkpoints\linearno.pth `
  --data-path D:\DrivAerStar\pklFile_10k `
  --max-test-samples 3
```

必须使用训练 checkpoint 对应的 YAML。多模型运行时，程序会验证任务和测试 case ID 完全一致；不一致会
立即报错。推理计时前执行一次 warm-up，CUDA 计时前后执行同步。

## 输出

默认输出到 `postprocess_results/`：

```text
summary.json
per_case_metrics.csv
figures/error_raincloud.png
figures/field_comparison_<case>.png
figures/efficiency_bubble.png
```

使用 `--save-predictions` 时，每个 case 的物理坐标、真实场和预测场保存为压缩 NPZ。

## 可选的气动系数评价

当前处理后的 DrivAerStar PKL 只有 `x_surface`、`y_spressure`、`y_wss`、`x_volume` 和
`y_velocity`，没有表面法向量、单元面积、参考迎风面积及完整来流条件。因此不能只靠当前 PKL 严格完成
压力/摩擦力积分并计算 `CD`。

如果已有真实和预测系数，可提供 CSV：

```csv
case_id,true_coefficient,Transolver,LinearNO,CANO
101,0.281,0.286,0.279,0.283
102,0.304,0.298,0.302,0.306
```

模型列名必须与 `--run LABEL` 一致。程序会增加 `coefficient_scatter.png`、
`coefficient_rank.png`，并在 JSON 中写入 coefficient relative L2、MAE 和 Spearman rho。
如果要由预测场直接积分 `CD`，预处理阶段还需保存 surface normals、cell areas、reference area、
freestream density/velocity，并明确压力系数或绝对压力的定义。

## 可选的 attention、规模扩展与消融图

attention NPZ：

```text
coords: (N, 3)
query_weight: (N,) 或 (N, K)
key_weight: (N,) 或 (N, K)
cluster_weight: (N,) 或 (N, K)
```

通过 `--attention-npz attention.npz --attention-index 3` 出图。

规模测试 CSV：

```csv
model,points,time_ms,memory_gb
Transolver,10000,12.3,1.8
Transolver,100000,78.4,7.2
```

通过 `--scaling-csv scaling.csv` 输出 time/memory scaling curves。

消融 CSV：

```csv
model,setting,relative_l2
CANO,32,0.081
CANO,64,0.076
CANO,128,0.077
```

通过 `--ablation-csv ablation.csv` 输出消融曲线。

## 文件分工

- `evaluate_models.py`：CLI、多个模型协调、统一校验和结果落盘；
- `data_split.py`：读取 YAML 并重建测试划分；
- `model_registry.py`：按训练 YAML 构建模型并加载 checkpoint；
- `inference.py`：统一 forward、反归一化、计时和逐 case 收集；
- `metrics/field_metrics.py`：场指标；
- `metrics/coefficient_metrics.py`：气动系数与 Spearman；
- `metrics/statistics.py`：bootstrap CI 和配对比较；
- `metrics/efficiency.py`：参数量和 checkpoint 大小；
- `plots/`：所有论文风格图。
