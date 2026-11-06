# `evaluate_models.py`

## 作用

整个后处理系统的主函数。它解析命令行参数，依次评价一个或多个模型，验证比较条件，保存 JSON/CSV，并调用各绘图模块。

## 核心函数

### `parse_args()`

支持两种运行方式：

```text
单模型：--config CONFIG --checkpoint CHECKPOINT
多模型：--run LABEL CONFIG CHECKPOINT（可以重复）
```

还支持数据路径覆盖、设备、AMP、测试样本上限、案例可视化、气动系数 CSV、attention NPZ、scaling CSV 和 ablation CSV。

默认会把运行配置、展平后的 YAML、测试 case ID、逐案例指标、完整模型指标、效率数据和模型间配对比较打印到终端。使用 `--no-print-per-case` 可以关闭逐案例行，但保留汇总输出。

运行结束还会枚举输出目录中的全部 JSON、CSV、NPZ 和 PNG，打印绝对路径及文件大小。

### `_write_case_csv(path, all_rows)`

把所有模型的逐案例指标合并写入 `per_case_metrics.csv`。每行对应一个模型的一个测试案例。

### `_validate_comparison(summaries)`

公平性检查：所有模型必须具有相同任务名，并且测试 case ID 及顺序完全一致，否则停止比较。

### `main()`

执行顺序：

1. 读取每个模型 YAML；
2. 调用 `inference.evaluate_run()`；
3. 做配对 bootstrap 模型比较；
4. 可选读取 CD/CL CSV；
5. 写 `summary.json` 和 `per_case_metrics.csv`；
6. 生成雨云图、场图、效率图及可选图。

## 主要输出

```text
postprocess_results/
├── summary.json
├── per_case_metrics.csv
├── predictions/                  # 使用 --save-predictions 时
└── figures/
```

## 注意事项

- checkpoint 必须使用训练时对应的 YAML；
- 多模型必须使用同一测试划分；
- 场指标由 `inference.py` 在反归一化后的物理单位中计算；
- 当前文件负责流程协调，不直接实现指标公式。

## 重定向到日志

macOS/Linux：

```bash
python postprocess/evaluate_models.py ... > postprocess.log 2>&1
```

如果希望运行时在终端显示，同时写日志：

```bash
python postprocess/evaluate_models.py ... 2>&1 | tee postprocess.log
```

PowerShell：

```powershell
python postprocess\evaluate_models.py ... *> postprocess.log
```

进度条由 `tqdm` 输出到 stderr，因此需要同时重定向 stdout 和 stderr。
