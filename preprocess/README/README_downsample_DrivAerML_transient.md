# DrivAerML 瞬态 PKL 下采样

对应脚本：`preprocess/downsample_DrivAerML_transient.py`。

该脚本用于处理如下拆分存储格式：

```text
run_1_pklFile/
├── geometry.pkl
├── normalization_scalars.pkl
├── 1.pkl
├── 2.pkl
└── ...
```

`geometry.pkl` 保存所有时间步共用的 `x_surface` 和 `x_volume`；编号 PKL
分别保存每个时间步的 `y_spressure` 和 `y_velocity`。

## 核心逻辑

脚本只生成一次面场索引和体场索引，并将其复用于全部时间步：

```text
surface_indices: x_surface <-> 所有时间步的 y_spressure
volume_indices : x_volume  <-> 所有时间步的 y_velocity
```

因此下采样后，不同时间步仍然对应完全相同的空间节点。脚本不会重新归一化
数据，也不会修改或删除输入文件。

输出 `geometry.pkl` 额外保存：

```text
surface_downsample_indices
volume_downsample_indices
```

可使用这两个字段将预测结果映射回原始完整点云。

## 使用方法

面场和体场均采样到 100000 点：

```bash
cd /Users/zhangbojun/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerML_transient.py \
  --input-dir preprocess/run_1_pklFile \
  --output-dir preprocess/run_1_pklFile_100k \
  --num-points 100000 \
  --seed 42
```

分别设置面场和体场数量：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/downsample_DrivAerML_transient.py \
  --input-dir preprocess/run_1_pklFile \
  --output-dir preprocess/run_1_pklFile_downsampled \
  --surface-points 100000 \
  --volume-points 500000 \
  --seed 42
```

已有输出时需要显式覆盖：

```text
--overwrite
```

`--surface-points` 和 `--volume-points` 分别覆盖 `--num-points`。如果未设置
`--num-points`，则两个独立参数必须同时提供。

## 输出格式

```text
run_1_pklFile_100k/
├── geometry.pkl                 # 下采样坐标和原始索引
├── normalization_scalars.pkl    # 原统计量及下采样元数据
├── 1.pkl                        # 下采样后的时间步物理场
├── 2.pkl
└── ...
```

归一化参数仍然是完整数据集的统计量。下采样只是选取已有节点，不能重新计算
mean/std 或 min/max，否则训练数据与物理量反归一化的尺度会发生变化。

## 内存和磁盘

原始 `geometry.pkl` 和单个时间步文件均约为 1.7 GB。脚本按顺序处理时间步，
但读取共享几何时仍需要数 GB 可用内存。建议使用随机采样；对于约 1.47 亿个
体场节点，体素采样会产生很大的临时数组，因此该脚本不提供体素模式。

输出目录必须和输入目录不同。`--overwrite` 只会清理输出目录中的编号 PKL、
`geometry.pkl`、`normalization_scalars.pkl` 和临时 PKL。
