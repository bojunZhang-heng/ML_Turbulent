# ShapeNetCar 复现说明

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate LinearNO
python main.py --config config.yml
```

默认的 `config.yml` 已指向 `/data/zhangbojun/mlcfd_data`。首次运行会完成以下工作：

1. 将 `training_data/param0.tar.gz` 到 `param8.tar.gz` 解压到原目录。
2. 从 VTK 文件生成 `preprocessed_data` 缓存。此步骤耗时较长，但只需执行一次。
3. 使用 `param1` 到 `param8` 训练 200 epochs，以 `param0` 作为验证/测试集。
4. 将模型保存到 `outputs/shapenet/fold0`，将预测和指标保存到 `results/shapenet/fold0`。

常用模式在 YAML 顶部的 `mode` 中设置：

- `prepare`：只解压、预处理和检查数据。
- `train`：只训练。
- `evaluate`：加载 `evaluation.checkpoint` 测试。
- `train_and_evaluate`：训练后立即测试，默认值。

若只想先检查数据流程，可把 `mode` 改为 `prepare`。若显存不足，不要增大 `batch_size`；当前 ShapeNetCar 模型仅支持 batch size 1。更换留一折时，应同时修改 `dataset.fold_id`、`output.directory` 和 `evaluation.results_directory`。
