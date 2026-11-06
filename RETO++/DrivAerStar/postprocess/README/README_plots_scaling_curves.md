# `plots/scaling_curves.py`

## 作用

绘制输入点数增长时，推理时间和峰值显存的扩展趋势。

## CSV 格式

```csv
model,points,time_ms,memory_gb
Transolver,10000,12.3,1.8
Transolver,100000,78.4,7.2
LinearNO,10000,9.1,1.4
```

## `plot_scaling_csv(path, output_dir)`

生成：

```text
scaling_time.png
scaling_memory.png
```

x 轴为对数尺度的输入网格点数，y 轴分别为推理时间和 CUDA 峰值显存。

## 使用建议

每个点数应在相同 GPU、batch size、精度类型和 warm-up 策略下测量。若显存不足的模型没有数据点，应在论文表格中明确标记 OOM，而不是用 0 替代。
