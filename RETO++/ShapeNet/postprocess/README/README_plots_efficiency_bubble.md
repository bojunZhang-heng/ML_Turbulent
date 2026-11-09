# `plots/efficiency_bubble.py`

## 作用

绘制 Transolver 论文风格的计算效率气泡图。

## `plot_efficiency_bubble(summaries, output)`

从每个模型 summary 读取：

```text
x 轴：milliseconds_per_case
y 轴：peak_cuda_memory_gb
气泡面积：parameter_count
文字标注：模型名和 case-equal mean relative L2
```

## 解释方式

理想模型位于左下区域：推理快且显存低；同时标注的 relative L2 越低越好。气泡越小表示参数量越少。

## CPU/MPS 注意事项

PyTorch 当前没有与 CUDA 完全等价的统一峰值显存接口。本脚本把 CPU/MPS 的 y 值放在 0，并在 y 轴标签中明确说明。因此正式 GPU 效率对比应在同一 CUDA GPU、同一 batch 和同一测试点数上运行。
