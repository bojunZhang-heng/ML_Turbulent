# `metrics/efficiency.py`

## 作用

提供模型静态规模统计。

## `model_size_summary(model, checkpoint)`

返回：

```python
{
    "parameter_count": ...,            # 全部参数
    "trainable_parameter_count": ...,  # requires_grad=True 参数
    "checkpoint_size_mb": ...,         # 磁盘文件大小 MiB
}
```

## 与推理效率的关系

该文件只计算静态规模。下面这些运行时指标由 `inference.py` 负责：

- inference seconds；
- milliseconds per case；
- points per second；
- peak CUDA memory。

checkpoint 文件大小不一定等于模型推理显存，因为文件可能还包含优化器状态、不同数据类型或压缩格式；当前 loader 主要面向模型 state dict。
