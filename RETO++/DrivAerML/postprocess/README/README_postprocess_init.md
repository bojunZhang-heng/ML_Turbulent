# `postprocess/__init__.py`

## 作用

该文件把 `postprocess/` 标记为 Python 包。目前只包含包级说明，不导入模型或绘图函数。

## 为什么保持为空

- 避免 `import postprocess` 时立即导入 PyTorch、VTK、Matplotlib 等重依赖；
- 避免模型目录尚未加入 `PYTHONPATH` 时产生循环导入；
- 各模块通过明确路径导入，便于定位依赖来源。

## 使用

通常不直接调用。它允许未来使用：

```python
import postprocess
```

真正的评价入口仍是 `evaluate_models.py`。
