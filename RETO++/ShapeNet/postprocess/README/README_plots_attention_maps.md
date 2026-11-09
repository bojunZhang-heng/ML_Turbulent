# `plots/attention_maps.py`

## 作用

绘制模型额外导出的 query、key、cluster 或 slice 空间权重，用于解释模型关注的几何区域。

## NPZ 输入格式

必须包含：

```text
coords: (N,3)
```

其余任意键可以是：

```text
(N,)   单通道空间权重
(N,K)  K 个 head/cluster/slice 通道
```

例如：

```python
np.savez(
    "attention.npz",
    coords=raw_coords,
    query_weight=query_weight,
    cluster_weight=cluster_weight,
)
```

## `plot_attention_npz(path, output_dir, index=0)`

对 `(N,K)` 数组使用 `index` 选择通道，为每个数组生成独立三维点云 PNG。

## 限制

本模块只负责画图，不会自动从模型内部恢复 attention。模型 forward 或 hook 必须先导出权重。数组第一维必须与 `coords` 节点一一对应。
