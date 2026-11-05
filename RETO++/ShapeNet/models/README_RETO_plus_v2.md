# RETO_plus_v2（ShapeNet-Car）

ShapeNet 输入为 `[XYZ, SDF, normal_x, normal_y, normal_z]` 七个通道。v2 对
前三个 XYZ 使用连续正余弦编码，对 SDF 和法向量使用轻量线性分支，再融合为
`n_hidden` 维特征；后续 RETO+ attention、RoPE、LayerScale 和 checkpoint 保持不变。

```bash
python main.py --config config_RETO_plus_v2.yml
```

连续编码最大波长由 `position_encoding.sincos_max_wavelength` 控制，v2 不能
直接加载旧 `RETO_plus` checkpoint。
