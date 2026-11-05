# RETO_plus_v2（DrivAerNet）

保留 RETO+ 的 cluster attention、连续 RoPE、LayerScale 和 checkpoint，
仅将 XYZ 输入预处理从 MLP 替换为三维连续正余弦编码。

输入为归一化坐标 `(B,N,3)`，输出为表面压强 `(B,N,1)`。

```bash
python main.py --config config_RETO_plus_v2.yml
```

连续编码的最大波长由 `position_encoding.sincos_max_wavelength` 控制。由于
输入编码结构变化，不能直接加载旧 `RETO_plus` checkpoint。
