# RETO（DrivAerStar）

本目录的 `RETO.py` 是独立的连续坐标 RoPE Transformer。输入为 DrivAerStar
预处理后的三维坐标，默认预测表面压强：

```text
输入:  (B, N, 3)
输出:  (B, N, 1)
```

坐标先经过连续正余弦编码和 MLP。每个 block 用独立 Q/K/V 投影，Q、K 使用
由坐标计算的 RoPE，V 不旋转，再进行全点 scaled dot-product attention 和
残差 MLP 更新。全点注意力的复杂度、显存均为 `O(N^2)`，超大网格建议降低
`n_layers`/`n_hidden` 或改用 `RETO_plus`。

## 运行

```bash
python main.py --config config_RETO.yml
```

可在配置中调整模型宽度、层数以及 `sincos_max_wavelength`、
`rope_max_wavelength`。RETO checkpoint 与 RETO_plus 不兼容，请重新训练。
