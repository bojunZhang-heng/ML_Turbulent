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

## Activation checkpoint

默认对 8 层中的前 6 层启用 checkpoint，每 2 层组成一个重计算分段：

```yaml
checkpoint:
  checkpoint_segment_size: 2
  checkpoint_num_blocks: 6
```

`checkpoint_num_blocks: 0` 表示关闭；提高到 8 会进一步降低训练显存，但增加
反向重计算时间。checkpoint 只在 `model.train()` 阶段生效，验证和测试不重算。
