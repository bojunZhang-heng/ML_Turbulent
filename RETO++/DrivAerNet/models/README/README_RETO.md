# RETO（DrivAerNet）

`RETO.py` 将 RETO 的连续坐标编码与 RoPE 注意力迁移到 DrivAerNet 的多车型
训练入口中。模型使用每个样本的三维归一化坐标：

```text
输入:  (B, N, 3)
输出:  (B, N, 1)  pressure
```

连续正余弦编码提供坐标基函数；每个 block 使用独立 Q/K/V，Q、K 采用坐标
RoPE 后进行全点 scaled dot-product attention，随后进行残差 MLP。它不使用
车型分区矩阵，也不需要额外的 `rope_coords` 参数。

RETO 是全点注意力，时间和显存复杂度为 `O(N^2)`。当节点数较大时，建议
减小 `n_layers`、`n_hidden` 或使用 `RETO_plus`。

## 运行

```bash
python main.py --config config_RETO.yml
```

车型选择、每车型样本数和数据划分在 YAML 的 `data.vehicle_datasets` 中设置。
`sincos_max_wavelength` 与 `rope_max_wavelength` 位于
`position_encoding`。该模型的 checkpoint 不能与 RETO_plus 直接复用。

## Activation checkpoint

RETO 使用分段 activation checkpoint 平衡显存与速度。默认 8 层模型重计算
前 6 层，每段包含 2 层，最后 2 层保持普通 autograd：

```yaml
checkpoint:
  checkpoint_segment_size: 2
  checkpoint_num_blocks: 6
```

设 `checkpoint_num_blocks: 0` 可关闭；设为与 `n_layers` 相同可获得更大的激活
显存节省，但训练更慢。该功能只在训练模式启用，验证和测试自动走普通前向。
