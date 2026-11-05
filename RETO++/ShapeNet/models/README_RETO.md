# RETO（ShapeNet-Car）

ShapeNet-Car 版本沿用 RETO 的连续坐标编码和 RoPE 注意力。当前数据加载器将
七个输入通道传给模型：

```text
输入:  (B, N, 7) = [x, y, z, sdf, normal_x, normal_y, normal_z]
输出:  (B, N, 4) = [velocity_x, velocity_y, velocity_z, pressure]
```

七个通道整体进入连续正余弦编码；每个 attention block 使用独立 Q/K/V，Q、K
使用由输入坐标生成的 RoPE，V 保持内容特征。模型执行全点 attention，因此
复杂度和注意力矩阵大小为 `O(N^2)`，大点云需要谨慎设置 `n_layers` 和
`n_hidden`。

## 运行

```bash
python main.py --config config_RETO.yml
```

`model.space_dim` 必须与 pickle 中的输入通道数一致（默认 7），`out_dim` 必须
为 4。`sincos_max_wavelength` 和 `rope_max_wavelength` 可在配置文件中调整。
RETO 与 RETO_plus/RETO_plus_v2 的 checkpoint 结构不同，请重新训练。

## Activation checkpoint

默认配置将 8 个 RETO block 的前 6 个按每 2 层一段执行 activation
checkpoint，最后 2 层正常保存激活：

```yaml
checkpoint:
  checkpoint_segment_size: 2
  checkpoint_num_blocks: 6
```

设 `checkpoint_num_blocks: 0` 可关闭，设为 8 可对全部层启用。更多 checkpoint
通常降低显存但增加反向计算时间；验证和测试阶段不会启用重计算。
