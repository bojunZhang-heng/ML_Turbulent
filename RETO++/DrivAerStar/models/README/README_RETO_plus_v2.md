# RETO_plus_v2

`RETO_plus_v2` 保留 `RETO_plus` 的 cluster attention、连续旋转位置编码
（RoPE）、LayerScale 和 activation checkpointing，只替换输入预处理编码。

## 输入编码

原始 `RETO_plus` 的输入路径是：

```text
XYZ (B,N,3) -> MLP -> hidden features (B,N,n_hidden)
```

v2 改为：

```text
XYZ (B,N,3) -> ContinuousSincosEmbed -> hidden features (B,N,n_hidden)
```

连续编码对每个坐标轴使用多尺度正弦和余弦基函数：

```text
sin(x * omega), cos(x * omega)
```

其中 `omega` 由 `sincos_max_wavelength` 控制。`n_hidden=256` 时，编码会自动
处理3个坐标轴不能整除的情况，并通过零填充得到精确的256维输出。

这里使用数据集已经完成的 min-max 归一化坐标；不会在模型内部重新归一化，
也不会改变物理量的 mean-std 归一化。

## 注意力部分

v2 仍使用 `RETO_plus.py` 中的：

- 独立 Q/K/V 投影；
- 内容驱动的 cluster assignment；
- 连续三维 RoPE 对 Q/K 的旋转；
- cluster-to-node cross attention；
- LayerScale 残差稳定化；
- 分段 activation checkpointing。

因此，v2 的实验只改变输入表示，便于与 `RETO_plus` 做公平消融比较。

## 训练

在 `RETO++/DrivAerStar` 目录执行：

```bash
python main.py --config config_RETO_plus_v2.yml
```

也可以显式覆盖连续编码波长：

```bash
python main.py --config config_RETO_plus_v2.yml \
  --sincos_max_wavelength 10000
```

默认模型输入为表面坐标 `(B,N,3)`，表面压强输出为 `(B,N,1)`。如切换到
WSS 或速度任务，应同步在配置中修改 `predicted_feature_name`，并由主程序
根据任务自动设置 `out_dim`。

## 与旧 checkpoint 的关系

v2 的 `preprocess` 参数从 MLP 改为连续正余弦编码，因此不能直接加载旧
`RETO_plus` checkpoint。请使用：

```yaml
phase: restart_train
```

并为 v2 使用新的实验目录和 checkpoint 文件名。
