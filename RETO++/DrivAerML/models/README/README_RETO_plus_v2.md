# RETO_plus_v2

`RETO_plus_v2` 保留 `RETO_plus` 的 cluster attention、连续旋转位置编码
（RoPE）、LayerScale 和 activation checkpointing，只替换输入预处理编码。

## 输入编码

原始 `RETO_plus`：

```text
XYZ (B,N,3) -> MLP -> hidden features (B,N,n_hidden)
```

v2：

```text
XYZ (B,N,3) -> ContinuousSincosEmbed -> hidden features (B,N,n_hidden)
```

连续编码对三个坐标轴使用多尺度正弦和余弦基函数：

```text
sin(x * omega), cos(x * omega)
```

`n_hidden=256` 时，编码工具会自动处理3个坐标轴不能整除的情况，并通过
零填充得到精确的256维输出。模型使用数据集已经完成 min-max 归一化的坐标，
不会在模型内部重新归一化，也不会改变物理量的 mean-std 归一化。

## 其余模型结构

v2 保留 `RETO_plus.py` 中的独立 Q/K/V、内容驱动 cluster assignment、连续
三维 RoPE、cluster-to-node cross attention、LayerScale 和分段 activation
checkpointing。因此该版本只改变输入表示，适合与 `RETO_plus` 做消融比较。

## 训练命令

在 `RETO++/DrivAerML` 目录执行：

```bash
python main.py --config config_RETO_plus_v2.yml
```

默认训练表面压强，输入为 `(B,N,3)`，输出为 `(B,N,1)`。连续编码的最大波长
可以通过 YAML 中的 `position_encoding.sincos_max_wavelength` 修改。

## Checkpoint

由于 v2 的 `preprocess` 从 MLP 改成了连续正余弦编码，不能直接加载旧的
`RETO_plus` checkpoint。请使用新的实验目录并保持：

```yaml
phase: restart_train
```
