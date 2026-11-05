# RETO（DrivAerML）

`RETO.py` 是从仓库 `RETO/modules_RT/model/model_transolver.py` 迁移的独立
实现，不再依赖外部 `RETO/modules_RT` 包。它适用于 DrivAerML 的三维坐标输入：

```text
输入:  (B, N, 3)  已归一化坐标
输出:  (B, N, 1)  表面压强（默认配置）
```

模型先用连续正余弦编码将坐标映射到 `n_hidden` 维，再通过 MLP 和可学习
placeholder 得到节点特征。每个 RETO block 使用独立的 Q/K/V 投影；Q、K 采用
坐标生成的 RoPE，V 保持内容特征，然后执行全点 scaled dot-product attention，
最后用 MLP 残差更新并解码物理量。

与 `RETO_plus` 的区别是 RETO 使用全点注意力，显存和计算复杂度为
`O(N^2)`；大点云应优先使用较少层数/较小 hidden，或继续使用 `RETO_plus`。

## 训练

在本目录运行：

```bash
python main.py --config config_RETO.yml
```

主要参数位于 `config_RETO.yml`：`n_layers`、`n_hidden`、`n_head`、
`mlp_ratio`、`sincos_max_wavelength` 和 `rope_max_wavelength`。

旧的 `RETO_plus` checkpoint 与本模型结构不同，不能直接加载；请使用新的
实验目录，并设置 `phase: restart_train`。
