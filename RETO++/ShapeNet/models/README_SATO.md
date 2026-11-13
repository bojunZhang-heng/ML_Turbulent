# ShapeNet SATO（7 推 4）

## 功能

该版本参考 `RETO++/DrivAerML` 和 `RETO++/DrivAerStar` 中的 SATO，面向
MLCFD 联合任务：

- 输入：`[x, y, z, SDF, normal_x, normal_y, normal_z]`，形状 `(B, N, 7)`；
- 输出：`[velocity_x, velocity_y, velocity_z, pressure]`，形状 `(B, N, 4)`；
- 前三个通道生成 Morton、转置 Morton、Hilbert、转置 Hilbert 四种序列；
- 七个输入通道全部进入 `preprocess`，空间序列只决定局部注意力的点排列。

## 模型逻辑

每个 SATO block 包含两条路径：

1. `Physics_Attention` 把所有点聚合成 physics tokens，学习全局流场关联；
2. 点云按当前空间序列排列，在长度为 `patch_size` 的局部窗口中做 attention；
3. 局部结果反序列化回原节点顺序；
4. 全局分支和带可学习门控的局部分支相加，再经过 MLP 残差更新。

不同层循环使用四种序列。模型最终统一解码为四个物理量，不在模型内部区分
表面点和体场点；区分由 `surface_mask` 和训练 loss 完成。

## 与 DrivAerML 版本的关系

核心结构及超参数含义保持一致。ShapeNet 迁移版额外保证：

- 序列生成支持真正的 batch 维；
- 序列索引和输入位于同一设备；
- 点数正好被窗口大小整除时不会添加一整块无效 padding；
- `n_iter > 1` 时仅在全部 block 结束后统一解码，避免通道提前从 hidden 降到 4；
- 模型可直接 `model(input)`，也可在后处理时传入预先计算的 `order/inverse`。

## 训练

从 `RETO++/ShapeNet` 运行：

```bash
python main.py --config config_SATO.yml
```

远端训练前，将 YAML 中的 `data.data_path` 和 `vtk_data_path` 改为服务器路径。
数据目录必须包含编号 PKL 和 `normalization_scalars.pkl`，联合任务还必须包含
`input`、`output`、`surface_mask`，其中通道数分别为 7 和 4。

## 关键超参数

- `slice_num`：全局 Physics-Attention 的 token 数量；
- `patch_size`：每个局部 attention 窗口的点数；
- `shift`：交错序列组数；
- `n_iter`：整组 SATO blocks 的重复次数；
- `pressure_weight`：联合 loss 中表面压力项的权重。

`patch_size` 增大会增强局部交互，但局部 attention 显存与计算量近似按
`patch_size²` 增长。大点云建议先保持 `patch_size: 20`、`shift: 2`。
