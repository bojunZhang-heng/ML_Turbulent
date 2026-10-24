# DrivAerNet VTK 数据预处理

对应脚本：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/process_DrivAerNet.py
```

本文档按照要求使用文件名 `DEADME_DrivAerNet.md`。

## 1. 脚本用途

该脚本读取 DrivAerNet 汽车表面 legacy VTK 文件，将每个样本转换为一个已经完成归一化的 PKL 文件。

当前测试数据目录：

```text
/Users/zhangbojun/Desktop/E_S_WW_WM_small
```

该目录中的测试文件包含点数据压力字段：

```text
p
```

脚本会自动将 `p` 识别为 `pressure`。

脚本同时支持：

```text
--field pressure
--field wss
```

但当前这 10 个测试 VTK 只包含压力，没有 WSS。因此处理当前数据时应使用：

```text
--field pressure
```

如果选择 `--field wss`，但输入 VTK 中没有 WSS，程序会停止并打印 point data、cell data 中实际存在的数组名。

## 2. 输出目录

默认输出目录为：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerNet_data
```

输出结构示例：

```text
DrivAerNet_data/
├── 1.pkl
├── 2.pkl
├── 3.pkl
├── ...
├── 10.pkl
└── normalization_scalars.pkl
```

`E_S_WW_WM_001.vtk` 会输出为 `1.pkl`，文件名末尾的数字用于生成样本编号。

原始 VTK 文件只会被读取，不会被修改或删除。

## 3. 单个样本 PKL 的格式

压力模式下，每个样本文件只包含三个键：

```python
{
    "coor": ...,       # (N, 3), float32
    "normals": ...,    # (N, 3), float32
    "pressure": ...,   # (N, 1), float32
}
```

WSS 模式下：

```python
{
    "coor": ...,       # (N, 3), float32
    "normals": ...,    # (N, 3), float32
    "wss": ...,        # (N, C), float32
}
```

其中 `C` 是 VTK 中 WSS 的分量数。矢量 WSS 通常为 3。

三个数组的第 `i` 行始终对应同一个 VTK 表面节点：

```text
coor[i] ↔ normals[i] ↔ pressure[i] 或 wss[i]
```

## 4. 归一化方式

### 4.1 空间坐标

脚本首先扫描所有输入 VTK，计算整个数据集的逐坐标轴最小值和最大值：

\[
x'=
\frac{x-x_{min}}
{x_{max}-x_{min}}.
\]

因此 `coor` 的三个分量均位于 `[0,1]` 附近。

这里使用的是全数据集统一的 min-max，而不是每辆汽车单独归一化。这样不同汽车之间的相对尺寸和位置基准不会因样本独立缩放而改变。

### 4.2 法向量

法向量不做 min-max 或 mean-std 归一化，只进行单位长度处理：

\[
\hat n_i=\frac{n_i}{\lVert n_i\rVert_2}.
\]

脚本通过 VTK 表面拓扑计算点法向量，并启用：

```text
SplittingOff
ConsistencyOn
AutoOrientNormalsOn
```

计算后会检查：

- 点数量没有改变；
- 点顺序没有改变；
- 不存在零法向量、NaN 或 Inf；
- 每个法向量的模长约等于 1。

### 4.3 压力或 WSS

物理量使用整个数据集所有节点的 mean-std：

\[
y'=\frac{y-\mu_y}{\sigma_y}.
\]

如果 WSS 是多分量矢量，则每个分量独立计算均值和标准差。

## 5. normalization_scalars.pkl

压力模式下主要包含：

```text
coords_min
coords_max
coords_range
pressure_mean
pressure_std
field_name
field_components
physical_point_count
samples
normalization
```

WSS 模式下相应字段为：

```text
wss_mean
wss_std
```

数组形状：

```text
coords_min:    (1, 3)
coords_max:    (1, 3)
coords_range:  (1, 3)
pressure_mean: (1, 1)
pressure_std:  (1, 1)
```

WSS 的 mean/std 形状为 `(1,C)`。

压力预测结果反归一化：

```python
pressure_physical = pressure_normalized * pressure_std + pressure_mean
```

坐标反归一化：

```python
coords_physical = coor * coords_range + coords_min
```

## 6. macOS M1 运行方法

进入工作目录：

```bash
cd /Users/zhangbojun/ML_Turbulent
```

使用 `pgd-no` 环境处理默认测试目录：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerNet.py \
  --field pressure
```

完整指定输入、输出目录：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerNet.py \
  --input-dir /Users/zhangbojun/Desktop/E_S_WW_WM_small \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerNet_data \
  --field pressure
```

如果输出目录已经存在 PKL 文件，脚本默认停止，防止误覆盖。确认需要重新生成时添加：

```bash
--overwrite
```

完整覆盖命令：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/process_DrivAerNet.py \
  --input-dir /Users/zhangbojun/Desktop/E_S_WW_WM_small \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerNet_data \
  --field pressure \
  --overwrite
```

只处理某一类文件可以使用 glob：

```bash
--pattern 'E_S_WW_WM_*.vtk'
```

注意在 zsh 中给 glob 加单引号，避免 shell 提前展开。

## 7. 处理流程

脚本采用两遍处理，避免把全部大规模点云同时放入内存。

第一遍：

```text
读取所有 VTK
→ 统计全局坐标 min/max
→ 统计 pressure 或 WSS 的全局 mean/std
```

第二遍：

```text
重新读取每个 VTK
→ 计算单位点法向量
→ 使用第一遍统计量归一化
→ 验证节点对应关系和数值范围
→ 写入最终 PKL
```

因此每次执行都从原始 VTK 开始，不会读取旧的归一化样本继续归一化，也不会发生重复归一化。

## 8. 检查生成的 PKL

```python
import pickle
import numpy as np

sample_path = (
    "/Users/zhangbojun/ML_Turbulent/"
    "preprocess/DrivAerNet_data/1.pkl"
)

with open(sample_path, "rb") as file:
    sample = pickle.load(file)

print(sample.keys())
print("coor:", sample["coor"].shape)
print("normals:", sample["normals"].shape)
print("pressure:", sample["pressure"].shape)
print("coordinate range:", sample["coor"].min(), sample["coor"].max())
print(
    "normal length range:",
    np.linalg.norm(sample["normals"], axis=1).min(),
    np.linalg.norm(sample["normals"], axis=1).max(),
)
```

## 9. 关于全数据统计和数据泄漏

当前实现按照要求，使用输入目录中的整个数据集计算 normalization scalars。

如果以后进行严格的公开基准对比，更规范的做法是：

```text
只使用训练集计算 min/max 和 mean/std
验证集、测试集复用训练集统计量
```

否则测试集数值分布会轻微参与归一化参数估计。当前脚本适合你明确要求的“根据整个数据生成 normalization_scalars.pkl”的流程。

## 10. 根据表面压力计算阻力系数还需要什么

仅有当前精简 PKL 中的坐标、点法向量和压力，还不足以可靠计算阻力系数。至少还需要以下信息。

### 10.1 表面网格拓扑和面积

压力力是表面积分：

\[
\boldsymbol F_p
=-
\int_S p\boldsymbol n\,\mathrm dS.
\]

必须知道每个三角形或多边形由哪些节点组成，才能计算：

- 每个表面单元面积；
- 单元法向量；
- 点压力如何插值到单元积分点。

当前样本 PKL 按要求只保存三个数组，没有保存 VTK cell connectivity。因此计算阻力时需要保留并读取原始 VTK，或者另外保存：

```text
faces / cells / connectivity
```

### 10.2 未归一化压力及压力定义

神经网络输出必须先通过 `pressure_mean` 和 `pressure_std` 反归一化。

还需要确认 VTK 中压力的物理含义和单位：

```text
Pa 绝对压力
Pa 表压/相对压力
压力系数 Cp
其他无量纲压力
```

如果保存的是压力系数 `Cp`，积分公式和尺度处理与直接使用 Pa 不同。

### 10.3 来流参数

阻力系数定义为：

\[
C_D=
\frac{F_D}
{\frac12\rho_\infty U_\infty^2 A_{ref}}.
\]

因此需要：

```text
rho_inf：来流密度，kg/m^3
U_inf：来流速度，m/s
A_ref：参考迎风面积，m^2
```

不同数据集可能使用固定的 `A_ref`，也可能按每辆汽车的投影迎风面积计算。训练和评价必须使用与原始 CFD 数据一致的定义。

### 10.4 阻力方向和坐标系

需要明确：

- 来流沿 X、Y 还是 Z；
- 阻力正方向；
- VTK 坐标单位是 m、mm 还是其他单位；
- 法向量是否朝向流体域外侧/车体外侧。

压力阻力为：

\[
F_{D,p}
=
\boldsymbol F_p\cdot\boldsymbol e_D.
\]

法向量方向错误会直接导致力的符号错误。

### 10.5 计算总阻力时还需要 WSS

只积分压力得到的是压力阻力：

\[
\boldsymbol F_p=-\int_Sp\boldsymbol n\,\mathrm dS.
\]

汽车总阻力还包括黏性摩擦阻力。若 VTK 中的 WSS 是流体对壁面的切向应力矢量，则：

\[
\boldsymbol F_\tau=\int_S\boldsymbol\tau_w\,\mathrm dS.
\]

总阻力为：

\[
F_D=
(\boldsymbol F_p+\boldsymbol F_\tau)
\cdot\boldsymbol e_D.
\]

因此：

```text
只有 pressure → 压力阻力系数 Cd_pressure
pressure + WSS → 可计算更完整的总阻力系数 Cd_total
```

还需要确认 WSS 的方向约定：它可能表示“流体作用于壁面”的力，也可能是“壁面作用于流体”的反力，两者符号相反。

### 10.6 表面积分范围

需要明确原始 VTK 是否只包含车体表面，还是还包含：

- 地面；
- 风洞边界；
- 入口/出口；
- 车轮内部或封闭面；
- 对称面。

计算汽车阻力时只能积分属于车辆的壁面区域。若一个 VTK 同时包含其他边界，需要额外的 boundary ID、patch name 或 cell mask。

### 10.7 推荐为阻力后处理保留的数据

推荐在原始 VTK 或单独的后处理文件中保留：

```text
原始物理坐标
表面 cell connectivity
每个 cell 的面积和朝外法向量
车辆表面 boundary mask
未归一化 pressure 或 Cp
未归一化 WSS 矢量（若计算总阻力）
rho_inf
U_inf
A_ref
阻力方向 e_D
坐标和物理量单位
```

当前脚本不会删除原始 VTK，因此后续可以将网络预测压力反归一化后，按相同节点顺序写回原始 VTK，再利用原始拓扑完成面积分。
