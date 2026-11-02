# DrivAerStar下载说明（Windows）

下载脚本：

```text
preprocess/download_DrivAerStar.py
```

数据源：

```text
https://huggingface.co/datasets/drivaerstar000/drivaerstar
```

该数据集需要手动申请访问权限。先在网页提交申请并等待批准，然后在
Windows PowerShell 中安装依赖并登录：

```powershell
py -m pip install -U huggingface_hub requests
hf auth login
hf auth whoami
```

不要把 Hugging Face Token 写入 Python 文件或提交到 Git。

## 数据结构

Hugging Face 当前没有保存 `case_E_batch_401.zip` 这种批次ZIP。每个车型由
五个文件组成，例如：

```text
case_E/00406/case/E_00406.case
case_E/00406/case/E_00406.geo
case_E/00406/case/E_00406.Pressure
case_E/00406/case/E_00406.Velocity
case_E/00406/case/E_00406.WallShearStress
```

401–800是车型编号范围，但编号不连续。脚本会读取仓库文件清单，只选择实际
存在的车型。当前仓库清单在这个范围内包含250个完整车型，首个编号是406。

## 小规模测试

先只预览前两个可用车型，不下载：

```powershell
py preprocess\download_DrivAerStar.py `
  --output-dir E:\DrivAerStar `
  --start-case 401 `
  --end-case 800 `
  --max-cases 2 `
  --file-types case
```

确认清单正确后，仅下载两个很小的 `.case` 描述文件：

```powershell
py preprocess\download_DrivAerStar.py `
  --output-dir E:\DrivAerStar `
  --start-case 401 `
  --end-case 800 `
  --max-cases 2 `
  --file-types case `
  --download
```

再用一个车型测试完整的五类文件：

```powershell
py preprocess\download_DrivAerStar.py `
  --output-dir E:\DrivAerStar `
  --start-case 401 `
  --end-case 800 `
  --max-cases 1 `
  --download
```

## 下载401–800范围内的全部可用车型

小规模测试通过后执行：

```powershell
py preprocess\download_DrivAerStar.py `
  --output-dir E:\DrivAerStar `
  --start-case 401 `
  --end-case 800 `
  --download
```

已有完整文件会被跳过，因此下载中断后可以重新执行相同命令继续。不要使用
`--force-download`，除非确定需要重新下载已有文件。

## 可选：为每个车型生成ZIP

Hugging Face不提供批次ZIP。如果确实需要ZIP，可以由脚本为每个车型生成一个
本地压缩包：

```powershell
py preprocess\download_DrivAerStar.py `
  --output-dir E:\DrivAerStar `
  --start-case 401 `
  --end-case 800 `
  --download `
  --zip-cases
```

ZIP生成在：

```text
E:\DrivAerStar\zip\case_E_00406.zip
```

## 只下载需要的物理量

例如只下载案例描述、几何和压力：

```powershell
py preprocess\download_DrivAerStar.py `
  --output-dir E:\DrivAerStar `
  --start-case 401 `
  --end-case 800 `
  --file-types case geo pressure `
  --download
```

## 常见问题

出现401或403：

- 确认网页访问申请已经批准；
- 确认 `hf auth login` 使用的是获批账号；
- 使用 `hf auth whoami` 检查当前账号。

连接超时或SSL错误：

- 说明当前网络无法连接 `huggingface.co:443`，并不代表车型文件不存在；
- 在能够访问Hugging Face的Windows网络环境重试；
- gated数据不建议使用第三方镜像，因为文件下载仍可能跳转回官方域名。

磁盘空间不足：

- 先用 `--max-cases 1` 完整测试；
- 下载前确认目标盘剩余空间；
- 不需要速度或WSS时，使用 `--file-types` 排除相应文件。
