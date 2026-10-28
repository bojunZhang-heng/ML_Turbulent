# 从 Hugging Face 下载 DrivAerNet

下载页面：

```text
https://huggingface.co/datasets/MoElrefaie/DrivAerNet
```

下载脚本：

```text
/Users/zhangbojun/ML_Turbulent/preprocess/Download_hf_DrivAerNet.py
```

## 1. 设计说明

DrivAerNet 可能包含大量高分辨率几何和 CFD 文件，因此脚本默认不会立即下载整个仓库。

默认流程是：

```text
访问 Hugging Face Hub
→ 获取匹配文件清单
→ 计算需要下载的总大小
→ 检查本地剩余磁盘空间
→ 显示下载计划
→ 不下载数据
```

确认下载计划后，添加：

```text
--download
```

才会正式下载。

脚本使用 `huggingface_hub.snapshot_download()`，具备以下能力：

- 自动下载 Git LFS/Xet 大文件；
- 已完成且校验有效的文件会被复用；
- 网络中断后，重复相同命令可以继续下载；
- 支持 include/exclude glob，只下载需要的数据；
- 下载前显示文件数、总大小和剩余磁盘空间；
- 下载完成后保存 `download_manifest.json`。

## 2. 安装依赖

当前 `pgd-no` 环境已经安装 `huggingface_hub` 时，可以直接运行。

如果提示缺少依赖：

```bash
/opt/anaconda3/envs/pgd-no/bin/python -m pip install -U \
  huggingface_hub hf_xet
```

`hf_xet` 用于提高采用 Hugging Face Xet 存储的大文件下载效率。

## 3. 默认下载位置

默认目标目录：

```text
/Users/zhangbojun/Downloads/DrivAerNet
```

可以通过 `--output-dir` 修改。

## 4. 第一步：查看完整文件列表

由于仓库目录和文件名可能更新，建议先查询真实文件路径：

```bash
cd /Users/zhangbojun/ML_Turbulent

/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --list-files
```

如果需要把文件清单保存为 JSON：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --list-files \
  --file-list-json /Users/zhangbojun/Downloads/DrivAerNet_file_list.json
```

该操作只读取远端元数据，不下载数据集文件。

## 5. 第二步：执行 dry-run

不带 `--download` 运行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py
```

脚本会显示类似信息：

```text
Matched files
Matched total size
Files to download
Bytes to download
Already cached files
Destination
Free disk space
```

这一步不会下载数据集 payload。

如果计划下载量超过剩余磁盘空间，脚本会停止。

## 6. 第三步：正式下载完整仓库

确认 dry-run 的文件数和大小后执行：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --download
```

明确指定目标目录：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --output-dir /Users/zhangbojun/Downloads/DrivAerNet \
  --download
```

不建议在未执行 dry-run 的情况下直接下载完整数据集。

## 7. 只下载部分文件

`--include` 使用 Hugging Face glob pattern，可以重复多次。

只下载 VTK 文件的预检示例：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '*.vtk'
```

确认后：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '*.vtk' \
  --download
```

如果 VTK 位于多级目录中，可以根据 `--list-files` 的结果使用：

```bash
--include '**/*.vtk'
```

下载某个子目录：

```bash
--include '目标子目录/**'
```

下载多个类型：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '**/*.vtk' \
  --include '**/*.csv'
```

确认后在末尾添加 `--download`。

注意：glob 必须放在单引号内，避免 zsh 在本地提前展开。

### 7.1 只下载几个文件进行测试

例如，只从匹配到的 VTK 中选择按仓库路径排序后的前 3 个文件进行预检：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '**/*.vtk' \
  --max-files 3
```

输出会同时显示：

```text
筛选条件原本匹配多少个文件
最终选择了多少个文件
被选择文件的准确路径和大小
```

确认无误后正式下载这 3 个文件：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '**/*.vtk' \
  --max-files 3 \
  --download
```

只下载 1 个文件：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '**/*.vtk' \
  --max-files 1 \
  --download
```

`--max-files` 会在 `--include` 和 `--exclude` 筛选完成后生效，并按照远端仓库路径排序，因此相同 revision 下重复运行会选择相同文件。

如果仓库中的目标数据不是 `.vtk`，先使用 `--list-files` 查看扩展名，再替换 `--include` pattern。

## 8. 排除不需要的文件

例如排除压缩包：

```bash
--exclude '**/*.zip'
```

可以组合 include 和 exclude：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '目标目录/**' \
  --exclude '**/*.zip'
```

先检查输出计划，再添加 `--download`。

## 9. Hugging Face 登录与访问权限

公开数据集通常无需登录。

如果页面要求接受许可协议：

1. 在浏览器登录 Hugging Face；
2. 打开数据集页面；
3. 阅读并接受使用条件或申请访问；
4. 在终端登录。

推荐：

```bash
hf auth login
```

粘贴从以下页面创建的 read token：

```text
https://huggingface.co/settings/tokens
```

也可以临时设置环境变量：

```bash
export HF_TOKEN='hf_xxxxxxxxxxxxxxxxx'
```

然后运行下载命令。

不要把 token 直接写进 Python、README、YAML 或 Git 仓库。

脚本只会打印是否检测到 token，不会打印 token 内容。

## 10. 网络中断与断点续传

网络中断后，重新执行完全相同的命令：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --output-dir /Users/zhangbojun/Downloads/DrivAerNet \
  --download
```

Hub 会检查本地文件和缓存，复用已经完成的内容。

不要删除目标目录中的：

```text
.cache/huggingface/
```

否则部分断点和文件元数据可能丢失。

## 11. 下载速度与并发数

默认：

```text
--max-workers 4
```

在 macOS M1 和普通网络环境下，这是相对保守的设置。

网络稳定时可以尝试：

```bash
--max-workers 8
```

出现连接重置、限速或代理不稳定时降低为：

```bash
--max-workers 1
```

增加元数据连接超时：

```bash
--etag-timeout 120
```

增加大文件下载超时可以在运行前设置：

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=120
```

## 12. 强制重新下载

通常不需要强制重新下载。只有怀疑缓存损坏或远端同 revision 文件发生变化时才使用：

```bash
--force-download
```

例如：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --include '具体文件路径' \
  --force-download \
  --download
```

## 13. 下载固定版本

默认下载：

```text
revision: main
```

为了保证实验可复现，可以指定 commit hash 或 tag：

```bash
--revision COMMIT_HASH
```

完整示例：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --revision COMMIT_HASH \
  --include '**/*.vtk' \
  --download
```

下载完成后的 `download_manifest.json` 会记录 revision 和文件筛选规则。

## 14. 与 process_DrivAerNet.py 配合

下载完成后，先用 `find` 确认 VTK 所在目录：

```bash
find /Users/zhangbojun/Downloads/DrivAerNet \
  -type f -name '*.vtk' | head
```

然后将包含 VTK 的目录传给前处理脚本：

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  /Users/zhangbojun/ML_Turbulent/preprocess/process_DrivAerNet.py \
  --input-dir /Users/zhangbojun/Downloads/DrivAerNet/实际VTK目录 \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/DrivAerNet_data \
  --field pressure
```

如果下载的是 WSS 文件：

```bash
--field wss
```

## 15. 常见错误

### Repository not found

检查：

```text
repo-id 是否正确
是否已经登录
是否已经在网页接受许可
```

### Gated repository

必须先在网页申请或接受访问条件，再执行：

```bash
hf auth login
```

### No files matched

说明 `--include` 路径与仓库实际结构不一致。先运行：

```bash
python preprocess/Download_hf_DrivAerNet.py --list-files
```

然后复制准确路径设计 glob。

例如 `--include '**/*.vtk'` 返回 0 个文件，只表示 Hugging Face 仓库中没有以该路径形式保存 VTK。数据也可能被保存为：

```text
.zip
.tar
.tar.gz
.parquet
其他分片格式
```

dry-run 中出现 `Download complete` 或 `Reconstruction complete` 是 Hugging Face 后端对元数据解析阶段的进度提示。只要最终显示 `0.00B` 和 no files matched，就没有下载数据集内容。

### Connection timed out

检查网络、代理和 Hugging Face 可访问性。恢复网络后重复相同命令即可继续。

### No space left on device

不要直接下载完整数据集。使用 `--include` 只选择训练需要的文件，并把 `--output-dir` 指向容量足够的磁盘。

## 16. 查看所有参数

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  preprocess/Download_hf_DrivAerNet.py \
  --help
```
