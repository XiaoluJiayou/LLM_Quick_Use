# 模型与数据集下载

导入这些示例不会下载。所有下载必须显式运行脚本，目录可通过 `--local-dir` 配置，默认写入当前目录的 `downloads/组织--仓库`。默认不强制使用镜像。

在项目根目录安装 Hugging Face 下载依赖：

```powershell
python -m pip install -e ".[download]"
python model_dataset_download/model_download_huggingface.py --help
python model_dataset_download/model_download_huggingface.py --repo-id openai-community/gpt2 --local-dir downloads/gpt2 --dry-run
```

确认计划后删除 `--dry-run` 才会联网下载。其他入口：

| 入口 | 用途 |
| --- | --- |
| `dataset_download_huggingface.py` | 数据集，默认 `zhangrjjj/Falcon`，可用 `--repo-id` / `--local-dir` |
| `model_download_huggingface_standard.py` | 模型快照 |
| `huggingface_standalone_file_download.py` | 单文件，支持 `--filename` |
| `model_download_modelscope.py` | ModelScope 模型快照，单独安装 `modelscope` |

Hugging Face 凭据优先读环境变量 `HF_TOKEN`（PowerShell：`$env:HF_TOKEN = "你的token"`）。`dataset_download_huggingface.py` 顶部也可填 `HF_TOKEN = "..."`，运行时会写入环境变量；不要把真实 token 提交进 git。也可用 `--token-env` 指定其他环境变量名；不通过命令行传令牌，不打印令牌。未设置时沿用 SDK 默认认证。默认走官方 Hub，并自动设置 `HF_HUB_DISABLE_XET=1` 以降低大文件卡住概率。`--endpoint` 可指定镜像，但当前 `hf-mirror.com` 对部分仓库会 308 回官方且缺少 ETag/commit，容易直接失败，不建议作为默认；`--revision` 固定版本。

ModelScope 的受限仓库使用其 SDK 的既有登录配置。不同 SDK 对 `MODELSCOPE_API_TOKEN` / `token` 参数的支持有版本差异；本入口不向旧 SDK 盲传新参数，认证方式请以安装版本的[官方命令文档](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)和[下载接口](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)为准。

`--dry-run` 不加载下载 SDK、不创建下载目录、不请求网络。实际访问权限、仓库文件名、网络及磁盘空间仍需在下载环境验证。
