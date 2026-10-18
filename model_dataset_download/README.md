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
| `dataset_download_huggingface.py` | 数据集，支持 `--repo-type` |
| `model_download_huggingface_standard.py` | 模型快照 |
| `huggingface_standalone_file_download.py` | 单文件，支持 `--filename` |
| `model_download_modelscope.py` | ModelScope 模型快照，单独安装 `modelscope` |

Hugging Face 私有或受限仓库凭据从 `HF_TOKEN` 读取，也可以用 `--token-env` 指定已有环境变量名称；不通过命令行传递令牌，不打印令牌。未设置时沿用 SDK 的默认认证。`--endpoint` 可显式指定所需镜像，`--revision` 固定版本。

ModelScope 的受限仓库使用其 SDK 的既有登录配置。不同 SDK 对 `MODELSCOPE_API_TOKEN` / `token` 参数的支持有版本差异；本入口不向旧 SDK 盲传新参数，认证方式请以安装版本的[官方命令文档](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)和[下载接口](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)为准。

`--dry-run` 不加载下载 SDK、不创建下载目录、不请求网络。实际访问权限、仓库文件名、网络及磁盘空间仍需在下载环境验证。
