"""Import-safe download example; run with --help for configuration."""
import os

try:
    from ._download_cli import run_huggingface
except ImportError:  # Direct script execution.
    from _download_cli import run_huggingface

# Hugging Face 访问令牌（本地下载可填；不要把含真实 token 的改动提交到 git）
# 也可在 PowerShell：$env:HF_TOKEN = "你的token"
HF_TOKEN = ""


def main(argv=None):
    if HF_TOKEN.strip():
        os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    # 默认走官方 Hub。hf-mirror 当前常把文件 308 回官方且缺元数据，会导致下载失败。
    # 若以后镜像恢复，可加：default_endpoint="https://hf-mirror.com"
    return run_huggingface(
        "zhangrjjj/Falcon",
        repo_type="dataset",
        default_local_dir=r"F:\百万级多模态大模型安全评测数据集\自建数据集_59.3W\multimodal_dataset",
        argv=argv,
    )


if __name__ == "__main__":
    main()
