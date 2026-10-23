"""Explicit, import-safe command lines for optional download backends."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def run_huggingface(
    default_repo: str,
    *,
    repo_type: str = "model",
    filename: str | None = None,
    default_local_dir: str | Path | None = None,
    default_endpoint: str | None = None,
    argv=None,
):
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载模型或数据集；导入不会下载。")
    parser.add_argument("--repo-id", default=default_repo)
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default=repo_type)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(default_local_dir) if default_local_dir is not None else None,
        help="本地下载目录；未指定时用入口脚本默认值，再没有则用 downloads/组织--仓库",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--endpoint",
        default=default_endpoint,
        help="可选镜像地址，例如 https://hf-mirror.com；未指定时用入口脚本默认值或官方 Hub",
    )
    parser.add_argument("--token-env", default="HF_TOKEN", help="存放令牌的环境变量名")
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不导入下载依赖或联网")
    if filename is not None:
        parser.add_argument("--filename", default=filename)
    args = parser.parse_args(argv)
    local_dir = args.local_dir or Path("downloads") / args.repo_id.replace("/", "--")
    kwargs = dict(repo_id=args.repo_id, repo_type=args.repo_type, local_dir=str(local_dir), revision=args.revision)
    if args.endpoint:
        kwargs["endpoint"] = str(args.endpoint).rstrip("/")
    if filename is not None:
        kwargs["filename"] = args.filename
    if args.dry_run:
        print(json.dumps(kwargs, ensure_ascii=False, indent=2))
        return kwargs
    try:
        import huggingface_hub
    except ImportError as exc:
        raise SystemExit("缺少 huggingface_hub，请安装项目的 download 依赖组。") from exc

    # 大文件走 Xet 时在国内更容易卡住；关闭后改用常规 HTTPS 续传。
    # 若需启用 Xet，可事先设置环境变量 HF_HUB_DISABLE_XET=0。
    if os.environ.get("HF_HUB_DISABLE_XET", "1") != "0":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    endpoint = kwargs.get("endpoint")
    if endpoint:
        # 与 huggingface_hub 常量对齐；仅传 endpoint= 时部分版本仍会读环境变量
        os.environ["HF_ENDPOINT"] = endpoint

    token = os.environ.get(args.token_env, "").strip()
    if token:
        kwargs["token"] = token
    downloader = huggingface_hub.hf_hub_download if filename is not None else huggingface_hub.snapshot_download
    try:
        result = downloader(**kwargs)
    except Exception as exc:
        message = str(exc)
        if endpoint and ("does not seem to be on huggingface.co" in message or "LocalEntryNotFoundError" in type(exc).__name__):
            raise SystemExit(
                "当前镜像无法提供完整文件元数据（常见于 hf-mirror 把 resolve 308 回官方站）。\n"
                "请去掉 --endpoint，改走官方源重试；已下载部分一般会断点续传：\n"
                f"  python model_dataset_download/dataset_download_huggingface.py --local-dir {local_dir}\n"
                f"原始错误：{exc}"
            ) from exc
        raise
    print(f"下载完成：{result}")
    return result


def run_modelscope(default_repo: str, argv=None):
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型；导入不会下载。")
    parser.add_argument("--repo-id", default=default_repo)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {"model_id": args.repo_id, "local_dir": str(args.local_dir or Path("downloads") / args.repo_id.replace("/", "--")), "revision": args.revision}
    if args.dry_run:
        print(json.dumps(kwargs, ensure_ascii=False, indent=2))
        return kwargs
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("缺少 modelscope，请按下载说明安装该可选依赖。") from exc
    result = snapshot_download(**kwargs)
    print(f"下载完成：{result}")
    return result
