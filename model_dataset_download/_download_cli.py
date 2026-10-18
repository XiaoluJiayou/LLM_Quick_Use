"""Explicit, import-safe command lines for optional download backends."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def run_huggingface(default_repo: str, *, repo_type: str = "model", filename: str | None = None, argv=None):
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载模型或数据集；导入不会下载。")
    parser.add_argument("--repo-id", default=default_repo)
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default=repo_type)
    parser.add_argument("--local-dir", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", help="可选镜像地址；默认遵循 Hugging Face 配置")
    parser.add_argument("--token-env", default="HF_TOKEN", help="存放令牌的环境变量名")
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划，不导入下载依赖或联网")
    if filename is not None:
        parser.add_argument("--filename", default=filename)
    args = parser.parse_args(argv)
    local_dir = args.local_dir or Path("downloads") / args.repo_id.replace("/", "--")
    kwargs = dict(repo_id=args.repo_id, repo_type=args.repo_type, local_dir=str(local_dir), revision=args.revision)
    if args.endpoint:
        kwargs["endpoint"] = args.endpoint
    if filename is not None:
        kwargs["filename"] = args.filename
    if args.dry_run:
        print(json.dumps(kwargs, ensure_ascii=False, indent=2))
        return kwargs
    try:
        import huggingface_hub
    except ImportError as exc:
        raise SystemExit("缺少 huggingface_hub，请安装项目的 download 依赖组。") from exc
    token = os.environ.get(args.token_env, "").strip()
    if token:
        kwargs["token"] = token
    downloader = huggingface_hub.hf_hub_download if filename is not None else huggingface_hub.snapshot_download
    result = downloader(**kwargs)
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
