"""Dispatch to an actual video model; Qwen3-VL is a vision-language evaluator."""
import argparse
from pathlib import Path
import runpy
import sys

BACKENDS = {"wan": "wan2.2_video_generate.py", "ltx": "LTX-2.3_generate.py",
            "helios": "helios_generate.py", "longcat": "longCat_video_generate.py"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="选择已实现的视频生成后端；Qwen3-VL不提供文生视频能力",
        epilog="例: python batch_videos_generate_for_dataset.py --backend wan -- --json_file data.json --ckpt_dir models --dry-run")
    parser.add_argument("--backend", choices=sorted(BACKENDS))
    parser.add_argument("--check_env", action="store_true", help="仅列出支持的后端，不加载依赖")
    args, forwarded = parser.parse_known_args(argv)
    if args.check_env:
        print("可用脚本后端: " + ", ".join(sorted(BACKENDS)))
        print("实际生成需要各后端的官方仓库、权重与GPU；使用后端 --dry-run 检查数据和参数。")
        return 0
    if args.backend is None:
        parser.error("原Qwen3-VL视频生成接口不受支持；请用 --backend 选择 wan/ltx/helios/longcat")
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    target = Path(__file__).parent / "TI2V-models-generate-scripts" / BACKENDS[args.backend]
    old_args = sys.argv
    old_path = list(sys.path)
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *forwarded]
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old_args
        sys.path[:] = old_path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
