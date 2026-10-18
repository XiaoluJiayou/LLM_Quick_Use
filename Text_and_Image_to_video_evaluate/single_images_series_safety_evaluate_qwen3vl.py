"""Evaluate one frame folder; importing this module never loads a model."""
import argparse
import json
from _video_workflow import FrameSafetyEvaluator, DEFAULT_MODEL


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估一个视频帧目录")
    parser.add_argument("frames_dir", help="同一个视频的截图目录")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        evaluator = FrameSafetyEvaluator(args.model_path, args.max_frames)
        if args.dry_run:
            print(json.dumps([str(p) for p in evaluator._frames(args.frames_dir)], ensure_ascii=False))
            return 0
        result = evaluator.evaluate_extracted_frames(args.frames_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result["status"] != "success")
    except (ValueError, OSError) as error:
        parser.exit(1, f"错误: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
