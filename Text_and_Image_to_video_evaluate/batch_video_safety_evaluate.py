"""Qwen3-VL frame evaluation; use --help or --dry-run without GPU dependencies."""
from _video_workflow import FrameSafetyEvaluator, run_evaluation_cli


class VideoSafetyEvaluator(FrameSafetyEvaluator):
    score_mode = 'safe'
    default_output = 'safety_evaluation_results.csv'


def main(argv=None):
    return run_evaluation_cli(VideoSafetyEvaluator, argv)


if __name__ == "__main__":
    raise SystemExit(main())
