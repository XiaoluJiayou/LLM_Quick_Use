"""文生图安全评测的路径与参数配置，每一项都可以用同名环境变量覆盖。"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# print(BASE_DIR)


def _env(name, default):
    return os.environ.get(name, default)


def _model_dir(path, marker):
    """ModelScope 缓存会把权重放在 snapshots/master 下，平铺目录则直接放在根上。两种都认。"""
    path = Path(path)
    if (path / marker).exists():
        return path
    nested = path / "snapshots" / "master"
    if (nested / marker).exists():
        return nested
    return path


def _model_name(path):
    if path.name == "master" and path.parent.name == "snapshots":
        return path.parents[1].name
    return path.name


# 数据与模型
DATA_CSV = Path(_env("DATA_CSV", BASE_DIR / "data" / "jade_benchmark_zh.csv"))
MODEL_ROOT = Path(_env("MODEL_ROOT", "/root/autodl-tmp/models"))
T2I_MODEL_DIR = _model_dir(_env("T2I_MODEL_DIR", MODEL_ROOT / "Qwen-Image-2.1"), "model_index.json")
JUDGE_MODEL_DIR = _model_dir(_env("JUDGE_MODEL_DIR", MODEL_ROOT / "Sing-Guard-8b"), "config.json")
POLICY_FILE = Path(_env("POLICY_FILE", BASE_DIR / "policy_jade_t2i.md"))

# 输出按出图模型分目录，换成 Turbo 版时结果不会混在一起
T2I_MODEL_NAME = _env("T2I_MODEL_NAME", _model_name(T2I_MODEL_DIR))
OUTPUT_DIR = Path(_env("OUTPUT_DIR", BASE_DIR / "outputs" / T2I_MODEL_NAME))
IMAGE_DIR = OUTPUT_DIR / "images"
GENERATIONS_FILE = OUTPUT_DIR / "generations.jsonl"
JUDGMENTS_FILE = OUTPUT_DIR / "judgments.jsonl"
REPORT_FILE = OUTPUT_DIR / "report.csv"

# 出图参数
WIDTH = int(_env("WIDTH", 1024))
HEIGHT = int(_env("HEIGHT", 1024))
NUM_INFERENCE_STEPS = int(_env("NUM_INFERENCE_STEPS", 40))
BASE_SEED = int(_env("BASE_SEED", 42))
# Qwen-Image-2.1 整个放进显存约需 38GB，显存小于该值（GB）时改为 CPU 卸载
OFFLOAD_BELOW_GB = float(_env("OFFLOAD_BELOW_GB", 40))

# 判定参数。JUDGE_MODE 可选 fast（只给结论）或 fast-slow（先给结论，再写推理，以最终答案为准）
JUDGE_MODE = _env("JUDGE_MODE", "fast")
if JUDGE_MODE not in ("fast", "fast-slow"):
    raise ValueError(f"JUDGE_MODE 只能是 fast 或 fast-slow，当前是 {JUDGE_MODE}")
JUDGE_MAX_NEW_TOKENS = int(_env("JUDGE_MAX_NEW_TOKENS", 128))
JUDGE_SLOW_MAX_NEW_TOKENS = int(_env("JUDGE_SLOW_MAX_NEW_TOKENS", 1024))
