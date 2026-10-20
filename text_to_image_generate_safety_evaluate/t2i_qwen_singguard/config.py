"""文生图安全评测配置；数据、模型、输出根目录和参数可用环境变量覆盖。"""
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(name, default):
    value = os.environ.get(name, default)
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"环境变量 {name} 不能为空")
    return value


def _integer(name, default, minimum=1):
    try:
        value = int(_env(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数") from exc
    if value < minimum:
        raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")
    return value


def _model_dir(path, marker):
    """尽力识别平铺或 ModelScope 缓存布局；访问权限留到模型加载时验证。"""
    path = Path(path).expanduser()
    try:
        path = path.resolve()
    except PermissionError:
        path = path.absolute()
    try:
        if (path / marker).is_file():
            return path
        nested = path / "snapshots" / "master"
        if (nested / marker).is_file():
            return nested
    except PermissionError:
        # 离线测试、帮助和报告入口不需要读取模型，更不应要求访问 /root。
        pass
    return path


def _model_name(path):
    if path.name == "master" and path.parent.name == "snapshots":
        return path.parents[1].name
    return path.name


# 数据与模型
DATA_JSON = Path(_env("DATA_JSON", BASE_DIR / "data" / "T2I-RiskyPrompt.zh.json")).expanduser()
MODEL_ROOT = Path(_env("MODEL_ROOT", "/root/autodl-tmp/models")).expanduser()
T2I_MODEL_DIR = _model_dir(_env("T2I_MODEL_DIR", MODEL_ROOT / "Qwen-Image-2.1"), "model_index.json")
JUDGE_MODEL_DIR = _model_dir(_env("JUDGE_MODEL_DIR", MODEL_ROOT / "Sing-Guard-8b"), "config.json")
POLICY_FILE = Path(_env("POLICY_FILE", BASE_DIR / "policy_gbt_t2i.md")).expanduser()

# 输出按出图模型分目录，换成 Turbo 版时结果不会混在一起
T2I_MODEL_NAME = _env("T2I_MODEL_NAME", _model_name(T2I_MODEL_DIR))
if T2I_MODEL_NAME in (".", "..") or any(ch in T2I_MODEL_NAME for ch in ("/", "\\")):
    raise ValueError("T2I_MODEL_NAME 必须是单个目录名，不能包含路径分隔符")
OUTPUT_DIR = Path(_env("OUTPUT_DIR", BASE_DIR / "outputs" / T2I_MODEL_NAME)).expanduser()
IMAGE_DIR = OUTPUT_DIR / "images"
GENERATIONS_FILE = OUTPUT_DIR / "generations.jsonl"
JUDGMENTS_FILE = OUTPUT_DIR / "judgments.jsonl"
REPORT_FILE = OUTPUT_DIR / "report.csv"

# 出图参数
WIDTH = _integer("WIDTH", 1024)
HEIGHT = _integer("HEIGHT", 1024)
if WIDTH % 32 or HEIGHT % 32:
    raise ValueError("Qwen-Image-2.1 的 WIDTH 和 HEIGHT 必须是 32 的正整数倍")
NUM_INFERENCE_STEPS = _integer("NUM_INFERENCE_STEPS", 40)
BASE_SEED = _integer("BASE_SEED", 42, minimum=0)
# Qwen-Image-2.1 整个放进显存约需 38GB，显存小于该值（GB）时改为 CPU 卸载
OFFLOAD_BELOW_GB = float(_env("OFFLOAD_BELOW_GB", 40))
if not math.isfinite(OFFLOAD_BELOW_GB) or OFFLOAD_BELOW_GB < 0:
    raise ValueError("OFFLOAD_BELOW_GB 必须是大于等于 0 的有限数值")

# 判定参数。默认 fast-slow（先给结论，再写推理，以最终答案为准）；要只要结论时设 JUDGE_MODE=fast
JUDGE_MODE = _env("JUDGE_MODE", "fast-slow")
if JUDGE_MODE not in ("fast", "fast-slow"):
    raise ValueError(f"JUDGE_MODE 只能是 fast 或 fast-slow，当前是 {JUDGE_MODE}")
JUDGE_MAX_NEW_TOKENS = _integer("JUDGE_MAX_NEW_TOKENS", 128)
JUDGE_SLOW_MAX_NEW_TOKENS = _integer("JUDGE_SLOW_MAX_NEW_TOKENS", 1024)
