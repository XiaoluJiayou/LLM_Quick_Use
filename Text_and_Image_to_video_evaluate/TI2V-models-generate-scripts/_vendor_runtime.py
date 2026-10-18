"""Offline validation and provenance for optional vendor video backends."""
import hashlib
import json
import math
import os
from pathlib import Path

from llm_quick_use.io import atomic_write_json


def validate_finite_options(options):
    """Check all numeric options, including scales where zero is meaningful."""
    for name, value in options.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name}必须为有限数值")


def load_prompts(json_file):
    with open(json_file, encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("prompts JSON须为非空数组")
    indices = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"第{position}条数据须为对象")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in indices:
            raise ValueError(f"第{position}条index须为不重复的非负整数")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"第{position}条prompt须为非空文本")
        if "category" in record and not isinstance(record["category"], str):
            raise ValueError(f"第{position}条category须为文本")
        indices.add(index)
    return records


def validate_run(json_file, paths, numbers, start=None, end=None, categories=None,
                 frames=None, frame_multiple=None):
    for name, value in numbers.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}必须为有限正数")
        if name in {"height", "width", "num_frames", "frame_num", "num_inference_steps", "sample_steps", "context_parallel_size"} and not isinstance(value, int):
            raise ValueError(f"{name}必须为整数")
    for name, value in paths.items():
        if value is not None and not Path(value).exists():
            raise FileNotFoundError(f"{name}不存在: {value}")
        if value is not None and name in {"checkpoint_path", "distilled_lora_path", "spatial_upsampler_path"} and not Path(value).is_file():
            raise ValueError(f"{name}须为权重文件")
        if value is not None and (name.endswith("_dir") or name in {"gemma_root", "base_model_path", "transformer_path"}) and not Path(value).is_dir():
            raise ValueError(f"{name}须为模型目录")
    for name, value in (("start_index", start), ("end_index", end)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name}须为非负整数")
    if start is not None and end is not None and start >= end:
        raise ValueError("start_index必须小于end_index")
    if frame_multiple and frames is not None and (frames - 1) % frame_multiple:
        raise ValueError(f"num_frames须满足{frame_multiple}k+1")
    records = load_prompts(json_file)
    selected = [p for p in records if (start is None or p["index"] >= start)
                and (end is None or p["index"] < end)
                and (not categories or p.get("category") in categories)]
    if not selected:
        raise ValueError("没有匹配的prompt")
    return selected


def show_dry_run(backend, prompts):
    print(json.dumps({"backend": backend, "validated_records": len(prompts),
                      "indices": [p["index"] for p in prompts], "model_loaded": False}, ensure_ascii=False))


def _digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def model_identity(value):
    """Local stat signatures detect checkpoint replacements without reading TBs."""
    path = Path(value)
    if not path.exists():
        return str(value)
    items = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return [{"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in items]


class RunTracker:
    """Only reuse videos whose complete bytes and run configuration match."""
    def __init__(self, output_dir, config, prompts):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.output_dir / "generation_manifest.json"
        identity = {"version": 1, "config": config, "prompts": prompts}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        if self.manifest.exists():
            self.data = json.loads(self.manifest.read_text(encoding="utf-8"))
            if self.data.get("fingerprint") != fingerprint:
                raise ValueError("输出目录的模型、参数或prompts已变化，请使用新目录")
        else:
            if any(self.output_dir.glob("*.mp4")):
                raise ValueError("输出目录含无来源记录的视频，请使用新目录")
            self.data = {"fingerprint": fingerprint, "identity": identity, "completed": {}}
            atomic_write_json(self.manifest, self.data)

    def output_path(self, index):
        return self.output_dir / f"{index}.mp4"

    def staging_path(self, index):
        return self.output_dir / f".{index}.partial.mp4"

    def should_skip(self, index):
        path = self.output_path(index)
        record = self.data["completed"].get(str(index))
        if not path.exists():
            return False
        if not record or path.stat().st_size == 0 or _digest(path) != record.get("sha256"):
            raise ValueError(f"已存在视频缺少有效来源或内容损坏: {path}；请使用新输出目录")
        return True

    def finish(self, index):
        staging = self.staging_path(index)
        if not staging.is_file() or staging.stat().st_size == 0:
            raise RuntimeError("后端没有生成非空视频")
        checksum = _digest(staging)
        # Commit provenance first. An interruption before publication is retried;
        # an interruption after publication can already verify the final bytes.
        self.data["completed"][str(index)] = {"sha256": checksum}
        atomic_write_json(self.manifest, self.data)
        os.replace(staging, self.output_path(index))
