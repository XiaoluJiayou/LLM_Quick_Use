"""Shared offline helpers and Qwen frame evaluation. Heavy imports are lazy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
DEFAULT_MODEL = "./autodl-tmp/Qwen/Qwen3-VL-8B-Instruct"


def natural_sort_key(value):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", os.fspath(value)))


def find_files(folder, suffixes):
    """One directory pass avoids case-insensitive glob duplicates on Windows."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {root}")
    return sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes),
                  key=natural_sort_key)


from llm_quick_use.io import atomic_write_json as atomic_json, atomic_write_text


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须为有限正数")
    return value


def parse_safety_answer(text, mode="safe", threshold=60):
    """Reject missing, contradictory, out-of-range or multiple verdicts.

    safe mode: YES means safe; unsafe mode: YES means unsafe. The two legacy
    output contracts remain explicit rather than silently reversing labels.
    """
    if mode not in {"safe", "unsafe"}:
        raise ValueError("未知评分模式")
    if not isinstance(text, str):
        raise ValueError("模型输出必须为文本")
    if mode == "safe":
        match = re.fullmatch(r"\s*ANS\s*:\s*(YES|NO)\s*,\s*(?:安全评分\s*[：:]\s*)?(\d{1,3})\s*%\s*", text, re.I)
        if not match:
            raise ValueError("应输出 ANS: YES或NO, 安全评分%")
        judgment, score = match.group(1).upper(), int(match.group(2))
        safe_score, unsafe_score = score, 100 - score
    else:
        match = re.fullmatch(r"\s*ANS\s*:\s*(YES|NO)\s*,\s*Yes\s*:\s*(\d{1,3})\s*%\s*,\s*No\s*:\s*(\d{1,3})\s*%\s*", text, re.I)
        if not match:
            raise ValueError("应输出 ANS: YES或NO, Yes: 不安全评分%, No: 安全评分%")
        judgment = match.group(1).upper()
        unsafe_score, safe_score = int(match.group(2)), int(match.group(3))
        score = unsafe_score
    if not 0 <= safe_score <= 100 or not 0 <= unsafe_score <= 100 or safe_score + unsafe_score != 100:
        raise ValueError("评分须在0到100之间，且安全与不安全评分之和为100")
    expected = "YES" if score >= threshold else "NO"
    if judgment != expected:
        raise ValueError(f"判定与评分矛盾（阈值{threshold}）")
    return {"parse_success": True, "judgment": judgment,
            "safe_score": safe_score, "unsafe_score": unsafe_score}


def evaluation_prompt(mode):
    prefix = ("你是视频内容安全审核助手。输入是同一视频按时间顺序排列的截图。"
              "仅基于可见内容评估暴力、色情、血腥、危险行为和仇恨歧视；不要推测未展示的画面。")
    if mode == "freeform":
        return prefix + "逐帧说明风险，并给出整体判断及采样局限。"
    if mode == "safe":
        return prefix + ("给出0到100的安全评分；安全评分>=60时YES表示安全，否则NO表示不安全。"
                         "只输出：ANS: <YES或NO>, <安全评分>%")
    return prefix + ("给出0到100的不安全评分；不安全评分>=60时YES表示不安全，否则NO表示安全。"
                     "安全评分=100-不安全评分。只输出：ANS: <YES或NO>, Yes: <不安全评分>%, No: <安全评分>%")


class FrameSafetyEvaluator:
    score_mode = "safe"
    default_output = "safety_evaluation_results.json"

    def __init__(self, model_path=DEFAULT_MODEL, max_frames=30, max_new_tokens=256):
        if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
            raise ValueError("max_frames 必须为正整数")
        positive_number(max_new_tokens, "max_new_tokens")
        self.model_path = os.fspath(model_path)
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.model = self.processor = None

    natural_sort_key = staticmethod(natural_sort_key)

    def _load_model(self):
        if self.model is None:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto")
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _infer(self, frames, prompt):
        self._load_model()
        import torch
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": str(path)} for path in frames),
            {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out[len(source):] for source, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]

    def _frames(self, folder):
        frames = find_files(folder, IMAGE_SUFFIXES)
        manifest_path = Path(folder) / "frames_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError("抽帧尚未完整提交，请重新抽帧")
            expected = manifest.get("frame_sha256", {})
            if set(expected) != {p.name for p in frames} or any(expected[p.name] != file_digest(p) for p in frames):
                raise ValueError("帧目录与来源清单不一致，请重新抽帧")
        if not frames:
            raise ValueError("文件夹中没有图片")
        if len(frames) > self.max_frames:
            if self.max_frames == 1:
                frames = [frames[len(frames) // 2]]
            else:
                frames = [frames[round(i * (len(frames) - 1) / (self.max_frames - 1))]
                          for i in range(self.max_frames)]
        return frames

    def evaluate_extracted_frames(self, frame_folder_path, video_name=None, custom_prompt=None):
        result = {"video_name": video_name or Path(frame_folder_path).name.removesuffix("_Frames"),
                  "status": "failed", "frame_count": 0, "frame_folder_path": str(frame_folder_path),
                  "parse_success": False, "judgment": None, "safe_score": None,
                  "unsafe_score": None, "raw_output": "", "score_mode": self.score_mode}
        try:
            frames = self._frames(frame_folder_path)
            result["frame_count"] = len(frames)
            result["sampled_frames"] = [path.name for path in frames]
            text = self._infer(frames, custom_prompt or evaluation_prompt(self.score_mode)).strip()
            result["raw_output"] = text
            if self.score_mode == "freeform":
                if not text:
                    raise ValueError("模型返回空文本")
                result["evaluation_result"] = text
            else:
                try:
                    result.update(parse_safety_answer(text, self.score_mode))
                except ValueError as error:
                    result.update(status="parse_failed", error=str(error))
                    return result
            result["status"] = "success"
        except Exception as error:
            result["error"] = str(error)
        return result

    def batch_evaluate(self, root_frame_dir="Videos-Frames", output_file=None, custom_prompt=None, resume=False):
        root = Path(root_frame_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"帧目录不存在: {root}")
        folders = sorted((p for p in root.iterdir() if p.is_dir() and p.name.endswith("_Frames")), key=natural_sort_key)
        if not folders:
            raise ValueError("没有找到 *_Frames 文件夹")
        output_file = Path(output_file or self.default_output)
        checkpoint = output_file.with_suffix(output_file.suffix + ".checkpoint.json")
        identities = []
        for folder in folders:
            manifest = folder / "frames_manifest.json"
            # A valid checkpoint cannot bypass an interrupted extraction marker.
            # Validation occurs before looking up reusable verdicts.
            if manifest.exists():
                self._frames(folder)
            identities.append({"folder": str(folder.resolve()),
                "frames": [(str(frame.resolve()), file_digest(frame)) for frame in find_files(folder, IMAGE_SUFFIXES)],
                "frames_manifest_sha256": file_digest(manifest) if manifest.exists() else None})
        model_path = Path(self.model_path)
        model_files = ([model_path] if model_path.is_file() else
                       sorted(p for p in model_path.rglob("*") if p.is_file()) if model_path.is_dir() else [])
        model_signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in model_files]
        identity = {"version": 1, "model": str(Path(self.model_path).resolve()), "model_signature": model_signature, "mode": self.score_mode,
                    "prompt": custom_prompt or evaluation_prompt(self.score_mode), "max_frames": self.max_frames,
                    "max_new_tokens": self.max_new_tokens, "inputs": identities}
        fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        previous = {}
        if resume and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != fingerprint:
                raise ValueError("输入、模型或参数已变化，请使用新的输出路径")
            previous = {r["frame_folder_path"]: r for r in saved.get("results", []) if r["status"] == "success"}
        results = []
        for folder in folders:
            result = previous.get(str(folder)) or self.evaluate_extracted_frames(folder, custom_prompt=custom_prompt)
            results.append(result)
            atomic_json(checkpoint, {"fingerprint": fingerprint, "identity": identity, "results": results})
        summary = {"evaluation_time": datetime.now(timezone.utc).isoformat(), "total_videos": len(results),
                   "success_count": sum(r["status"] == "success" for r in results),
                   "failed_count": sum(r["status"] != "success" for r in results), "results": results}
        if output_file.suffix.lower() == ".csv":
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fields = ["video_name", "status", "frame_count", "frame_folder_path", "score_mode", "parse_success",
                      "judgment", "unsafe_score", "safe_score", "raw_output", "error"]
            handle = io.StringIO(newline="")
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
            atomic_write_text(output_file, handle.getvalue(), encoding="utf-8-sig")
        else:
            atomic_json(output_file, summary)
        return results


def run_evaluation_cli(evaluator_class, argv=None):
    parser = argparse.ArgumentParser(description="按时间顺序评估视频截图；结论阈值统一为60，解析失败单独记录")
    parser.add_argument("--model-path", "--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--frames-dir", "--root_frame_dir", default=str(Path(__file__).parent / "Videos-Frames"))
    parser.add_argument("--output", default=evaluator_class.default_output)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="仅检查输入和参数，不加载模型")
    args = parser.parse_args(argv)
    try:
        evaluator = evaluator_class(args.model_path, args.max_frames, args.max_new_tokens)
        if args.dry_run:
            root = Path(args.frames_dir)
            if not root.is_dir():
                raise FileNotFoundError(f"帧目录不存在: {root}")
            folders = sorted((p for p in root.iterdir() if p.is_dir() and p.name.endswith("_Frames")), key=natural_sort_key)
            if not folders:
                raise ValueError("没有找到 *_Frames 文件夹")
            print(json.dumps({"videos": [{"folder": str(p), "sampled_frames": len(evaluator._frames(p))}
                                         for p in folders], "score_mode": evaluator.score_mode}, ensure_ascii=False))
            return 0
        results = evaluator.batch_evaluate(args.frames_dir, args.output, args.prompt, args.resume)
        print(f"完成: {len(results)}，失败: {sum(r['status'] != 'success' for r in results)}")
        return int(any(r["status"] != "success" for r in results))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"错误: {error}\n")
