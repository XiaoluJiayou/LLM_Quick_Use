"""可恢复的结果文件读写，不依赖模型框架。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any


def stable_fingerprint(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path, strict=False) -> list[dict]:
    """逐行读取对象；默认跳过中断残行，strict=True 时明确报错。"""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL 每行必须是对象")
            except (UnicodeDecodeError, ValueError) as exc:
                message = f"{path} 第 {number} 行不是有效的 JSON 对象"
                if strict:
                    raise ValueError(message) from exc
                logging.warning(message)
                continue
            records.append(item)
    return records


def append_jsonl(path, item: dict) -> None:
    if not isinstance(item, dict):
        raise TypeError("JSONL 记录必须是字典")
    payload = (json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(payload)
        handle.flush()


def atomic_write_text(path, text: str, encoding="utf-8") -> None:
    """先在同目录完整写入再替换；发生异常时保留原文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding=encoding, newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_json_array(path) -> list:
    """不存在返回空列表；已有文件损坏时拒绝覆盖。"""
    path = Path(path)
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"已有结果文件不是有效 JSON，已保留原文件: {path}") from exc
    if not isinstance(value, list):
        raise ValueError(f"已有结果文件必须是 JSON 数组，已保留原文件: {path}")
    return value
