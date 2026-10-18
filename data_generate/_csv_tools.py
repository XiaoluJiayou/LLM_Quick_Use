"""Streaming CSV readers and atomic writers shared by data utilities."""
from __future__ import annotations

import csv
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def csv_header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header or any(not name.strip() for name in header) or len(header) != len(set(header)):
        raise ValueError(f"CSV 缺少表头或有重复列：{path}")
    return header


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV 第 {line} 行列数与表头不一致：{path}")
            yield row


def validate_output(inputs, output, overwrite=False):
    output = Path(output).resolve()
    if output in {Path(path).resolve() for path in inputs}:
        raise ValueError("输出路径不能与输入路径相同")
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在：{output}；显式使用 --overwrite 覆盖")
    return output


@contextmanager
def atomic_csv_writer(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            yield writer
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
