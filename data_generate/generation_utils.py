"""Validated JSON handling for the dataset generation notebooks."""
from __future__ import annotations

from llm_quick_use.io import atomic_write_json, load_json_array
from llm_quick_use.parsing import extract_json


def parser_simple_json(text):
    questions = extract_json(text, expected_type=list)
    if not all(isinstance(item, str) and item.strip() for item in questions):
        raise ValueError("模型必须返回非空问题字符串组成的 JSON 数组")
    return list(dict.fromkeys(item.strip() for item in questions))


def read_existing_data(file_path):
    data = load_json_array(file_path)
    if not all(isinstance(item, str) and item.strip() for item in data):
        raise ValueError("已有问题文件必须是字符串数组，已保留原文件")
    return data


def update_json_file(file_path, new_data):
    if not isinstance(new_data, list) or not all(isinstance(item, str) and item.strip() for item in new_data):
        raise ValueError("新问题必须是非空字符串组成的数组")
    existing = read_existing_data(file_path)
    combined = list(dict.fromkeys([*existing, *(item.strip() for item in new_data)]))
    atomic_write_json(file_path, combined)
    return len(combined) - len(existing)
