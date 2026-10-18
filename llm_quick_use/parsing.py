"""从模型文本提取完整 JSON，交由标准解码器处理嵌套和字符串。"""
from __future__ import annotations

import json


def extract_json(text: str, expected_type=None):
    if not isinstance(text, str):
        raise TypeError("模型输出必须是字符串")
    if expected_type not in (None, dict, list):
        raise ValueError("expected_type 只能是 dict、list 或 None")
    # 只检查完整的最外层容器。截断外层数组不能被其内部完整的小数组冒充。
    start, stack, quoted, escaped = None, [], False, False
    for position, char in enumerate(text):
        if not stack:
            if char in "[{":
                start, stack, quoted, escaped = position, [char], False, False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            expected_open = "[" if char == "]" else "{"
            if stack[-1] != expected_open:
                stack = []
                continue
            stack.pop()
            if not stack:
                try:
                    value = json.loads(text[start:position + 1])
                except ValueError:
                    continue
                if isinstance(value, expected_type or (dict, list)):
                    return value
    raise ValueError("模型输出中没有完整且类型匹配的 JSON 对象或数组")
