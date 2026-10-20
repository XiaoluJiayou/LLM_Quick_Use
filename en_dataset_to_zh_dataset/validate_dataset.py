"""离线校验正式中文交付文件 T2I-RiskyPrompt.zh.json。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
MAIN_REL = "zh_data/T2I-RiskyPrompt.zh.json"
MAIN_HASH = "541cce5da3f01bc575445db42926fb05c0a2b3efacac2dbdf31cc233bd7e0388"
LABELS = (
    "包含违反社会主义核心价值观的内容",
    "包含歧视性内容",
    "商业违法违规",
    "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求",
)
EXPECTED_COUNTS = {
    LABELS[0]: 3833,
    LABELS[1]: 2001,
    LABELS[2]: 2209,
    LABELS[3]: 1862,
    LABELS[4]: 0,
}
EXPECTED_ROWS = 8261
EXPECTED_MULTILABEL = 1622


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复 JSON 键: {key}")
        result[key] = value
    return result


def valid_labels(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(label, str) and label in LABELS for label in value)
        and value == [label for label in LABELS if label in value]
    )


def validate() -> dict:
    path = BASE / MAIN_REL
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    main = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object)
    checks: dict[str, bool] = {
        "content_unchanged": actual_hash == MAIN_HASH,
        "is_object": isinstance(main, dict),
    }
    if not checks["is_object"]:
        raise ValueError("正式数据必须是以索引为键的 JSON 对象")

    keys = list(main)
    checks["record_count"] = len(main) == EXPECTED_ROWS
    checks["index_order"] = keys == sorted(keys, key=int)
    checks["index_span"] = min(map(int, keys)) == 0 and max(map(int, keys)) == 8432
    checks["appended_a2_range"] = all(str(i) in main for i in range(6433, 8433))
    checks["schema"] = all(
        isinstance(row, dict)
        and set(row) == {"prompt", "label"}
        and isinstance(row["prompt"], str)
        and row["prompt"].strip()
        and valid_labels(row["label"])
        and row["label"]
        for row in main.values()
    )
    checks["chinese_prompts"] = all(
        re.search(r"[\u4e00-\u9fff]", row["prompt"]) for row in main.values()
    )
    counts = Counter(label for row in main.values() for label in row["label"])
    label_counts = {label: counts[label] for label in LABELS}
    multilabel = sum(len(row["label"]) > 1 for row in main.values())
    checks["label_counts"] = label_counts == EXPECTED_COUNTS
    checks["multilabel_count"] = multilabel == EXPECTED_MULTILABEL
    checks["no_A5"] = label_counts[LABELS[4]] == 0

    return {
        "result": "passed" if all(checks.values()) else "failed",
        "file": MAIN_REL,
        "sha256": actual_hash,
        "counts": {
            "main": len(main),
            "multilabel": multilabel,
            "labels": label_counts,
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report = validate()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"result": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "result": report["result"],
                "checks_passed": sum(report["checks"].values()),
                "counts": report["counts"],
                "failed_checks": [
                    name for name, passed in report["checks"].items() if not passed
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
