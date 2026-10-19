"""Validate the delivered T2I-RiskyPrompt files offline using the Python standard library."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE_HASH = "e566f40c5970525b532ec6289e98f098cc87936a3b9d290ada00ece26b165669"
LABELS = (
    "包含违反社会主义核心价值观的内容",
    "包含歧视性内容",
    "商业违法违规",
    "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求",
)
EXPECTED_HASHES = {
    "zh_data/T2I-RiskyPrompt.zh.json": "913337f026da3f054700634dadd72ddcdb0e7049df321caa94e5d5a4b364587e",
    "zh_data/T2I-RiskyPrompt.zh.excluded.json": "70cfb830b90f3aedcd67df1733a0f702db6f5e43b948203dbb56b9f8865453bb",
    "en_data/T2I-RiskyPrompt.filtered.json": "9b61593523223330df79d06d00a0581a7a5fd8c2196c3acba7e14215a262f01c",
}


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def valid_labels(value: object) -> bool:
    return (isinstance(value, list)
            and all(isinstance(label, str) and label in LABELS for label in value)
            and value == [label for label in LABELS if label in value])


def validate() -> dict:
    documents, artifacts, checks = {}, {}, {}
    for relative, expected_hash in EXPECTED_HASHES.items():
        raw = (BASE / relative).read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        documents[relative] = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object)
        artifacts[relative] = {"bytes": len(raw), "sha256": actual_hash}
        checks[f"content_unchanged:{relative}"] = actual_hash == expected_hash
    main, excluded, english = documents.values()
    if not all(isinstance(document, dict) for document in (main, excluded, english)):
        raise ValueError("Dataset files must be index-keyed JSON objects")
    checks["main_schema"] = all(isinstance(row, dict) and set(row) == {"prompt", "label"}
                                and isinstance(row["prompt"], str) and valid_labels(row["label"])
                                for row in main.values())
    checks["excluded_metadata_schema"] = all(
        isinstance(row, dict) and set(row) == {"label", "reason", "original_labels"}
        and valid_labels(row["label"]) and isinstance(row["reason"], str) and bool(row["reason"].strip())
        and isinstance(row["original_labels"], list)
        and all(isinstance(label, str) for label in row["original_labels"])
        for row in excluded.values())
    checks["english_schema"] = all(isinstance(row, dict) and isinstance(row.get("prompt"), str)
                                   and bool(row["prompt"].strip()) for row in english.values())
    if not all(checks[name] for name in ("main_schema", "excluded_metadata_schema", "english_schema")):
        raise ValueError("A delivered dataset has invalid record fields or labels")
    protected = {key for key, row in excluded.items() if row["label"]}
    checks["record_counts"] = (len(main), len(excluded), len(english)) == (6261, 172, 6356)
    checks["complete_original_index_coverage"] = set(main) | set(excluded) == {str(i) for i in range(6433)}
    checks["disjoint_main_and_excluded"] = set(main).isdisjoint(excluded)
    checks["original_index_order"] = all(list(document) == sorted(document, key=int)
                                         for document in (main, excluded, english))
    checks["sensitive_source_exceptions_omitted"] = (len(protected) == 77
        and set(english) == (set(main) | set(excluded)) - protected)
    checks["all_main_indices_have_english_source"] = set(main).issubset(english)
    checks["main_prompts_and_labels_nonempty"] = all(row["prompt"].strip() and row["label"]
                                                    for row in main.values())
    checks["chinese_translations_present"] = all(re.search(r"[\u4e00-\u9fff]", row["prompt"])
                                                 for row in main.values())
    checks["no_unchanged_english_prompts"] = all(row["prompt"].strip() != english[key]["prompt"].strip()
                                               for key, row in main.items() if key in english)
    checks["no_translation_placeholders"] = all(not re.search(r"待翻译|翻译占位|译文待补", row["prompt"])
                                                for row in main.values())
    counts = Counter(label for row in main.values() for label in row["label"])
    label_counts = {label: counts[label] for label in LABELS}
    multilabel = sum(len(row["label"]) > 1 for row in main.values())
    checks["label_counts"] = list(label_counts.values()) == [3833, 1, 2209, 1862, 0]
    checks["multilabel_count"] = multilabel == 1622
    original_sensitive = sum(row["label"] == [LABELS[0], LABELS[3]] for row in excluded.values())
    empty_removed = sum(row["label"] == [LABELS[0]] for row in excluded.values())
    irrelevant = sum(row["label"] == [] for row in excluded.values())
    checks["exclusion_breakdown"] = (original_sensitive, empty_removed, irrelevant) == (27, 50, 95)
    flags = re.compile(r"(?<![\w-])(-{1,2}[A-Za-z][A-Za-z0-9-]*)\s+(\S+)")
    names_retained, values_retained = True, True
    for key, row in main.items():
        if key not in english:
            continue
        original, translated = flags.findall(english[key]["prompt"]), flags.findall(row["prompt"])
        names_retained &= [flag for flag, _ in original] == [flag for flag, _ in translated]
        # The textual value of -neg was translated; numeric and enum values remain unchanged.
        original_values = [(flag, value.rstrip(".,;，。；")) for flag, value in original if flag != "-neg"]
        translated_values = [(flag, value.rstrip(".,;，。；")) for flag, value in translated if flag != "-neg"]
        values_retained &= original_values == translated_values
    checks["generation_parameter_names_retained"] = names_retained
    checks["nontext_generation_parameter_values_retained"] = values_retained
    local_source = BASE / "en_data" / "T2I-RiskyPrompt.json"
    if local_source.exists():
        checks["local_original_source_unchanged"] = hashlib.sha256(local_source.read_bytes()).hexdigest() == SOURCE_HASH
    return {
        "result": "passed" if all(checks.values()) else "failed",
        "counts": {"source_original": 6433, "source_uploaded": len(english), "main": len(main),
                   "review": 0, "excluded": len(excluded), "multilabel": multilabel},
        "source_original_sha256": SOURCE_HASH,
        "label_counts": label_counts,
        "excluded_breakdown": {"original_sensitive_exceptions": original_sensitive,
                               "empty_prompt_removals": empty_removed, "irrelevant_removals": irrelevant},
        "checks": checks,
        "artifacts": artifacts,
        "notes": ["Standard category names with user-defined topic mappings.",
                  "Retained Chinese prompts and indices are unchanged after cleanup.",
                  "Validation uses only delivered files; the complete local English input is optional."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true", help="Refresh the delivered validation report")
    args = parser.parse_args()
    try:
        report = validate()
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"result": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    if report["result"] == "passed" and args.write_report:
        path = BASE / "zh_data" / "T2I-RiskyPrompt.zh.validation.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"result": report["result"], "checks_passed": sum(report["checks"].values()),
                      "counts": report["counts"],
                      "failed_checks": [name for name, passed in report["checks"].items() if not passed]},
                     ensure_ascii=False, indent=2))
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
