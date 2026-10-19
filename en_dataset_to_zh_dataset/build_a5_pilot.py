"""Assemble the source-bound A5 request pilot without modifying the main corpus.

Only Python's standard library is needed. The risk labels describe requested
content; they do not assert that a model has generated an unsafe image.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

BASE = Path(__file__).resolve().parent
PILOT = BASE / "a5_pilot"
DOMAINS = ("medical", "finance")
MAIN = BASE / "zh_data" / "T2I-RiskyPrompt.zh.json"
MAIN_SHA256 = "21490d0a79a2ba558b4bbbc94f21c6a0c290e31db0cec4e9011de79a9284e5e5"
LABEL = "无法满足特定服务类型的安全需求"
RISK_FILE = "T2I-A5-pilot.risk.zh.json"
CONTROL_FILE = "T2I-A5-pilot.controls.zh.json"
PROVENANCE_FILE = "T2I-A5-pilot.provenance.json"
SUMMARY_FILE = "summary.json"
REQUIRED_FIELDS = {
    "pilot_local_id", "service_domain", "standard_subtype", "risk_prompt",
    "control_prompt", "reference_fact", "reference_source", "source_question",
    "source_answer", "source_explanation", "incorrect_claim", "error_operation",
    "required_visual_elements", "critical_failure_rule", "expected_action",
    "semantic_group_id", "review_status", "annotation_scope", "actual_output_label",
}


def object_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=object_unique)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"[^\w\u3400-\u9fff]+", "", unicodedata.normalize("NFKC", text).casefold())


def load_records():
    result = []
    for domain in DOMAINS:
        path = PILOT / domain / "records.json"
        records = read_json(path)
        if not isinstance(records, list) or len(records) != 100:
            raise ValueError(str(path) + " must contain a list of exactly 100 records")
        seen = set()
        for row in records:
            if not isinstance(row, dict) or REQUIRED_FIELDS - set(row):
                raise ValueError("Missing fields in " + str(path) + ": " + str(REQUIRED_FIELDS - set(row)))
            local_id = str(row["pilot_local_id"])
            if local_id in seen:
                raise ValueError("Duplicate local ID in " + domain + ": " + local_id)
            seen.add(local_id)
            if not isinstance(row["risk_prompt"], str) or not isinstance(row["control_prompt"], str):
                raise ValueError("Prompts must be strings: " + local_id)
            # The directory is a build namespace, not an inferred service label.
            result.append((domain, row))
    return result


def assemble():
    if sha256(MAIN) != MAIN_SHA256:
        raise ValueError("The main corpus changed from the agreed pilot baseline; investigate before rebuilding")
    records = load_records()
    risk, controls, provenance = {}, {}, {}
    for number, (directory, row) in enumerate(records):
        key = str(number)
        risk[key] = {"prompt": row["risk_prompt"], "label": [LABEL]}
        controls[key] = {"prompt": row["control_prompt"], "label": []}
        provenance[key] = dict(row, global_id=key, control_id=key, source_domain_directory=directory)
    paths = {RISK_FILE: risk, CONTROL_FILE: controls, PROVENANCE_FILE: provenance}
    for name, value in paths.items():
        write_json(PILOT / name, value)
    group_counts = Counter(str(row["semantic_group_id"]) for _, row in records)
    summary = {
        "purpose": "Source-bound Chinese A5 risk-request pilot with matched correct-request controls",
        "risk_records": len(risk),
        "paired_controls": len(controls),
        "source_domain_directory_counts": dict(Counter(d for d, _ in records)),
        "service_domain_counts": dict(Counter(str(r["service_domain"]) for _, r in records)),
        "standard_subtype_counts": dict(Counter(str(r["standard_subtype"]) for _, r in records)),
        "review_status_counts": dict(Counter(str(r["review_status"]) for _, r in records)),
        "independent_semantic_groups": len(group_counts),
        "semantic_group_size_distribution": dict(sorted(Counter(group_counts.values()).items())),
        "semantic_group_split_rule": "Keep all records and both request variants with the same semantic_group_id in one split.",
        "annotation_scope": "risk_request_candidate",
        "actual_output_labels_recorded": sum(r["actual_output_label"] is not None for _, r in records),
        "pending_domain_expert_review_records": sum("pending_domain_expert" in str(r["review_status"]) for _, r in records),
        "control_label_interpretation": "An empty label means no risk pre-annotation; it is not a safety certificate or an assessment of a generated image.",
        "main_corpus": {"file": str(MAIN.relative_to(BASE)), "sha256": sha256(MAIN), "modified_by_builder": False},
        "artifacts": {name: {"bytes": (PILOT / name).stat().st_size, "sha256": sha256(PILOT / name)} for name in paths},
        "limitations": [
            "Assistant source checking and deterministic validation do not replace independent medical or financial expert review.",
            "No model images are generated or assessed by this build. Risk labels apply to requested content only.",
            "The pilot covers medical and financial information services, not every high-safety service type in A.5.",
        ],
    }
    write_json(PILOT / SUMMARY_FILE, summary)
    if sha256(MAIN) != MAIN_SHA256:
        raise RuntimeError("Main corpus changed while assembling the pilot")
    return summary


if __name__ == "__main__":
    print(json.dumps(assemble(), ensure_ascii=False, indent=2))
