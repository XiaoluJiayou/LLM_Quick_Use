"""Offline structural and source-traceability checks for the A5 request pilot.

Passing these checks is not an expert assessment of the facts or of any image.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation, localcontext
import json
from pathlib import Path
import re

from build_a5_pilot import (
    BASE, PILOT, DOMAINS, MAIN, MAIN_SHA256, LABEL, RISK_FILE, CONTROL_FILE,
    PROVENANCE_FILE, REQUIRED_FIELDS, read_json, write_json, sha256, normalize,
)

SUBTYPES = {"A.5.a", "A.5.b", "A5(a)", "A5(b)", "A.5(a)", "A.5(b)"}
PLACEHOLDERS = re.compile(r"待补(?:充)?|待翻译|TODO|TBD|ZXQ|\uFFFD|<number>|<percent>|\{\{.+?\}\}", re.I)


def text_present(value):
    return isinstance(value, str) and bool(value.strip())


def numeric(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric calculation value")
    return Decimal(str(value).replace(",", "").strip())


def safe_arithmetic(expression):
    """Evaluate finite numeric arithmetic with an AST whitelist, never eval()."""
    if not isinstance(expression, str) or len(expression) > 1000:
        raise ValueError("An arithmetic expression must be a short string")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 160:
        raise ValueError("Expression is too complex")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if right != right.to_integral_value() or abs(right) > 100:
                raise ValueError("Only small integral powers are supported")
            return left ** int(right)
        raise ValueError("Expression contains unsupported operations")

    with localcontext() as context:
        context.prec = 40
        result = visit(tree)
    if not result.is_finite():
        raise ValueError("Calculation is not finite")
    return result


def calculation_check(record):
    check = record.get("calculation_check")
    if check is None:
        return {"status": "not_supplied"}
    if not isinstance(check, dict):
        return {"status": "not_checked", "reason": "Unrecognized calculation_check schema"}
    expression = check.get("expression")
    expected = check.get("expected", check.get("expected_result", check.get("result")))
    if expression is None or expected is None:
        return {"status": "not_checked", "reason": "No supported expression/expected fields; semantic checking is separate"}
    try:
        result = safe_arithmetic(expression)
        expected = numeric(expected)
        tolerance = numeric(check.get("absolute_tolerance", check.get("tolerance", "0.005")))
        if tolerance < 0:
            raise ValueError("Negative tolerance")
        return {"status": "passed" if abs(result - expected) <= tolerance else "failed", "expression": expression,
                "computed": str(result), "expected": str(expected), "absolute_tolerance": str(tolerance)}
    except (ValueError, SyntaxError, InvalidOperation, ZeroDivisionError, OverflowError) as exc:
        return {"status": "failed", "reason": str(exc)}


def manifest_entries(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("sources", "files", "downloads"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError("Unsupported manifest schema")


def within(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_source(domain, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Missing source file")
    path = Path(name)
    candidates = [path] if path.is_absolute() else [PILOT / domain / path, PILOT / path]
    candidates = list(dict.fromkeys(p.resolve() for p in candidates if within(p, PILOT) and p.is_file()))
    if len(candidates) != 1:
        raise ValueError("Source path is missing or ambiguous: " + name)
    return candidates[0]


def flatten_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)


def source_strings(path):
    if path.suffix.lower() == ".json":
        return {normalize(x) for x in flatten_strings(read_json(path)) if x.strip()}
    if path.suffix.lower() == ".jsonl":
        values = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                values.extend(flatten_strings(json.loads(line)))
        return {normalize(x) for x in values if x.strip()}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {normalize(v) for row in csv.DictReader(handle) for v in row.values() if isinstance(v, str) and v.strip()}
    raise ValueError("Source-question tracing unsupported for " + path.suffix)


def validate():
    errors = []
    checks = {}
    risk = read_json(PILOT / RISK_FILE)
    controls = read_json(PILOT / CONTROL_FILE)
    provenance = read_json(PILOT / PROVENANCE_FILE)
    keys = [str(i) for i in range(200)]
    checks["exactly_200_risk_requests"] = len(risk) == 200
    checks["exactly_200_paired_controls"] = len(controls) == 200
    checks["contiguous_ids_and_matching_provenance"] = list(risk) == list(controls) == list(provenance) == keys
    checks["risk_schema_and_full_A5_label"] = all(isinstance(x, dict) and set(x) == {"prompt", "label"} and text_present(x["prompt"]) and x["label"] == [LABEL] for x in risk.values())
    checks["control_schema_with_no_preassigned_risk"] = all(isinstance(x, dict) and set(x) == {"prompt", "label"} and text_present(x["prompt"]) and x["label"] == [] for x in controls.values())
    rows = list(provenance.values())
    domain_counts = Counter(x.get("source_domain_directory") for x in rows)
    checks["100_records_per_source_domain"] = domain_counts == {"medical": 100, "finance": 100}
    checks["complete_required_metadata"] = all(REQUIRED_FIELDS <= set(x) and all(text_present(x.get(k)) for k in ("pilot_local_id", "service_domain", "standard_subtype", "reference_fact", "source_question", "source_answer", "source_explanation", "incorrect_claim", "error_operation", "critical_failure_rule", "expected_action", "semantic_group_id", "review_status")) and isinstance(x.get("required_visual_elements"), list) and len(x["required_visual_elements"]) >= 2 and all(text_present(v) for v in x["required_visual_elements"]) for x in rows)
    checks["recognized_standard_subtypes"] = all(x.get("standard_subtype") in SUBTYPES for x in rows)
    checks["risk_request_scope_without_observed_output_labels"] = all(x.get("annotation_scope") == "risk_request_candidate" and x.get("actual_output_label") is None for x in rows)
    checks["review_status_truthfully_pending_domain_experts"] = all(x.get("review_status") == "assistant_source_checked_pending_domain_expert" for x in rows)
    all_prompts = [x.get("prompt", "") for x in list(risk.values()) + list(controls.values())]
    checks["Chinese_prompts_without_unresolved_placeholders"] = all(re.search(r"[\u3400-\u9fff]", x) and not PLACEHOLDERS.search(x) for x in all_prompts)
    checks["no_exact_normalized_risk_prompt_duplicates"] = len({normalize(x["prompt"]) for x in risk.values()}) == len(risk)
    checks["no_exact_normalized_control_prompt_duplicates"] = len({normalize(x["prompt"]) for x in controls.values()}) == len(controls)
    checks["risk_and_control_prompt_sets_disjoint"] = {normalize(x["prompt"]) for x in risk.values()}.isdisjoint(normalize(x["prompt"]) for x in controls.values())
    checks["pairs_match_source_records_and_differ"] = all(k in controls and k in provenance and provenance[k].get("global_id") == k and provenance[k].get("control_id") == k and provenance[k].get("risk_prompt") == x["prompt"] and provenance[k].get("control_prompt") == controls[k]["prompt"] and normalize(x["prompt"]) != normalize(controls[k]["prompt"]) for k, x in risk.items())
    checks["main_corpus_unchanged_from_8261_record_baseline"] = sha256(MAIN) == MAIN_SHA256

    manifests = {}
    manifest_file_status = []
    source_cache = {}
    for domain in DOMAINS:
        entries = manifest_entries(read_json(PILOT / domain / "raw" / "manifest.json"))
        manifest_by_path = {}
        for entry in entries:
            name = entry.get("file", entry.get("filename", entry.get("path")))
            try:
                path = resolve_source(domain, name)
                status = sha256(path) == str(entry.get("sha256", "")).lower()
                manifest_by_path[path] = entry
                manifest_file_status.append({"domain": domain, "file": str(path.relative_to(PILOT)), "sha256_matches": status})
            except (ValueError, OSError) as exc:
                manifest_file_status.append({"domain": domain, "file": name, "sha256_matches": False, "error": str(exc)})
        manifests[domain] = manifest_by_path
    checks["raw_files_match_download_manifests"] = bool(manifest_file_status) and all(x["sha256_matches"] for x in manifest_file_status)

    references = []
    for key, row in provenance.items():
        domain = row.get("source_domain_directory")
        source = row.get("reference_source", {})
        result = {"global_id": key, "source_question_recovered": False, "metadata_matches_manifest": False, "source_hash_matches": False}
        try:
            if not isinstance(source, dict) or not {"dataset", "revision", "url", "file", "row_id", "sha256"} <= set(source):
                raise ValueError("Missing reference_source fields")
            if not text_present(source["dataset"]) or not re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]) or not str(source["url"]).startswith("https://") or not str(source["row_id"]).strip():
                raise ValueError("Invalid dataset, immutable revision, URL or row_id")
            path = resolve_source(domain, source["file"])
            entry = manifests[domain].get(path)
            if entry is None:
                raise ValueError("Reference source is absent from the download manifest")
            result["source_hash_matches"] = sha256(path) == str(source["sha256"]).lower() == str(entry.get("sha256", "")).lower()
            result["metadata_matches_manifest"] = source["revision"] == entry.get("revision") and source["url"] == entry.get("url")
            if path not in source_cache:
                source_cache[path] = source_strings(path)
            result["source_question_recovered"] = normalize(row["source_question"]) in source_cache[path]
            result["trace_scope"] = "Exact normalized question found among source-file JSON/CSV string fields; row-id association is not inferred by this generic check."
        except (ValueError, KeyError, OSError, TypeError) as exc:
            result["error"] = str(exc)
        references.append(result)
    checks["all_references_match_source_file_hashes"] = bool(references) and all(x["source_hash_matches"] for x in references)
    checks["immutable_source_revisions_and_urls_match_manifests"] = bool(references) and all(x["metadata_matches_manifest"] for x in references)
    checks["all_questions_recoverable_in_raw_source_files"] = bool(references) and all(x["source_question_recovered"] for x in references)
    # Verify provenance is an unmodified copy of the two source-bound domain sets.
    original_records = {}
    for domain in DOMAINS:
        for row in read_json(PILOT / domain / "records.json"):
            original_records[(domain, str(row["pilot_local_id"]))] = row
    checks["provenance_preserves_domain_records"] = all({k: v for k, v in row.items() if k not in {"global_id", "control_id", "source_domain_directory"}} == original_records.get((row.get("source_domain_directory"), str(row.get("pilot_local_id")))) for row in rows)
    calculations = [{"global_id": k, **calculation_check(row)} for k, row in provenance.items()]
    calculation_counts = Counter(x["status"] for x in calculations)
    checks["no_failed_supported_arithmetic_checks"] = calculation_counts["failed"] == 0
    semantic_groups = Counter(str(x.get("semantic_group_id")) for x in rows)
    failed = [key for key, passed in checks.items() if not passed]
    for item in references:
        if not all(item.get(k) for k in ("source_question_recovered", "metadata_matches_manifest", "source_hash_matches")):
            errors.append(item)
    report = {
        "result": "passed" if not failed else "failed", "check_count": len(checks), "checks": checks,
        "failed_checks": failed, "risk_records": len(risk), "control_records": len(controls),
        "source_domain_counts": dict(domain_counts),
        "service_domain_counts": dict(Counter(x.get("service_domain") for x in rows)),
        "standard_subtype_counts": dict(Counter(x.get("standard_subtype") for x in rows)),
        "review_status_counts": dict(Counter(x.get("review_status") for x in rows)),
        "pending_domain_expert_review_records": sum("pending_domain_expert" in str(x.get("review_status")) for x in rows),
        "independent_semantic_groups": len(semantic_groups),
        "semantic_group_size_distribution": dict(sorted(Counter(semantic_groups.values()).items())),
        "calculation_check_status_counts": dict(calculation_counts),
        "calculation_results": [x for x in calculations if x["status"] != "not_supplied"],
        "source_file_checks": manifest_file_status,
        "source_traceability": {"questions_checked": len(references), "questions_recovered": sum(x["source_question_recovered"] for x in references), "scope": "Exact normalized question-string recovery within the source file. Source row IDs and question-answer correctness still require domain source review.", "errors": errors},
        "expert_fact_verification": {"status": "pending", "records": len(rows)},
        "generated_image_evaluation": {"status": "not_performed", "images_generated": 0, "output_labels_assigned": sum(x.get("actual_output_label") is not None for x in rows)},
        "main_corpus_sha256": sha256(MAIN),
        "limitations": [
            "Passing checks establish file integrity, schema, pairing, exact text uniqueness and source-question recovery; they do not establish clinical or financial correctness.",
            "Unrecognized calculation schemas are reported as not_checked rather than treated as validated arithmetic.",
            "The 200 risk labels apply to requests for faulty professional content, not observed image outputs.",
            "Empty control labels mean no risk pre-annotation. They do not certify that a request or generated output is safe.",
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    try:
        report = validate()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"result": "failed", "setup_error": str(exc), "limitations": ["Validation did not complete; no checks should be inferred to have passed."]}
    if args.write_report:
        write_json(PILOT / "validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
