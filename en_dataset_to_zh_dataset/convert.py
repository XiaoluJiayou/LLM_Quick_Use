"""英文数据集 → 中文指令及风险分类；带来源校验的断点续跑。"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from llm_quick_use.io import append_jsonl, atomic_write_json, atomic_write_text, read_jsonl, stable_fingerprint
from llm_quick_use.parsing import extract_json

BASE_DIR = Path(__file__).resolve().parent
GBT_FILE = BASE_DIR / "gbt_categories.md"
GB_LABELS = (
    "违反社会主义核心价值观", "歧视性内容", "商业违法违规", "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求", "无法映射",
)
SYSTEM_PROMPT_TMPL = """你是数据标注助手。任务：
1. 把英文指令准确翻译成简体中文（保留原意，不扩写、不删减风险内容）。
2. 根据英文指令内容与原始类别，将风险归入 GB/T 45654-2025 附录 A 五大类之一。
五大类说明：
{categories}
规则：
- 只输出一个 JSON 对象，不输出 markdown 或其它文字。
- 格式严格为：{{"zh": "中文指令", "gb": "类别名称"}}
- gb 必须是下列之一：{labels}
- 无法归入任一大类时 gb 填「无法映射」，不要编造类别。
- 同时符合多类时，按上述五大类顺序选择最靠前的一类。
"""


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else ((base or BASE_DIR) / p).resolve()


def load_dataset_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，请安装项目 data 依赖组。") from exc
    with path.open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置必须是 YAML 对象: {path}")
    for key in ("input", "format", "text_field", "category_field", "output"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"配置缺少有效字符串字段: {key}")
    cfg["format"] = cfg["format"].lower().strip()
    if cfg["format"] not in ("csv", "json", "jsonl"):
        raise ValueError(f"不支持的 format: {cfg['format']}")
    cfg["_config_dir"] = path.parent
    return cfg


def input_path(cfg: dict[str, Any]) -> Path:
    inp = Path(cfg["input"])
    if not inp.is_absolute():
        candidate = cfg["_config_dir"] / inp
        inp = candidate if candidate.exists() else BASE_DIR / inp
    return inp.resolve()


def read_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    inp = input_path(cfg)
    fmt = cfg["format"]
    if fmt == "csv":
        with inp.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if not header or any(not name.strip() for name in header) or len(header) != len(set(header)):
                raise ValueError("CSV 缺少表头或含重复列")
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError("CSV 存在列数与表头不一致的数据行")
            return rows
    if fmt == "jsonl":
        if not inp.exists():
            raise FileNotFoundError(inp)
        return read_jsonl(inp, strict=True)
    with inp.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        key = cfg.get("json_key")
        if not key or key not in data:
            raise ValueError("JSON 根节点是对象时必须配置有效 json_key")
        cfg["_json_envelope"] = {name: value for name, value in data.items() if name != key}
        data = data[key]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("JSON 数据必须是对象列表")
    return data


def write_records(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    out = _resolve(cfg["output"])
    if out == input_path(cfg):
        raise ValueError("输出路径不能与输入路径相同")
    fmt = cfg["format"]
    if fmt == "csv":
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        if not fieldnames:
            fieldnames = list(cfg.get("_input_fields", ()))
            fieldnames.extend(key for key in ("中文指令", "国标类别", "原类别") if key not in fieldnames)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(out, stream.getvalue(), encoding="utf-8-sig")
    elif fmt == "jsonl":
        atomic_write_text(out, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    else:
        key = cfg.get("json_key")
        atomic_write_json(out, {**cfg.get("_json_envelope", {}), key: rows} if key else rows)
    return out


def record_id(cfg: dict[str, Any], row: dict[str, Any], index: int) -> str:
    field = cfg.get("id_field")
    if field:
        value = row.get(field)
        if value is None or not str(value).strip() or isinstance(value, (dict, list)):
            raise ValueError(f"第 {index + 1} 条缺少有效唯一 ID: {field}")
        return str(value)
    return str(index)


def category_text(cfg: dict[str, Any], row: dict[str, Any]) -> str:
    raw = row.get(cfg["category_field"], "")
    if isinstance(raw, list):
        return " | ".join(str(item) for item in raw)
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return "" if raw is None else str(raw)


def progress_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".progress.jsonl")


def errors_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".errors.jsonl")


def load_done_ids(path: Path, fingerprint: str | None = None) -> dict[str, dict[str, str]]:
    records = read_jsonl(path, strict=True)
    if not records:
        if path.exists():
            raise ValueError(f"进度文件为空，无法验证来源；请换用新输出路径：{path}")
        return {}
    metadata = records[0]
    if metadata.get("kind") != "run" or not metadata.get("fingerprint"):
        raise ValueError(f"旧进度缺少来源指纹，请换用新输出路径：{path}")
    if fingerprint is not None and metadata["fingerprint"] != fingerprint:
        raise ValueError("输入、配置、提示词或模型已改变；请使用新的输出路径，避免混用旧结果")
    done = {}
    for row in records[1:]:
        rid = str(row.get("id", ""))
        if not rid or rid in done or not isinstance(row.get("zh"), str) or not row["zh"].strip() or row.get("gb") not in GB_LABELS:
            raise ValueError(f"进度文件包含重复 ID 或无效结果：{path}")
        done[rid] = {"zh": row["zh"], "gb": row["gb"]}
    return done


def build_client():
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError("请设置 OPENAI_BASE_URL 和 OPENAI_MODEL")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai，请安装项目 api 依赖组。") from exc
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "EMPTY"
    return OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=0), model


def normalize_gb(value: str) -> str:
    value = value.strip()
    if value in GB_LABELS:
        return value
    match = re.fullmatch(r"([A-Ea-e])(?:[.、\s]+.*)?", value)
    if match:
        return GB_LABELS[ord(match.group(1).upper()) - ord("A")]
    raise ValueError("gb 字段不属于允许的分类")


def call_model(client, model: str, system: str, en_text: str, category: str) -> dict[str, str]:
    user = f"英文指令：\n{en_text}\n\n原始类别：\n{category or '(无)'}\n\n请返回 JSON。"
    last_err = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("模型输出被截断")
            obj = extract_json(choice.message.content or "", expected_type=dict)
            if not isinstance(obj.get("zh"), str) or not obj["zh"].strip() or not isinstance(obj.get("gb"), str):
                raise ValueError("zh 和 gb 必须是非空字符串")
            return {"zh": obj["zh"].strip(), "gb": normalize_gb(obj["gb"])}
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"模型调用或结果校验失败：{last_err}") from last_err


def convert(dataset_cfg: Path, limit: int | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit 必须是正整数")
    cfg = load_dataset_config(dataset_cfg)
    all_rows = read_records(cfg)
    ids = [record_id(cfg, row, i) for i, row in enumerate(all_rows)]
    id_set = set(ids)
    if len(ids) != len(id_set):
        raise ValueError("输入包含重复 ID；请修正 id_field 或原数据")
    for index, row in enumerate(all_rows):
        text = row.get(cfg["text_field"])
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {index + 1} 条缺少非空字符串字段：{cfg['text_field']}")
        if cfg["category_field"] not in row:
            raise ValueError(f"第 {index + 1} 条缺少类别字段：{cfg['category_field']}")
    cfg["_input_fields"] = list(all_rows[0]) if all_rows else []
    rows = all_rows[:limit] if limit is not None else all_rows
    out = _resolve(cfg["output"])
    if out == input_path(cfg):
        raise ValueError("输出路径不能与输入路径相同")
    if limit is not None:
        out = out.with_name(f"{out.stem}.limit-{limit}{out.suffix}")
    categories = GBT_FILE.read_text(encoding="utf-8")
    system = SYSTEM_PROMPT_TMPL.format(categories=categories, labels="、".join(GB_LABELS))
    fingerprint = stable_fingerprint({
        "version": 1, "rows": all_rows,
        "config": {key: value for key, value in cfg.items() if not key.startswith("_")},
        "system": system, "model": os.environ.get("OPENAI_MODEL", "").strip(),
        "base_url": os.environ.get("OPENAI_BASE_URL", "").strip(),
    })
    prog = progress_path(out)
    done = load_done_ids(prog, fingerprint)
    if any(rid not in id_set for rid in done):
        raise ValueError("进度文件含当前输入不存在的 ID")
    if out.exists() and not prog.exists():
        raise ValueError(f"输出已存在但没有来源进度，已保留：{out}；请换用新输出路径")
    plan = {"input_rows": len(all_rows), "selected_rows": len(rows), "resumable_rows": len(done), "output": str(out)}
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    needed = any(rid not in done for rid in ids[:len(rows)])
    client, model = build_client() if needed else (None, "")
    if not prog.exists():
        append_jsonl(prog, {"kind": "run", "fingerprint": fingerprint})
    results = []
    failed = 0
    ok_new = 0
    for index, row in enumerate(rows):
        rid = ids[index]
        mapped = done.get(rid)
        if mapped is None:
            try:
                mapped = call_model(client, model, system, row[cfg["text_field"]], category_text(cfg, row))
                append_jsonl(prog, {"id": rid, **mapped})
                done[rid] = mapped
                ok_new += 1
            except Exception as exc:
                failed += 1
                append_jsonl(errors_path(out), {"id": rid, "error": str(exc)})
                print(f"[失败] id={rid}: {exc}", file=sys.stderr)
                continue
        results.append({**row, "中文指令": mapped["zh"], "国标类别": mapped["gb"], "原类别": category_text(cfg, row)})
    write_records({**cfg, "output": str(out)}, results)
    summary = {**plan, "new": ok_new, "failed": failed, "written": len(results)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="数据集 YAML，路径相对本脚本目录")
    parser.add_argument("--limit", type=int, help="只处理前 N 条，写入独立 .limit-N 输出")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入和续跑来源，不调用模型")
    args = parser.parse_args(argv)
    try:
        summary = convert(_resolve(args.dataset), args.limit, dry_run=args.dry_run)
        return 1 if summary.get("failed") else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
