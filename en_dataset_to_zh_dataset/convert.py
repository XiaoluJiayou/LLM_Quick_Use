"""英文数据集 → 中文指令 + GB/T 45654-2025 附录 A 五大类。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
GBT_FILE = BASE_DIR / "gbt_categories.md"

GB_LABELS = (
    "违反社会主义核心价值观",
    "歧视性内容",
    "商业违法违规",
    "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求",
    "无法映射",
)

SYSTEM_PROMPT_TMPL = """你是数据标注助手。任务：
1. 把英文指令准确翻译成简体中文（保留原意，不要扩写、不要删减风险内容）。
2. 根据英文指令内容与原始类别，将风险归入 GB/T 45654-2025 附录 A 的五大类之一。

五大类说明：
{categories}

规则：
- 只输出一个 JSON 对象，不要 markdown 代码块，不要其它文字。
- 格式严格为：{{"zh": "中文指令", "gb": "类别名称"}}
- gb 必须是下列之一：{labels}
- 无法归入任一大类时 gb 填「无法映射」，不要编造类别。
- 同时符合多类时，按「违反社会主义核心价值观」「歧视性内容」「商业违法违规」「侵犯他人合法权益」「无法满足特定服务类型的安全需求」的顺序选最靠前的一类。
"""


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or BASE_DIR) / p


def load_dataset_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置必须是 YAML 对象: {path}")
    for key in ("input", "format", "text_field", "category_field", "output"):
        if key not in cfg or cfg[key] in (None, ""):
            raise ValueError(f"配置缺少字段: {key}")
    fmt = str(cfg["format"]).lower().strip()
    if fmt not in ("csv", "json", "jsonl"):
        raise ValueError(f"不支持的 format: {fmt}")
    cfg["format"] = fmt
    cfg["_config_dir"] = path.parent
    return cfg


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def read_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = cfg["_config_dir"]
    # 相对路径优先相对配置文件目录，再回退项目根
    inp = Path(cfg["input"])
    if not inp.is_absolute():
        cand = base / inp
        inp = cand if cand.exists() else BASE_DIR / inp
    fmt = cfg["format"]

    if fmt == "csv":
        with inp.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if fmt == "jsonl":
        rows: list[dict[str, Any]] = []
        with inp.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"jsonl 第 {line_no} 行不是对象")
                rows.append(obj)
        return rows

    # json
    with inp.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        key = cfg.get("json_key")
        if not key:
            raise ValueError("json 根节点是对象时必须配置 json_key")
        rows = data[key]
    else:
        raise ValueError("json 根节点必须是列表或对象")
    if not isinstance(rows, list):
        raise ValueError("json 数据必须是对象列表")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"json 第 {i} 条不是对象")
    return rows


def write_records(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    out = _resolve(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = cfg["format"]

    if fmt == "csv":
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return out

    if fmt == "jsonl":
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out

    # json
    key = cfg.get("json_key")
    payload: Any = {key: rows} if key else rows
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def record_id(cfg: dict[str, Any], row: dict[str, Any], index: int) -> str:
    field = cfg.get("id_field")
    if field and field in row and row[field] not in (None, ""):
        return str(row[field])
    return str(index)


def category_text(cfg: dict[str, Any], row: dict[str, Any]) -> str:
    field = cfg["category_field"]
    raw = row.get(field, "")
    if isinstance(raw, list):
        return " | ".join(str(x) for x in raw)
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return "" if raw is None else str(raw)


def progress_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".progress.jsonl")


def errors_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".errors.jsonl")


def load_done_ids(path: Path) -> dict[str, dict[str, str]]:
    done: dict[str, dict[str, str]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(obj.get("id", ""))
            if rid and "zh" in obj and "gb" in obj:
                done[rid] = {"zh": str(obj["zh"]), "gb": str(obj["gb"])}
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_client() -> tuple[OpenAI, str]:
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "EMPTY"
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not base_url:
        raise SystemExit("请设置环境变量 OPENAI_BASE_URL")
    if not model:
        raise SystemExit("请设置环境变量 OPENAI_MODEL")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        obj = json.loads(fence.group(1).strip())
        if isinstance(obj, dict):
            return obj
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"无法解析 JSON: {text[:200]}")


def normalize_gb(value: str) -> str:
    v = value.strip()
    for label in GB_LABELS:
        if v == label or label in v:
            return label
    # 允许模型用 A/B/C/D/E
    letter_map = {
        "A": GB_LABELS[0],
        "B": GB_LABELS[1],
        "C": GB_LABELS[2],
        "D": GB_LABELS[3],
        "E": GB_LABELS[4],
    }
    m = re.match(r"^([A-Ea-e])[\.、\s]", v)
    if m:
        return letter_map[m.group(1).upper()]
    if v.upper() in letter_map:
        return letter_map[v.upper()]
    return "无法映射"


def call_model(
    client: OpenAI,
    model: str,
    system: str,
    en_text: str,
    category: str,
) -> dict[str, str]:
    user = (
        f"英文指令：\n{en_text}\n\n"
        f"原始类别：\n{category or '(无)'}\n\n"
        '请返回 JSON：{"zh": "...", "gb": "..."}'
    )
    last_err: Exception | None = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
            obj = extract_json(content)
            zh = str(obj.get("zh", "")).strip()
            gb = normalize_gb(str(obj.get("gb", "")).strip())
            if not zh:
                raise ValueError("缺少 zh 字段")
            if gb not in GB_LABELS:
                gb = "无法映射"
            return {"zh": zh, "gb": gb}
        except Exception as e:  # noqa: BLE001 — 单条失败不中断整批
            last_err = e
    raise RuntimeError(str(last_err))


def convert(dataset_cfg: Path, limit: int | None) -> None:
    cfg = load_dataset_config(dataset_cfg)
    rows = read_records(cfg)
    if limit is not None:
        rows = rows[: max(0, limit)]

    out = _resolve(cfg["output"])
    prog = progress_path(out)
    err_path = errors_path(out)
    done = load_done_ids(prog)

    categories = GBT_FILE.read_text(encoding="utf-8")
    system = SYSTEM_PROMPT_TMPL.format(
        categories=categories,
        labels="、".join(GB_LABELS),
    )
    client, model = build_client()

    text_field = cfg["text_field"]
    results: list[dict[str, Any]] = []
    ok_new = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows):
        rid = record_id(cfg, row, i)
        en_text = row.get(text_field, "")
        en_text = "" if en_text is None else str(en_text)
        cat = category_text(cfg, row)

        if rid in done:
            mapped = done[rid]
            skipped += 1
        else:
            try:
                mapped = call_model(client, model, system, en_text, cat)
                append_jsonl(
                    prog,
                    {
                        "id": rid,
                        "zh": mapped["zh"],
                        "gb": mapped["gb"],
                        "原类别": cat,
                    },
                )
                done[rid] = mapped
                ok_new += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                append_jsonl(
                    err_path,
                    {
                        "id": rid,
                        "error": str(e),
                        "text": en_text[:500],
                        "原类别": cat,
                    },
                )
                print(f"[跳过] id={rid}: {e}", file=sys.stderr)
                continue

        out_row = dict(row)
        out_row["中文指令"] = mapped["zh"]
        out_row["国标类别"] = mapped["gb"]
        out_row["原类别"] = cat
        results.append(out_row)
        print(f"[{i + 1}/{len(rows)}] id={rid} → {mapped['gb']}")

    written = write_records({**cfg, "output": str(out)}, results)
    print(
        f"完成: 新成功 {ok_new}, 续跑跳过 {skipped}, 失败 {failed}; "
        f"输出 {written}"
    )
    if failed:
        print(f"错误日志: {err_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="英文数据集翻译为中文并映射 GB/T 45654-2025 附录 A 五大类"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="数据集 yaml 配置路径，例如 configs/example.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条，用于小样核对",
    )
    args = parser.parse_args()
    dataset_path = _resolve(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"找不到配置: {dataset_path}")
    convert(dataset_path, args.limit)


if __name__ == "__main__":
    main()
