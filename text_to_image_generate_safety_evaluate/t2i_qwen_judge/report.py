"""汇总裁判模型的判定结果。

国标策略按五大类分组，其他策略按数据集标签分组。
解析失败单独统计，不混入违规率和命中率的有效判定分母。
"""
import argparse
import csv
import logging
import sys
import unicodedata
from collections import Counter

import config
from utils import (
    GB_CATEGORIES,
    expected_gb_categories,
    latest_matching_judgments,
    load_policy,
    load_prompts,
    positive_int,
    resolve_policy_path,
    select_per_type,
)

COLUMNS = [
    "层级", "类别", "所属大类", "期望国标类",
    "题目数", "已判定", "有效判定", "可对照数", "违规数", "违规率(%)", "命中率(%)", "解析失败率(%)",
]


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def report_path(mode, context="image", fingerprint=None):
    parts = []
    if fingerprint is not None:
        parts.append(fingerprint)
    if mode != "fast":
        parts.append(mode)
    if context != "image":
        parts.append(context.replace("+", "_"))
    if not parts:
        return config.REPORT_FILE
    return config.REPORT_FILE.with_name("report_" + "_".join(parts) + ".csv")


def percent(n, total):
    return f"{n / total * 100:.1f}" if total else "-"


def summarize(level, name, parent, expected, group, judgments, hit_enabled):
    judged = [(r, judgments[r["id"]]) for r in group if r["id"] in judgments]
    valid = [(r, j) for r, j in judged if j["parse_ok"] and j["verdict"] in ("safe", "unsafe")]
    unsafe = [(r, j) for r, j in valid if j["verdict"] == "unsafe"]
    comparable = [(r, j) for r, j in valid if expected_gb_categories(r)] if hit_enabled else []
    if hit_enabled:
        hits = sum(j["verdict"] == "unsafe" and j["category"] in expected_gb_categories(r)
                   for r, j in comparable)
        hit_rate = percent(hits, len(comparable))
    else:
        hit_rate = "-"
    parse_failed = sum(not j["parse_ok"] for _, j in judged)
    return {
        "层级": level,
        "类别": name,
        "所属大类": parent,
        "期望国标类": expected,
        "题目数": len(group),
        "已判定": len(judged),
        "有效判定": len(valid),
        "可对照数": len(comparable) if hit_enabled else "-",
        "违规数": len(unsafe),
        "违规率(%)": percent(len(unsafe), len(valid)),
        "命中率(%)": hit_rate,
        "解析失败率(%)": percent(parse_failed, len(judged)),
    }


def subtype_parents(all_records):
    """数据集没有标注子类属于哪个大类，取它在整个数据集里最常对应的第一个大类。"""
    votes = {}
    for r in all_records:
        for sub in r["subtypes"]:
            votes.setdefault(sub, Counter())[r["types"][0] if r["types"] else ""] += 1
    return {sub: v.most_common(1)[0][0] for sub, v in votes.items()}


def build_rows(records, judgments, rule_names, parents, hit_enabled):
    """国标策略按规则顺序分组；其他策略按原始数据集大类分组。"""
    rule_names = list(rule_names)
    present = {s for r in records for s in r["subtypes"]}
    type_order = list(dict.fromkeys(t for r in records for t in r["types"]))
    subtypes = sorted(
        (s for s in parents if s in present),
        key=lambda s: type_order.index(parents[s]) if parents[s] in type_order else len(type_order),
    )

    rows = [summarize("总体", "全部", "", "", records, judgments, hit_enabled)]
    group_names = rule_names if hit_enabled else type_order
    for name in group_names:
        if hit_enabled:
            group = [r for r in records if name in expected_gb_categories(r)]
            expected = name
        else:
            group = [r for r in records if name in r["types"]]
            expected = ""
        rows.append(summarize("大类", name, "", expected, group, judgments, hit_enabled))
    for sub in subtypes:
        group = [r for r in records if sub in r["subtypes"]]
        expected = "、".join(sorted(set().union(*(expected_gb_categories(r) for r in group)))) if hit_enabled else ""
        rows.append(summarize("子类", sub, parents[sub], expected, group, judgments, hit_enabled))
    return rows


def display_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def format_table(rows):
    table = [COLUMNS] + [[str(row[c]) for c in COLUMNS] for row in rows]
    widths = [max(display_width(line[i]) for line in table) for i in range(len(COLUMNS))]
    return "\n".join(
        "  ".join(cell + " " * (w - display_width(cell)) for cell, w in zip(line, widths)).rstrip()
        for line in table
    )


def main():
    parser = argparse.ArgumentParser(description="汇总裁判模型判定结果，生成评测报告")
    parser.add_argument("--per-type", type=positive_int, default=None, help="每个大类只取前 k 条（与小样本试跑保持一致），默认全量")
    parser.add_argument("--mode", choices=("fast", "fast-slow"), default=config.JUDGE_MODE,
                        help="只统计该模式的判定。默认读配置或环境变量 JUDGE_MODE")
    parser.add_argument("--policy", default=None,
                        help="策略文件路径，或 default（裁判模型内置分类）。默认用 config.POLICY_FILE")
    parser.add_argument("--with-prompt", action="store_true",
                        help="只统计「提示词+图片」输入方式的判定结果")
    args = parser.parse_args()
    init_logging()

    context = "prompt+image" if args.with_prompt else "image"
    policy_arg = resolve_policy_path(args.policy)
    _, rules, fingerprint = load_policy(policy_arg)
    hit_enabled = set(rules.values()) == GB_CATEGORIES

    all_records = load_prompts()
    records = select_per_type(all_records, args.per_type) if args.per_type is not None else all_records
    judgments, stale = latest_matching_judgments(records, fingerprint, args.mode, context)
    if stale:
        logging.warning(f"忽略了 {stale} 条与当前数据、图片、模型、策略、模式或输入方式不匹配的判定记录")
    if not hit_enabled:
        logging.info("当前策略不使用完整国标五大类，按数据集标签分组，命中率显示为 -")

    rows = build_rows(records, judgments, rules.values(), subtype_parents(all_records), hit_enabled)

    out = report_path(args.mode, context, fingerprint)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(format_table(rows))
    overall = rows[0]
    logging.info(
        f"出图模型 {config.T2I_MODEL_NAME}，模式 {args.mode}，输入 {context}："
        f"已判定 {overall['已判定']}/{overall['题目数']}，"
        f"有效判定 {overall['有效判定']}，可对照 {overall['可对照数']}，"
        f"违规率 {overall['违规率(%)']}%，命中率 {overall['命中率(%)']}%，解析失败率 {overall['解析失败率(%)']}%"
    )
    if overall["已判定"] < overall["题目数"] or overall["有效判定"] < overall["已判定"]:
        logging.warning("当前报告包含未判定或解析失败样本，请结合覆盖率阅读，不能视为完整评测")
    logging.info(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
