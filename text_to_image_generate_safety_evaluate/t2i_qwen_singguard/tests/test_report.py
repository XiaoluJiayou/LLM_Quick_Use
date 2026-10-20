"""无需模型或 GPU 的报告统计回归测试。"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import report
from utils import DEFAULT_RULES


def record(record_id, types=None, subtypes=None):
    return {
        "id": record_id, "prompt": f"sample {record_id}",
        "types": types or [], "subtypes": subtypes or [],
    }


def judgment(verdict="safe", category="Safe", parse_ok=True):
    return {"verdict": verdict, "category": category, "parse_ok": parse_ok}


class ReportTests(unittest.TestCase):
    def test_default_policy_groups_by_dataset_type(self):
        records = [record(1, ["违反社会主义核心价值观"], [])]
        rows = report.build_rows(
            records, {1: judgment()}, DEFAULT_RULES.values(),
            report.subtype_parents(records), False,
        )
        group = next(row for row in rows if row["层级"] == "大类")
        self.assertEqual(group["类别"], "违反社会主义核心价值观")
        self.assertEqual(group["题目数"], 1)
        self.assertEqual(group["已判定"], 1)
        self.assertEqual(group["命中率(%)"], "-")

    def test_parse_failures_are_excluded_from_valid_denominators(self):
        records = [record(i, ["违反社会主义核心价值观"]) for i in range(1, 5)]
        judgments = {
            1: judgment("unsafe", "违反社会主义核心价值观"),
            2: judgment(),
            3: judgment("unsafe", None, False),
        }
        row = report.summarize("总体", "全部", "", "", records, judgments, True)
        self.assertEqual(row["题目数"], 4)
        self.assertEqual(row["已判定"], 3)
        self.assertEqual(row["有效判定"], 2)
        self.assertEqual(row["违规数"], 1)
        self.assertEqual(row["违规率(%)"], "50.0")
        self.assertEqual(row["命中率(%)"], "50.0")
        self.assertEqual(row["解析失败率(%)"], "33.3")

    def test_hit_denominator_only_includes_mapped_valid_records(self):
        records = [record(1, ["违反社会主义核心价值观"]), record(2, ["未收录标签"])]
        judgments = {1: judgment("unsafe", "违反社会主义核心价值观"), 2: judgment()}
        row = report.summarize("总体", "全部", "", "", records, judgments, True)
        self.assertEqual(row["有效判定"], 2)
        self.assertEqual(row["可对照数"], 1)
        self.assertEqual(row["违规率(%)"], "50.0")
        self.assertEqual(row["命中率(%)"], "100.0")

    def test_no_judgments_does_not_report_zero_percent(self):
        row = report.summarize("总体", "全部", "", "", [record(1)], {}, True)
        self.assertEqual(row["已判定"], 0)
        for column in ("违规率(%)", "命中率(%)", "解析失败率(%)"):
            self.assertEqual(row[column], "-")

    def test_report_paths_separate_policy_mode_and_context(self):
        with mock.patch.object(config, "REPORT_FILE", Path("outputs/report.csv")):
            self.assertEqual(report.report_path("fast", "image", "bundled").name, "report_bundled.csv")
            self.assertEqual(report.report_path("fast", "image", "default").name, "report_default.csv")
            self.assertEqual(report.report_path("fast-slow", "prompt+image", "bundled").name,
                             "report_bundled_fast-slow_prompt_image.csv")
            paths = {
                report.report_path("fast", "image", "bundled"),
                report.report_path("fast", "image", "default"),
                report.report_path("fast", "image", "custom"),
                report.report_path("fast-slow", "image", "custom"),
                report.report_path("fast", "prompt+image", "custom"),
            }
            self.assertEqual(len(paths), 5)


if __name__ == "__main__":
    unittest.main()
