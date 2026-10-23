"""数据校验、结果解析和续跑隔离的离线回归，无需模型或 GPU。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import config
import utils

GB_A = "违反社会主义核心价值观"
GB_B = "歧视性内容"
GB_C = "商业违法违规"


class UtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.history = self.root / "judgments.jsonl"
        self.patch = mock.patch.multiple(
            config, IMAGE_DIR=self.images, JUDGMENTS_FILE=self.history,
            GENERATIONS_FILE=self.root / "generations.jsonl",
            DATA_JSON=self.root / "data.json", JUDGE_MODEL_DIR=self.root / "judge",
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.record = {"id": 1, "prompt": "A blue sky", "types": [GB_A], "subtypes": []}
        self.rules = {"A": GB_A, "B": GB_B}

    def write_json(self, data):
        config.DATA_JSON.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    def judgment(self, **changes):
        if not utils.image_path(1).exists():
            Image.new("RGB", (32, 32), "blue").save(utils.image_path(1))
        utils.append_jsonl(config.GENERATIONS_FILE, {
            "id": 1, "generation_fingerprint": utils.generation_fingerprint(self.record),
            "image_sha256": utils.file_sha256(utils.image_path(1)),
        })
        item = {
            "id": 1, "policy": "policy1", "mode": "fast", "context": "image",
            "record_fingerprint": utils.record_fingerprint(self.record),
            "image_sha256": utils.file_sha256(utils.image_path(1)),
            "judge_model": str(config.JUDGE_MODEL_DIR.resolve()),
            "parse_ok": True, "verdict": "unsafe", "category": self.rules["A"],
        }
        item.update(changes)
        return item

    def test_json_loads_gb_full_names_and_normalizes(self):
        self.write_json({
            "9": {
                "prompt": "  测试提示  ",
                "label": ["包含歧视性内容", "包含违反社会主义核心价值观的内容"],
            },
            "2": {"prompt": "另一条", "label": ["商业违法违规"]},
        })
        records = utils.load_prompts()
        self.assertEqual([r["id"] for r in records], [2, 9])
        self.assertEqual(records[1]["prompt"], "测试提示")
        self.assertEqual(records[1]["types"], [GB_A, GB_B])
        self.assertEqual(records[1]["subtypes"], [])
        self.assertEqual(records[0]["types"], [GB_C])

    def test_json_rejects_padded_or_non_integer_ids(self):
        for key in ("01", "1.0"):
            with self.subTest(key=key):
                self.write_json({key: {"prompt": "Sky", "label": [GB_A]}})
                with self.assertRaisesRegex(ValueError, "非负整数"):
                    utils.load_prompts()

    def test_json_rejects_wrong_schema(self):
        self.write_json({"1": {"prompt": "Sky", "label": [GB_A], "extra": 1}})
        with self.assertRaisesRegex(ValueError, "只允许 prompt、label"):
            utils.load_prompts()

    def test_json_rejects_empty_invalid_or_overflow_data(self):
        cases = [
            {"1": {"prompt": "  ", "label": [GB_A]}},
            {"bad": {"prompt": "Sky", "label": [GB_A]}},
            {"-1": {"prompt": "Sky", "label": [GB_A]}},
            {str(2 ** 64): {"prompt": "Sky", "label": [GB_A]}},
            {"1": {"prompt": "Sky", "label": ["未收录标签"]}},
        ]
        for data in cases:
            with self.subTest(data=data):
                self.write_json(data)
                with self.assertRaises(ValueError):
                    utils.load_prompts()

    def test_json_rejects_non_object_or_empty_dataset(self):
        for payload in ("[]", "{}", "null"):
            with self.subTest(payload=payload):
                config.DATA_JSON.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    utils.load_prompts()

    def test_missing_dataset_explains_how_to_supply_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "DATA_JSON|T2I-RiskyPrompt"):
            utils.load_prompts()

    def test_per_type_zero_or_negative_never_means_full_dataset(self):
        for invalid in (0, -1, "no"):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                utils.select_per_type([self.record], invalid)

    def test_per_type_selection_uses_lowest_ids(self):
        records = [dict(self.record, id=5), dict(self.record, id=1),
                   dict(self.record, id=2, types=[GB_B])]
        self.assertEqual([r["id"] for r in utils.select_per_type(records, 1)], [1, 2])

    def test_jsonl_resume_separates_partial_tail_from_new_record(self):
        self.history.write_bytes(b'{"id": 1}\n{"id":')
        utils.append_jsonl(self.history, {"id": 2, "text": "完成"})
        self.assertEqual([row["id"] for row in utils.read_jsonl(self.history)], [1, 2])

    def test_jsonl_recovers_partial_utf8_and_ignores_nonobjects(self):
        self.history.write_bytes(b'{"id": 1}\nnull\n[]\n{"text":"\xe4\xb8')
        utils.append_jsonl(self.history, {"id": 2})
        self.assertEqual(utils.read_jsonl(self.history), [{"id": 1}, {"id": 2}])

    def test_jsonl_preserves_last_complete_line_without_newline(self):
        self.history.write_text('{"id":1}', encoding="utf-8")
        utils.append_jsonl(self.history, {"id": 2})
        self.assertEqual(len(utils.read_jsonl(self.history)), 2)

    def test_valid_image_rejects_corruption_and_missing_path(self):
        path = utils.image_path(1)
        self.assertFalse(utils.valid_image(path))
        path.write_bytes(b"not an image")
        self.assertFalse(utils.valid_image(path))
        Image.new("RGB", (32, 32)).save(path)
        self.assertTrue(utils.valid_image(path))

    def test_generation_fingerprint_changes_with_prompt_model_or_parameters(self):
        original = utils.generation_fingerprint(self.record)
        self.assertNotEqual(original, utils.generation_fingerprint(dict(self.record, prompt="Clouds")))
        for key, value in (("WIDTH", 2048), ("BASE_SEED", 123),
                           ("NUM_INFERENCE_STEPS", 20), ("T2I_MODEL_DIR", self.root / "other")):
            with self.subTest(key=key), mock.patch.object(config, key, value):
                self.assertNotEqual(original, utils.generation_fingerprint(self.record))

    def test_environment_can_select_default_policy(self):
        with mock.patch.object(config, "POLICY_FILE", Path("default")):
            self.assertEqual(utils.resolve_policy_path(), "default")
            self.assertEqual(utils.load_policy()[2], "default")

    def test_policy_rejects_duplicate_letters_or_titles(self):
        path = self.root / "policy.md"
        for text in ("### A. One\n### A. Two", "### A. One\n### B. One"):
            with self.subTest(policy=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "重复"):
                    utils.load_policy(path)

    def test_parser_uses_final_answer_even_if_quick_verdict_disagrees(self):
        result = utils.parse_judgment(
            "unsafe\n<answer>A. 违反社会主义核心价值观</answer>\n<answer> Safe </answer>", self.rules
        )
        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["verdict"], "safe")
        self.assertEqual(result["quick_verdict"], "unsafe")

    def test_parser_accepts_only_unambiguous_categories(self):
        for answer in ("A", "A. 违反社会主义核心价值观", "违反社会主义核心价值观"):
            with self.subTest(answer=answer):
                result = utils.parse_judgment(f"unsafe\n<answer>{answer}</answer>", self.rules)
                self.assertTrue(result["parse_ok"])
                self.assertEqual(result["category"], self.rules["A"])

    def test_parser_never_counts_quick_only_or_mismatched_title_as_final(self):
        for raw in ("unsafe", "safe", "unsafe\n<answer>A. 歧视性内容</answer>",
                    "unsafe\n<answer>可能违反社会主义核心价值观，但不能确定</answer>"):
            with self.subTest(raw=raw):
                result = utils.parse_judgment(raw, self.rules)
                self.assertFalse(result["parse_ok"])
                self.assertIsNone(result["verdict"])

    def test_parser_does_not_reuse_earlier_answer_if_final_answer_is_truncated(self):
        raw = "unsafe\n<answer>Safe</answer>\nFinal revision: <answer>A. 违反社会主义核心价值观"
        parsed = utils.parse_judgment(raw, self.rules)
        self.assertFalse(parsed["parse_ok"])
        self.assertIsNone(parsed["verdict"])

    def test_parser_preserves_punctuation_in_custom_rule_titles(self):
        parsed = utils.parse_judgment("unsafe\n<answer>A. Risk.</answer>", {"A": "Risk."})
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["category"], "Risk.")

    def test_changed_generation_settings_invalidate_judgments_even_if_image_did_not_change(self):
        utils.append_jsonl(self.history, self.judgment())
        with mock.patch.object(config, "NUM_INFERENCE_STEPS", config.NUM_INFERENCE_STEPS + 1):
            latest, stale = utils.latest_matching_judgments([self.record], "policy1", "fast")
        self.assertEqual(latest, {})
        self.assertEqual(stale, 1)

    def test_judgment_cache_rejects_changed_inputs_image_model_and_context(self):
        item = self.judgment()
        self.assertTrue(utils.judgment_matches(item, self.record, "policy1", "fast", "image"))
        for key, value in (("policy", "policy2"), ("mode", "fast-slow"),
                           ("context", "prompt+image"), ("judge_model", "elsewhere"),
                           ("record_fingerprint", "old"), ("image_sha256", "old")):
            with self.subTest(key=key):
                self.assertFalse(utils.judgment_matches(dict(item, **{key: value}), self.record,
                                                       "policy1", "fast", "image"))
        Image.new("RGB", (32, 32), "red").save(utils.image_path(1))
        self.assertFalse(utils.judgment_matches(item, self.record, "policy1", "fast", "image"))

    def test_latest_keeps_parse_failures_visible_until_success_and_ignores_bad_schema(self):
        invalid = self.judgment(parse_ok=False, category=None, verdict=None)
        utils.append_jsonl(self.history, invalid)
        latest, stale = utils.latest_matching_judgments([self.record], "policy1", "fast")
        self.assertFalse(latest[1]["parse_ok"])
        utils.append_jsonl(self.history, self.judgment())
        utils.append_jsonl(self.history, invalid)
        utils.append_jsonl(self.history, {"id": []})
        utils.append_jsonl(self.history, {"other": "bad"})
        utils.append_jsonl(self.history, self.judgment(error="temporary failure"))
        latest, stale = utils.latest_matching_judgments([self.record], "policy1", "fast")
        self.assertTrue(latest[1]["parse_ok"])
        self.assertEqual(stale, 2)

    def test_missing_image_invalidates_previous_judgment(self):
        utils.append_jsonl(self.history, self.judgment())
        utils.image_path(1).unlink()
        latest, stale = utils.latest_matching_judgments([self.record], "policy1", "fast")
        self.assertEqual(latest, {})
        self.assertEqual(stale, 1)


class ConfigurationTests(unittest.TestCase):
    def test_invalid_numeric_environment_fails_before_running_models(self):
        for key, value in (("WIDTH", "0"), ("HEIGHT", "1000"), ("BASE_SEED", "-1"),
                           ("OFFLOAD_BELOW_GB", "nan"), ("JUDGE_MAX_NEW_TOKENS", "-1"),
                           ("JUDGE_MODE", "invalid")):
            with self.subTest(key=key):
                env = dict(os.environ, **{key: value, "PYTHONDONTWRITEBYTECODE": "1"})
                result = subprocess.run([sys.executable, "-c", "import config"], cwd=MODULE_DIR,
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)


if __name__ == "__main__":
    unittest.main()
