"""公共文件恢复与 JSON 解析回归，不请求外部服务。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from llm_quick_use.io import (
    append_jsonl, atomic_write_json, atomic_write_text, load_json_array,
    read_jsonl, stable_fingerprint,
)
from llm_quick_use.parsing import extract_json


class CommonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_jsonl_recovers_truncated_utf8_and_separates_next_record(self):
        path = self.root / "results.jsonl"
        path.write_bytes(b'{"id":1}\n{"text":"\xe4\xb8')
        append_jsonl(path, {"id": 2})
        self.assertEqual(read_jsonl(path), [{"id": 1}, {"id": 2}])

    def test_strict_jsonl_rejects_broken_or_nonobject_records(self):
        path = self.root / "results.jsonl"
        for content in ('[]\n', '{"id":\n'):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_jsonl(path, strict=True)

    def test_invalid_existing_json_is_preserved(self):
        path = self.root / "results.json"
        path.write_text("[{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_json_array(path)
        self.assertEqual(path.read_text(encoding="utf-8"), "[{broken")

    def test_atomic_replacement_failure_keeps_original_and_cleans_temp_file(self):
        path = self.root / "results.json"
        atomic_write_json(path, [{"id": 1}])
        with mock.patch("llm_quick_use.io.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                atomic_write_json(path, [{"id": 2}])
        self.assertEqual(load_json_array(path), [{"id": 1}])
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_atomic_csv_text_can_preserve_utf8_bom(self):
        path = self.root / "results.csv"
        atomic_write_text(path, "id,提示词\n1,blue sky\n", encoding="utf-8-sig")
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_fingerprint_ignores_mapping_order_but_detects_config_change(self):
        self.assertEqual(stable_fingerprint({"a": 1, "b": 2}), stable_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(stable_fingerprint({"a": 1}), stable_fingerprint({"a": 2}))


class JsonParsingTests(unittest.TestCase):
    def test_nested_arrays_and_brackets_in_strings_are_not_truncated(self):
        payload = [{"text": "contains ] and }", "nested": [1, {"a": 2}]}]
        self.assertEqual(extract_json("```json\n" + json.dumps(payload) + "\n```", list), payload)

    def test_truncated_outer_container_does_not_return_valid_inner_array(self):
        with self.assertRaises(ValueError):
            extract_json('[{"nested":[1,2]},', list)

    def test_escaped_quotes_and_fences_work(self):
        payload = {"text": 'a quote " and slash \\', "rating": "Safe"}
        self.assertEqual(extract_json("assistant\n" + json.dumps(payload) + "\nend", dict), payload)

    def test_invalid_prose_brackets_do_not_hide_later_json(self):
        self.assertEqual(extract_json('[note] result: {"ok": true}', dict), {"ok": True})

    def test_wrong_shape_and_bad_json_are_errors(self):
        for text in ('{"a":1}', 'no json', '[{"a":1}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_json(text, list)


if __name__ == "__main__":
    unittest.main()
