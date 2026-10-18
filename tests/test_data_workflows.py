"""Offline regression tests for data correctness and restart behavior."""
import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import csv_file_merge
from data_generate.data_random_sample import random_sample_csv
from data_generate.generation_utils import parser_simple_json, update_json_file
from en_dataset_to_zh_dataset import convert
import batch_pretrain_data_processing_test as pretrain

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_generate.stratified_sample", ROOT / "data_generate" / "large-scale_data_random_sample.py")
stratified = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stratified)


class TempFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_csv(self, name, rows, fields=None):
        path = self.root / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_csv(self, path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))


class SamplingTests(TempFiles):
    def test_random_sample_reproducible_and_caps_small_input(self):
        source = self.write_csv("input.csv", [{"id": str(i), "question": "q"} for i in range(30)])
        first, second = self.root / "one.csv", self.root / "two.csv"
        random_sample_csv(source, first, 7, seed=3)
        random_sample_csv(source, second, 7, seed=3)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(len({r["id"] for r in self.read_csv(first)}), 7)
        random_sample_csv(source, self.root / "all.csv", 100)
        self.assertEqual(len(self.read_csv(self.root / "all.csv")), 30)

    def test_zero_negative_and_output_protection(self):
        source = self.write_csv("input.csv", [{"question": "q"}])
        output = self.root / "zero.csv"
        random_sample_csv(source, output, 0)
        self.assertEqual(self.read_csv(output), [])
        with self.assertRaises(ValueError):
            random_sample_csv(source, source, 1, overwrite=True)
        with self.assertRaises(ValueError):
            random_sample_csv(source, self.root / "negative.csv", -1)
        with self.assertRaises(FileExistsError):
            random_sample_csv(source, output, 1)

    def test_stratification_includes_empty_categories_and_exact_size(self):
        rows = [{"id": str(i), "一级领域": "A" if i < 2 else ""} for i in range(9)]
        source = self.write_csv("input.csv", rows)
        output = self.root / "sample.csv"
        stratified.stratified_sample_csv(source, output, 5)
        result = self.read_csv(output)
        self.assertEqual(len(result), 5)
        self.assertEqual(sum(row["一级领域"] == "" for row in result), 4)
        self.assertEqual(len({row["id"] for row in result}), 5)

    def test_all_empty_strata_and_header_only(self):
        source = self.write_csv("input.csv", [{"一级领域": "", "id": str(i)} for i in range(4)])
        output = self.root / "sample.csv"
        self.assertEqual(stratified.stratified_sample_csv(source, output, 2), 2)
        empty = self.write_csv("empty.csv", [], fields=["一级领域", "id"])
        self.assertEqual(stratified.stratified_sample_csv(empty, self.root / "empty-out.csv", 20), 0)

    def test_corrupt_csv_never_replaces_existing_output(self):
        source = self.root / "bad.csv"
        source.write_text("a,b\n1,2,3\n", encoding="utf-8")
        output = self.root / "old.csv"
        output.write_text("old", encoding="utf-8")
        with self.assertRaises(ValueError):
            random_sample_csv(source, output, 2, overwrite=True)
        self.assertEqual(output.read_text(), "old")


class MergeTests(TempFiles):
    def test_rerun_excludes_previous_output_and_order_is_stable(self):
        self.write_csv("b.csv", [{"id": "2"}])
        self.write_csv("a.csv", [{"id": "1"}])
        output = self.root / "merged.csv"
        csv_file_merge.merge_multiple_csv(self.root, output)
        csv_file_merge.merge_multiple_csv(self.root, output, overwrite=True)
        self.assertEqual([row["id"] for row in self.read_csv(output)], ["1", "2"])

    def test_schema_mismatch_requires_explicit_union(self):
        self.write_csv("a.csv", [{"a": "1"}])
        self.write_csv("b.csv", [{"b": "2"}])
        output = self.root / "merged.csv"
        with self.assertRaises(ValueError):
            csv_file_merge.merge_multiple_csv(self.root, output)
        csv_file_merge.merge_multiple_csv(self.root, output, allow_schema_union=True)
        self.assertEqual(self.read_csv(output), [{"a": "1", "b": ""}, {"a": "", "b": "2"}])

    def test_failed_merge_preserves_existing_output(self):
        self.write_csv("a.csv", [{"a": "1"}])
        (self.root / "b.csv").write_text("a\n1,2\n", encoding="utf-8")
        output = self.root / "out.csv"
        output.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            csv_file_merge.merge_multiple_csv(self.root, output, overwrite=True)
        self.assertEqual(output.read_text(), "original")


class ConversionTests(TempFiles):
    def setUp(self):
        super().setUp()
        self.input = self.write_csv("input.csv", [{"id": "a", "prompt": "Hello", "category": "none"}, {"id": "b", "prompt": "World", "category": "none"}])
        self.output = self.root / "translated.csv"
        self.cfg = {"input": str(self.input), "output": str(self.output), "format": "csv", "id_field": "id", "text_field": "prompt", "category_field": "category", "_config_dir": self.root}
        self.config_patch = patch.object(convert, "load_dataset_config", side_effect=lambda _: dict(self.cfg))
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.environment = patch.dict(os.environ, {"OPENAI_MODEL": "offline-model", "OPENAI_BASE_URL": "https://example.invalid/v1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client_patch = patch.object(convert, "build_client", return_value=(object(), "offline-model"))
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.model_patch = patch.object(convert, "call_model", side_effect=lambda client, model, system, text, category: {"zh": "译文:" + text, "gb": "无法映射"})
        self.model = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def run_convert(self, **kwargs):
        return convert.convert(self.root / "config.yaml", **kwargs)

    def test_resume_reuses_only_matching_input_and_model(self):
        self.run_convert()
        self.assertEqual(self.model.call_count, 2)
        self.client.reset_mock()
        self.run_convert()
        self.assertEqual(self.model.call_count, 2)
        self.client.assert_not_called()
        with patch.dict(os.environ, {"OPENAI_MODEL": "different-model"}), self.assertRaises(ValueError):
            self.run_convert()

    def test_changed_input_rejects_resume(self):
        self.run_convert()
        before = self.output.read_bytes()
        self.write_csv("input.csv", [{"id": "a", "prompt": "Changed", "category": "none"}])
        with self.assertRaises(ValueError):
            self.run_convert()
        self.assertEqual(self.output.read_bytes(), before)

    def test_duplicate_id_and_missing_prompt_fail_before_model(self):
        self.write_csv("input.csv", [{"id": "x", "prompt": "one", "category": "none"}, {"id": "x", "prompt": "two", "category": "none"}])
        with self.assertRaises(ValueError):
            self.run_convert()
        self.write_csv("input.csv", [{"id": "x", "prompt": "", "category": "none"}])
        with self.assertRaises(ValueError):
            self.run_convert()
        self.model.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_limit_output_cannot_overwrite_full_output(self):
        self.output.write_text("preserve full output", encoding="utf-8")
        result = self.run_convert(limit=1)
        self.assertEqual(self.output.read_text(), "preserve full output")
        self.assertTrue(self.output.with_name("translated.limit-1.csv").exists())
        self.assertEqual(result["written"], 1)

    def test_dry_run_never_calls_model_or_writes(self):
        self.assertEqual(self.run_convert(dry_run=True)["selected_rows"], 2)
        self.client.assert_not_called()
        self.model.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(convert.progress_path(self.output).exists())

    def test_legacy_or_corrupt_progress_is_not_silently_ignored(self):
        path = convert.progress_path(self.output)
        path.write_text('{"id":"a","zh":"old","gb":"无法映射"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_convert()
        path.write_text('{"kind":', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_convert()
        path.write_text('', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_convert()
        self.model.assert_not_called()

    def test_failed_rows_are_logged_and_retried_on_resume(self):
        self.model.side_effect = [RuntimeError("offline failure"), {"zh": "世界", "gb": "无法映射"}]
        self.assertEqual(self.run_convert()["failed"], 1)
        self.model.side_effect = lambda *args: {"zh": "你好", "gb": "无法映射"}
        self.assertEqual(self.run_convert()["written"], 2)
        self.assertEqual(self.model.call_count, 3)

    def test_input_cannot_be_output(self):
        self.cfg["output"] = str(self.input)
        with self.assertRaises(ValueError):
            self.run_convert()
        self.client.assert_not_called()

    def test_json_envelope_metadata_is_preserved(self):
        source = self.root / "input.json"
        source.write_text(json.dumps({"license": "test", "records": [{"id": "a", "prompt": "hello", "category": "none"}]}), encoding="utf-8")
        self.cfg.update(input=str(source), output=str(self.root / "out.json"), format="json", json_key="records")
        self.run_convert()
        result = json.loads((self.root / "out.json").read_text(encoding="utf-8"))
        self.assertEqual(result["license"], "test")
        self.assertEqual(result["records"][0]["中文指令"], "译文:hello")


class ModelParsingTests(unittest.TestCase):
    def response(self, text, finish="stop"):
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)
        client = Mock()
        client.chat.completions.create.return_value = types.SimpleNamespace(choices=[choice])
        return client

    def test_translation_rejects_truncation_missing_and_wrong_types(self):
        for text, finish in [('{"zh":"ok","gb":"无法映射"}', "length"), ('{"zh":null,"gb":"无法映射"}', "stop"), ('{"zh":"ok","gb":"not-a-label"}', "stop")]:
            with self.subTest(text=text, finish=finish), self.assertRaises(RuntimeError):
                convert.call_model(self.response(text, finish), "model", "system", "hello", "")

    def test_valid_wrapped_json_is_parsed(self):
        result = convert.call_model(self.response('result: {"zh":"你好","gb":"B"}'), "model", "system", "hello", "")
        self.assertEqual(result["gb"], "歧视性内容")


class NotebookAndPretrainTests(TempFiles):
    def test_question_parser_handles_bracket_inside_string_and_deduplicates(self):
        self.assertEqual(parser_simple_json('```json\n["a]b", "c", "c"]\n```'), ["a]b", "c"])
        with self.assertRaises(ValueError):
            parser_simple_json('[{"question":"wrong shape"}]')

    def test_question_append_preserves_corrupt_old_data(self):
        path = self.root / "questions.json"
        path.write_text('["truncated', encoding="utf-8")
        with self.assertRaises(ValueError):
            update_json_file(path, ["new"])
        self.assertEqual(path.read_text(), '["truncated')
        path.write_text('["old"]', encoding="utf-8")
        update_json_file(path, ["old", "new"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), ["old", "new"])

    def test_all_notebook_cells_parse_and_outputs_are_cleared(self):
        for path in (ROOT / "data_generate").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for number, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    with self.subTest(path=path.name, cell=number):
                        ast.parse("".join(cell["source"]))
                        self.assertEqual(cell.get("outputs", []), [])

    def test_pretrain_dry_run_and_index_validation(self):
        source = self.root / "corpus.jsonl"
        source.write_text('{"text":"hello"}\n', encoding="utf-8")
        output = self.root / "labels.csv"
        args = ["--input", str(source), "--output", str(output), "--endpoint", "https://example.invalid/v1/chat/completions", "--model", "offline", "--dry-run"]
        with patch.object(pretrain, "run_chat_batch") as run:
            self.assertEqual(pretrain.main(args), 0)
            run.assert_not_called()
        with self.assertRaises(SystemExit):
            pretrain.main(args + ["--start", "5"])
        self.assertFalse(output.exists())

    def test_pretrain_mocked_workflow_preserves_selected_corpus(self):
        source = self.root / "corpus.jsonl"
        source.write_text('{"text":"first"}\n{"text":"second"}\n', encoding="utf-8")
        output = self.root / "labels.csv"
        client = Mock()
        client.public_config.return_value = {"endpoint": "https://example.invalid", "model": "offline"}
        client.complete.return_value = {"status": "ok", "content": "Safe", "error": "", "attempts": 1}
        args = ["--input", str(source), "--output", str(output), "--endpoint", "https://example.invalid/v1/chat/completions", "--model", "offline", "--start", "1"]
        with patch.object(pretrain, "ChatClient", return_value=client):
            self.assertEqual(pretrain.main(args), 0)
            self.assertEqual(pretrain.main(args), 0)
        self.assertEqual(client.complete.call_count, 1)
        rows = self.read_csv(output)
        self.assertEqual(rows[0]["Responses"], "second")
        self.assertEqual(rows[0]["Labels"], "Safe")


if __name__ == "__main__":
    unittest.main()
