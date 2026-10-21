"""裁判流程回归测试；只使用微型本地图片和模拟模型，不下载权重。"""
import builtins
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import judge_singguard as judge_module
import utils


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.judge = judge_module.SingGuardJudge.__new__(judge_module.SingGuardJudge)
        self.judge.image_class = Image
        self.judge.policy = "### A. Risk\nThe complete custom rule."
        self.judge.requested_mode = "fast"

    def test_nested_kwargs_are_tried_when_direct_mode_is_ignored(self):
        calls = []

        def render(messages, kwargs):
            calls.append(kwargs)
            mode = kwargs.get("chat_template_kwargs", {}).get("thinking_type", "fast-slow")
            return self.judge.policy + f"\n<thinking_type>{mode}</thinking_type>"

        self.judge._prompt_text = render
        kwargs, mode = self.judge._pick_template_kwargs()
        self.assertEqual(len(calls), 2)
        self.assertEqual(mode, "fast")
        self.assertEqual(kwargs["chat_template_kwargs"]["thinking_type"], "fast")

    def test_fast_mode_never_silently_falls_back(self):
        self.judge._prompt_text = mock.Mock(
            return_value=self.judge.policy + "\n<thinking_type>fast-slow</thinking_type>"
        )
        with self.assertRaisesRegex(RuntimeError, "不会自动切换模式"):
            self.judge._pick_template_kwargs()
        self.assertEqual(self.judge._prompt_text.call_count, 2)

    def test_matching_heading_without_full_policy_is_rejected(self):
        self.judge._prompt_text = mock.Mock(
            return_value="### A. Risk\n<thinking_type>fast</thinking_type>"
        )
        with self.assertRaises(RuntimeError):
            self.judge._pick_template_kwargs()

    def test_fast_slow_is_explicitly_verified(self):
        self.judge.requested_mode = "fast-slow"
        self.judge._prompt_text = mock.Mock(
            return_value=self.judge.policy + "\n<thinking_type>fast-slow</thinking_type>"
        )
        kwargs, mode = self.judge._pick_template_kwargs()
        self.assertEqual(mode, "fast-slow")
        self.assertEqual(kwargs["thinking_type"], "fast-slow")


class JudgeFlowTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.model_dir = self.root / "judge_model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        self.data_path = self.root / "data.json"
        self.policy_path = self.root / "policy.md"
        self.policy_path.write_text("### A. Test Risk\n  - A test risk.\n### Safe\n  - Safe.", encoding="utf-8")
        self.judgments_path = self.root / "judgments.jsonl"
        self.generations_path = self.root / "generations.jsonl"
        for name, value in {
            "DATA_JSON": self.data_path,
            "IMAGE_DIR": self.image_dir,
            "JUDGE_MODEL_DIR": self.model_dir,
            "POLICY_FILE": self.policy_path,
            "JUDGMENTS_FILE": self.judgments_path,
            "GENERATIONS_FILE": self.generations_path,
            "T2I_MODEL_DIR": self.root / "generation_model",
            "T2I_MODEL_NAME": "TestGenerator",
            "WIDTH": 32,
            "HEIGHT": 32,
            "NUM_INFERENCE_STEPS": 4,
            "JUDGE_MODE": "fast",
            "BASE_SEED": 42,
        }.items():
            self.stack.enter_context(mock.patch.object(config, name, value))
        self.stack.enter_context(mock.patch.object(judge_module, "init_logging"))
        self.stack.enter_context(mock.patch.object(judge_module, "logging"))
        self.stack.enter_context(mock.patch.object(judge_module.traceback, "print_exc"))
        self._write_data(1)
        self._write_image(1)
        self.fake_model = mock.Mock(mode="fast")
        self.fake_model.judge.return_value = "safe\n<answer>Safe</answer>"
        self.fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=mock.Mock())
        )

    def _write_data(self, count):
        payload = {
            str(record_id): {
                "prompt": f"Test prompt {record_id}",
                "label": ["商业违法违规"],
            }
            for record_id in range(1, count + 1)
        }
        self.data_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _write_image(self, record_id):
        path = self.image_dir / f"{record_id}.png"
        Image.new("RGB", (32, 32), "white").save(path)
        record = next(r for r in utils.load_records() if r["id"] == record_id)
        utils.append_jsonl(self.generations_path, {
            "id": record_id,
            "generation_fingerprint": utils.generation_fingerprint(record),
            "image_sha256": utils.file_sha256(path),
        })

    def _run(self, *args):
        with (
            mock.patch.object(sys, "argv", ["judge_singguard.py", *args]),
            mock.patch.dict(sys.modules, {"torch": self.fake_torch}),
            mock.patch.object(judge_module, "SingGuardJudge", return_value=self.fake_model),
        ):
            return judge_module.main()

    def test_dry_run_does_not_import_torch_or_transformers(self):
        real_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torch", "transformers"}:
                raise AssertionError(f"dry-run imported {name}")
            return real_import(name, *args, **kwargs)

        with (
            mock.patch.object(sys, "argv", ["judge_singguard.py", "--dry-run"]),
            mock.patch.object(builtins, "__import__", side_effect=checked_import),
            mock.patch.object(judge_module, "SingGuardJudge") as model_class,
        ):
            self.assertEqual(judge_module.main(), 0)
        model_class.assert_not_called()
        self.assertFalse(self.judgments_path.exists())

    def test_dry_run_allows_missing_model_and_images(self):
        (self.image_dir / "1.png").unlink()
        with mock.patch.object(config, "JUDGE_MODEL_DIR", self.root / "absent_model"):
            self.assertEqual(self._run("--dry-run"), 0)
        self.fake_model.judge.assert_not_called()

    def test_missing_or_corrupt_image_is_failure(self):
        path = self.image_dir / "1.png"
        for content in (None, b"not a png"):
            with self.subTest(content=content):
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
                self.assertEqual(self._run(), 1)
        self.fake_model.judge.assert_not_called()

    def test_partial_missing_images_keep_failure_status(self):
        self._write_data(2)
        self.assertEqual(self._run(), 1)
        self.assertEqual(len(utils.read_jsonl(self.judgments_path)), 1)

    def test_changed_prompt_rejects_previous_generated_image(self):
        content = self.data_path.read_text(encoding="utf-8")
        self.data_path.write_text(
            content.replace("Test prompt 1", "Changed prompt 1"), encoding="utf-8"
        )
        self.assertEqual(self._run("--with-prompt"), 1)
        self.assertEqual(self._run("--dry-run"), 0)
        self.fake_model.judge.assert_not_called()
        self.assertFalse(self.judgments_path.exists())

    def test_changed_steps_reject_previous_generated_image(self):
        with mock.patch.object(config, "NUM_INFERENCE_STEPS", 5):
            self.assertEqual(self._run(), 1)
            self.assertEqual(self._run("--dry-run"), 0)
        self.fake_model.judge.assert_not_called()
        self.assertFalse(self.judgments_path.exists())

    def test_image_without_generation_metadata_is_rejected(self):
        self.generations_path.unlink()
        self.assertEqual(self._run(), 1)
        self.fake_model.judge.assert_not_called()

    def test_parse_failure_is_saved_and_retried(self):
        self.fake_model.judge.side_effect = ["safe without answer", "safe\n<answer>Safe</answer>"]
        self.assertEqual(self._run(), 0)
        first = utils.read_jsonl(self.judgments_path)[0]
        self.assertFalse(first["parse_ok"])
        self.assertIsNone(first["verdict"])
        self.assertEqual(self._run(), 0)
        self.assertEqual(self.fake_model.judge.call_count, 2)
        self.assertEqual(self._run(), 0)
        self.assertEqual(self.fake_model.judge.call_count, 2)

    def test_fast_slow_parse_failure_retries_once_with_more_tokens(self):
        self.fake_model.mode = "fast-slow"
        self.fake_model.max_new_tokens = config.JUDGE_SLOW_MAX_NEW_TOKENS
        self.fake_model.judge.side_effect = [
            "unsafe\nthinking without answer",
            "unsafe\n<answer>A. Test Risk</answer>",
        ]
        with mock.patch.object(config, "JUDGE_MODE", "fast-slow"):
            self.assertEqual(self._run(), 0)
        self.assertEqual(self.fake_model.judge.call_count, 2)
        self.fake_model.judge.assert_called_with(
            self.image_dir / "1.png",
            prompt=None,
            max_new_tokens=config.JUDGE_SLOW_RETRY_MAX_NEW_TOKENS,
        )
        saved = utils.read_jsonl(self.judgments_path)
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0]["parse_ok"])
        self.assertEqual(saved[0]["verdict"], "unsafe")

    def test_model_exception_is_saved_with_nonzero_status(self):
        self.fake_model.judge.side_effect = RuntimeError("simulated inference failure")
        self.assertEqual(self._run(), 1)
        saved = utils.read_jsonl(self.judgments_path)[0]
        self.assertEqual(saved["error"], "simulated inference failure")
        self.assertIn("image_sha256", saved)
        self.fake_torch.cuda.empty_cache.assert_called_once()

    def test_success_stores_metadata_and_prompt_context(self):
        self.assertEqual(self._run("--with-prompt"), 0)
        saved = utils.read_jsonl(self.judgments_path)[0]
        record = utils.load_records()[0]
        self.assertEqual(saved["record_fingerprint"], utils.record_fingerprint(record))
        self.assertEqual(saved["image_sha256"], utils.file_sha256(self.image_dir / "1.png"))
        self.assertEqual(saved["judge_model"], str(self.model_dir.resolve()))
        self.assertEqual(saved["context"], "prompt+image")
        self.assertEqual(saved["mode"], "fast")
        self.assertTrue(saved["parse_ok"])
        self.fake_model.judge.assert_called_once_with(self.image_dir / "1.png", prompt=record["prompt"])


if __name__ == "__main__":
    unittest.main()
