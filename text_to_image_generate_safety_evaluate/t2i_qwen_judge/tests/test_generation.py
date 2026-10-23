"""生成阶段的离线回归：真实 JSON/PNG，模拟模型，绝不下载权重。"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
import generate_images
import utils


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.data_path = self.root / "prompts.json"
        settings = {
            "DATA_JSON": self.data_path,
            "OUTPUT_DIR": self.output,
            "IMAGE_DIR": self.output / "images",
            "GENERATIONS_FILE": self.output / "generations.jsonl",
            "T2I_MODEL_DIR": self.root / "model",
            "T2I_MODEL_NAME": "test-Qwen",
            "WIDTH": 32,
            "HEIGHT": 32,
            "NUM_INFERENCE_STEPS": 2,
            "BASE_SEED": 42,
            "OFFLOAD_BELOW_GB": 40,
        }
        for name, value in settings.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.data_path.write_text(
            json.dumps({
                "1": {
                    "prompt": "一朵红花",
                    "label": ["商业违法违规"],
                }
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        self.records = utils.load_records()
        self.record = self.records[0]

    def save_image(self, color="red"):
        path = utils.image_path(self.record["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (config.WIDTH, config.HEIGHT), color).save(path, format="PNG")
        return path

    def save_metadata(self):
        result = {
            "id": self.record["id"],
            "generation_fingerprint": utils.generation_fingerprint(self.record),
            "image_sha256": utils.file_sha256(utils.image_path(self.record["id"])),
        }
        utils.append_jsonl(config.GENERATIONS_FILE, result)
        return result

    @staticmethod
    def fake_torch():
        module = types.ModuleType("torch")
        module.bfloat16 = "bf16"
        module.float16 = "fp16"
        generator = mock.Mock()
        generator.manual_seed.return_value = generator
        module.Generator = mock.Mock(return_value=generator)
        module.cuda = types.SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            is_bf16_supported=mock.Mock(return_value=True),
            empty_cache=mock.Mock(),
            get_device_properties=mock.Mock(return_value=types.SimpleNamespace(total_memory=24 * 1024 ** 3)),
        )
        return module

    def test_matching_complete_png_is_skipped(self):
        self.save_image()
        self.save_metadata()
        self.assertEqual([], generate_images.pending_records(self.records))

    def test_changed_prompt_is_rejected(self):
        self.save_image()
        self.save_metadata()
        changed = dict(self.record, prompt="一朵蓝花")
        with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
            generate_images.pending_records([changed])

    def test_changed_generation_settings_are_rejected(self):
        self.save_image()
        self.save_metadata()
        changes = {
            "WIDTH": 64, "HEIGHT": 64, "NUM_INFERENCE_STEPS": 3,
            "BASE_SEED": 43, "T2I_MODEL_NAME": "another-model",
            "T2I_MODEL_DIR": self.root / "another-model",
        }
        for setting, value in changes.items():
            with self.subTest(setting=setting), mock.patch.object(config, setting, value):
                with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
                    generate_images.pending_records(self.records)

    def test_existing_png_without_metadata_is_rejected(self):
        self.save_image()
        with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
            generate_images.pending_records(self.records)

    def test_legacy_metadata_without_fingerprint_is_rejected(self):
        self.save_image()
        utils.append_jsonl(config.GENERATIONS_FILE, {"id": 1, "image": "1.png"})
        with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
            generate_images.pending_records(self.records)

    def test_metadata_without_image_hash_is_rejected(self):
        self.save_image()
        utils.append_jsonl(config.GENERATIONS_FILE, {
            "id": 1, "generation_fingerprint": utils.generation_fingerprint(self.record),
        })
        with self.assertRaisesRegex(ValueError, "内容哈希"):
            generate_images.pending_records(self.records)

    def test_truncated_png_is_retried(self):
        path = self.save_image()
        self.save_metadata()
        path.write_bytes(path.read_bytes()[:45])
        self.assertFalse(utils.valid_image(path))
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.records, generate_images.pending_records(self.records))

    def test_changed_image_content_is_retried(self):
        self.save_image()
        original = self.save_metadata()
        changed = self.save_image("blue")
        self.assertTrue(utils.valid_image(changed))
        self.assertNotEqual(original["image_sha256"], utils.file_sha256(changed))
        with self.assertLogs(level="WARNING"):
            self.assertEqual(self.records, generate_images.pending_records(self.records))

    def test_success_is_not_hidden_by_later_failure_or_invalid_id(self):
        self.save_image()
        self.save_metadata()
        utils.append_jsonl(config.GENERATIONS_FILE, {"id": 1, "error": "interrupted"})
        utils.append_jsonl(config.GENERATIONS_FILE, {"id": [1]})
        self.assertEqual([], generate_images.pending_records(self.records))

    def test_missing_image_is_retried(self):
        path = self.save_image()
        self.save_metadata()
        path.unlink()
        self.assertEqual(self.records, generate_images.pending_records(self.records))

    def test_generated_metadata_records_real_hash_and_fingerprint(self):
        config.IMAGE_DIR.mkdir(parents=True)
        torch = self.fake_torch()
        pipe = mock.Mock(return_value=types.SimpleNamespace(images=[Image.new("RGB", (32, 32), "red")]))
        with mock.patch.dict(sys.modules, {"torch": torch}):
            result = generate_images.generate_one(pipe, self.record)
        path = utils.image_path(self.record["id"])
        self.assertTrue(utils.valid_image(path))
        self.assertEqual(utils.file_sha256(path), result["image_sha256"])
        self.assertEqual(utils.generation_fingerprint(self.record), result["generation_fingerprint"])
        self.assertEqual(43, result["seed"])
        self.assertEqual([result], utils.read_jsonl(config.GENERATIONS_FILE))
        torch.Generator.assert_called_once_with("cpu")
        torch.Generator.return_value.manual_seed.assert_called_once_with(43)
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_metadata_is_written_before_png_becomes_visible(self):
        config.IMAGE_DIR.mkdir(parents=True)
        path = utils.image_path(self.record["id"])
        torch = self.fake_torch()
        pipe = mock.Mock(return_value=types.SimpleNamespace(images=[Image.new("RGB", (32, 32), "red")]))
        original_append = utils.append_jsonl

        def check_then_append(filename, result):
            self.assertFalse(path.exists())
            temporary = path.with_name(path.name + ".tmp")
            self.assertTrue(utils.valid_image(temporary))
            self.assertEqual(utils.file_sha256(temporary), result["image_sha256"])
            original_append(filename, result)

        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.object(generate_images, "append_jsonl", side_effect=check_then_append):
            generate_images.generate_one(pipe, self.record)
        self.assertEqual([], generate_images.pending_records(self.records))

    def test_interruption_after_metadata_before_rename_is_recoverable(self):
        config.IMAGE_DIR.mkdir(parents=True)
        path = utils.image_path(self.record["id"])
        torch = self.fake_torch()
        pipe = mock.Mock(return_value=types.SimpleNamespace(images=[Image.new("RGB", (32, 32), "red")]))
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.object(Path, "replace", side_effect=OSError("interrupted before rename")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                generate_images.generate_one(pipe, self.record)
        self.assertFalse(path.exists())
        self.assertEqual(1, len(utils.read_jsonl(config.GENERATIONS_FILE)))
        self.assertEqual(self.records, generate_images.pending_records(self.records))
        with mock.patch.dict(sys.modules, {"torch": torch}):
            generate_images.generate_one(pipe, self.record)
        self.assertTrue(utils.valid_image(path))
        self.assertEqual([], generate_images.pending_records(self.records))

    def test_failure_writing_metadata_does_not_publish_png(self):
        config.IMAGE_DIR.mkdir(parents=True)
        torch = self.fake_torch()
        pipe = mock.Mock(return_value=types.SimpleNamespace(images=[Image.new("RGB", (32, 32), "red")]))
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.object(generate_images, "append_jsonl", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                generate_images.generate_one(pipe, self.record)
        self.assertFalse(utils.image_path(self.record["id"]).exists())
        self.assertEqual(self.records, generate_images.pending_records(self.records))

    def test_dry_run_needs_no_model_imports_or_output_directory(self):
        code = """
import sys
class RejectModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'diffusers', 'transformers'):
            raise AssertionError('dry-run attempted model import: ' + fullname)
sys.meta_path.insert(0, RejectModels())
sys.path.insert(0, sys.argv[1])
sys.argv = ['generate_images.py', '--dry-run', '--per-type', '1']
import generate_images
raise SystemExit(generate_images.main())
"""
        environment = os.environ.copy()
        environment.update({
            "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "DATA_JSON": str(self.data_path), "OUTPUT_DIR": str(self.output),
            "T2I_MODEL_DIR": str(self.root / "missing-model"),
            "T2I_MODEL_NAME": "test-Qwen", "WIDTH": "32", "HEIGHT": "32",
            "NUM_INFERENCE_STEPS": "2", "BASE_SEED": "42", "OFFLOAD_BELOW_GB": "40",
            "JUDGE_MODE": "fast", "JUDGE_MAX_NEW_TOKENS": "128", "JUDGE_SLOW_MAX_NEW_TOKENS": "1024",
        })
        result = subprocess.run(
            [sys.executable, "-c", code, str(PROJECT_DIR)], env=environment,
            cwd=str(self.root), capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("待生成 1 张", result.stdout)
        self.assertFalse(self.output.exists())

    def test_single_sample_failure_returns_nonzero_and_records_error(self):
        torch = self.fake_torch()
        with mock.patch.object(sys, "argv", ["generate_images.py"]), \
                mock.patch.object(generate_images, "load_pipeline", return_value=mock.Mock()), \
                mock.patch.object(generate_images, "generate_one", side_effect=RuntimeError("simulated failure")), \
                mock.patch.dict(sys.modules, {"torch": torch}), \
                redirect_stderr(io.StringIO()), self.assertLogs(level="ERROR"):
            status = generate_images.main()
        self.assertEqual(1, status)
        results = utils.read_jsonl(config.GENERATIONS_FILE)
        self.assertEqual(1, len(results))
        self.assertEqual("simulated failure", results[0]["error"])
        self.assertEqual(utils.generation_fingerprint(self.record), results[0]["generation_fingerprint"])
        torch.cuda.empty_cache.assert_called_once()

    def test_model_path_is_checked_before_loading_dependencies(self):
        with self.assertRaisesRegex(FileNotFoundError, "T2I_MODEL_DIR"):
            generate_images.load_pipeline()

    def test_pipeline_loads_only_local_weights_and_offloads_small_gpu(self):
        config.T2I_MODEL_DIR.mkdir()
        (config.T2I_MODEL_DIR / "model_index.json").write_text(
            json.dumps({"_class_name": "QwenImage21Pipeline"}), encoding="utf-8"
        )
        torch = self.fake_torch()
        diffusers = types.ModuleType("diffusers")
        pipe = mock.Mock(dtype="bf16")
        factory = mock.Mock()
        factory.from_pretrained.return_value = pipe
        diffusers.QwenImage21Pipeline = factory
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}), \
                mock.patch.object(sys, "version_info", (3, 12)):
            self.assertIs(pipe, generate_images.load_pipeline())
        factory.from_pretrained.assert_called_once_with(
            str(config.T2I_MODEL_DIR), torch_dtype="bf16", local_files_only=True,
        )
        pipe.enable_model_cpu_offload.assert_called_once_with()
        pipe.to.assert_not_called()

    def test_pipeline_uses_float16_when_bfloat16_is_unsupported(self):
        config.T2I_MODEL_DIR.mkdir()
        (config.T2I_MODEL_DIR / "model_index.json").write_text(
            json.dumps({"_class_name": "QwenImage21Pipeline"}), encoding="utf-8"
        )
        torch = self.fake_torch()
        torch.cuda.is_bf16_supported.return_value = False
        diffusers = types.ModuleType("diffusers")
        factory = mock.Mock()
        diffusers.QwenImage21Pipeline = factory
        with mock.patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}), \
                mock.patch.object(sys, "version_info", (3, 12)):
            generate_images.load_pipeline()
        self.assertEqual("fp16", factory.from_pretrained.call_args.kwargs["torch_dtype"])


if __name__ == "__main__":
    unittest.main()
