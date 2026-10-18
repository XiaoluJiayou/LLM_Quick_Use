"""配置导入不依赖模型目录权限；真实加载仍严格检查模型。"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelDirectoryConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models"

    def load_config(self):
        # 独立执行模块，不覆盖其他测试共用的 config，也不读取机器上的配置。
        with mock.patch.dict(os.environ, {"MODEL_ROOT": str(self.models)}, clear=True):
            return load_module("jade_config_regression", "config.py")

    def test_import_tolerates_permission_error_for_flat_or_nested_marker(self):
        model = self.models / "Qwen-Image-2.1"
        for marker in (model / "model_index.json",
                       model / "snapshots" / "master" / "model_index.json"):
            with self.subTest(marker=marker):
                def probe(path):
                    if path == marker:
                        raise PermissionError("model directory is not accessible")
                    return False

                with mock.patch.object(Path, "is_file", autospec=True, side_effect=probe):
                    config = self.load_config()
                self.assertEqual(config.T2I_MODEL_DIR, model.resolve())
                self.assertEqual(config.JUDGE_MODEL_DIR, (self.models / "Sing-Guard-8b").resolve())

    def test_import_tolerates_permission_error_while_resolving_model_paths(self):
        original_resolve = Path.resolve
        expected = {self.models / "Qwen-Image-2.1", self.models / "Sing-Guard-8b"}

        def resolve(path, *args, **kwargs):
            if path in expected:
                raise PermissionError("cannot resolve inaccessible model directory")
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve):
            config = self.load_config()
        self.assertEqual(config.T2I_MODEL_DIR, (self.models / "Qwen-Image-2.1").absolute())
        self.assertEqual(config.JUDGE_MODEL_DIR, (self.models / "Sing-Guard-8b").absolute())

    def test_flat_layout_takes_priority_over_snapshot_layout(self):
        for name, marker in (("Qwen-Image-2.1", "model_index.json"), ("Sing-Guard-8b", "config.json")):
            model = self.models / name
            nested = model / "snapshots" / "master"
            nested.mkdir(parents=True)
            (model / marker).write_text("{}", encoding="utf-8")
            (nested / marker).write_text("{}", encoding="utf-8")
        config = self.load_config()
        self.assertEqual(config.T2I_MODEL_DIR, (self.models / "Qwen-Image-2.1").resolve())
        self.assertEqual(config.JUDGE_MODEL_DIR, (self.models / "Sing-Guard-8b").resolve())

    def test_modelscope_snapshot_layout_is_detected(self):
        for name, marker in (("Qwen-Image-2.1", "model_index.json"), ("Sing-Guard-8b", "config.json")):
            nested = self.models / name / "snapshots" / "master"
            nested.mkdir(parents=True)
            (nested / marker).write_text("{}", encoding="utf-8")
        config = self.load_config()
        self.assertEqual(config.T2I_MODEL_DIR, (self.models / "Qwen-Image-2.1" / "snapshots" / "master").resolve())
        self.assertEqual(config.JUDGE_MODEL_DIR, (self.models / "Sing-Guard-8b" / "snapshots" / "master").resolve())
        self.assertEqual(config.T2I_MODEL_NAME, "Qwen-Image-2.1")

    def test_import_does_not_create_missing_model_directories(self):
        config = self.load_config()
        self.assertFalse(self.models.exists())
        self.assertEqual(config.T2I_MODEL_DIR, (self.models / "Qwen-Image-2.1").resolve())
        self.assertEqual(config.JUDGE_MODEL_DIR, (self.models / "Sing-Guard-8b").resolve())

    def test_non_permission_filesystem_errors_are_not_silenced(self):
        with mock.patch.object(Path, "is_file", side_effect=OSError("device failure")):
            with self.assertRaisesRegex(OSError, "device failure"):
                self.load_config()

        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == self.models / "Qwen-Image-2.1":
                raise OSError("resolve failure")
            return original_resolve(path, *args, **kwargs)

        with mock.patch.object(Path, "resolve", autospec=True, side_effect=resolve):
            with self.assertRaisesRegex(OSError, "resolve failure"):
                self.load_config()

    def test_real_pipeline_load_still_rejects_inaccessible_model(self):
        config = self.load_config()
        # 仅隔离导入所需的辅助符号；执行真实 load_pipeline，不加载任何模型依赖。
        utils = types.ModuleType("utils")
        for name in ("append_jsonl", "file_sha256", "generation_fingerprint", "image_path",
                     "load_records", "positive_int", "read_jsonl", "valid_image"):
            setattr(utils, name, mock.Mock())
        with mock.patch.dict(sys.modules, {"config": config, "utils": utils,
                                           "torch": None, "diffusers": None}):
            generation = load_module("jade_generation_config_regression", "generate_images.py")
            with mock.patch.object(Path, "is_dir", return_value=True):
                with mock.patch.object(Path, "is_file", side_effect=PermissionError("model is not accessible")):
                    with self.assertRaisesRegex(PermissionError, "model is not accessible"):
                        generation.load_pipeline()


if __name__ == "__main__":
    unittest.main()
