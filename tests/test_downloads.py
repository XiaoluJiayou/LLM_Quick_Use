"""Downloads are verified through mocks only; no external requests."""
import contextlib
import importlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from model_dataset_download._download_cli import run_huggingface, run_modelscope


class DownloadTests(unittest.TestCase):
    def test_imports_never_download_or_print(self):
        hf = types.SimpleNamespace(snapshot_download=Mock(), hf_hub_download=Mock())
        ms = types.SimpleNamespace(snapshot_download=Mock())
        names = ("dataset_download_huggingface", "model_download_huggingface", "model_download_huggingface_standard",
                 "huggingface_standalone_file_download", "model_download_modelscope", "function_111")
        output = io.StringIO()
        with patch.dict(sys.modules, {"huggingface_hub": hf, "modelscope": ms}), contextlib.redirect_stdout(output):
            for name in names:
                importlib.reload(importlib.import_module("model_dataset_download." + name))
        hf.snapshot_download.assert_not_called()
        hf.hf_hub_download.assert_not_called()
        ms.snapshot_download.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_hf_dry_run_needs_no_optional_dependency(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {"huggingface_hub": None}), contextlib.redirect_stdout(output):
            plan = run_huggingface("org/model", argv=["--dry-run"])
        self.assertNotIn("token", plan)
        self.assertNotIn("endpoint", plan)
        self.assertEqual(plan["repo_type"], "model")

    def test_hf_default_local_dir_can_be_overridden(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}), contextlib.redirect_stdout(io.StringIO()):
            default_plan = run_huggingface(
                "zhangrjjj/Falcon",
                repo_type="dataset",
                default_local_dir="downloads/zhangrjjj--Falcon",
                argv=["--dry-run"],
            )
            override_plan = run_huggingface(
                "zhangrjjj/Falcon",
                repo_type="dataset",
                default_local_dir="downloads/zhangrjjj--Falcon",
                argv=["--dry-run", "--local-dir", "D:/datasets/Falcon"],
            )
        self.assertEqual(Path(default_plan["local_dir"]), Path("downloads/zhangrjjj--Falcon"))
        self.assertEqual(Path(override_plan["local_dir"]), Path("D:/datasets/Falcon"))

    def test_hf_default_endpoint_can_be_overridden(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}), contextlib.redirect_stdout(io.StringIO()):
            default_plan = run_huggingface("zhangrjjj/Falcon", repo_type="dataset", argv=["--dry-run"])
            override_plan = run_huggingface(
                "zhangrjjj/Falcon",
                repo_type="dataset",
                default_endpoint="https://hf-mirror.com",
                argv=["--dry-run", "--endpoint", "https://huggingface.co"],
            )
        self.assertNotIn("endpoint", default_plan)
        self.assertEqual(override_plan["endpoint"], "https://huggingface.co")

    def test_hf_mirror_metadata_error_gets_actionable_message(self):
        download = Mock(side_effect=RuntimeError("Distant resource does not seem to be on huggingface.co. firewall"))
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
                patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "", "HF_ENDPOINT": ""}, clear=False), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                run_huggingface(
                    "zhangrjjj/Falcon",
                    repo_type="dataset",
                    argv=["--endpoint", "https://hf-mirror.com", "--local-dir", "my-data"],
                )
            self.assertIn("去掉 --endpoint", str(ctx.exception))
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_XET"), "1")
            self.assertEqual(os.environ.get("HF_ENDPOINT"), "https://hf-mirror.com")

    def test_hf_parameters_and_environment_token(self):
        download = Mock(return_value="local/result")
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
                patch.dict(os.environ, {"TEST_DOWNLOAD_TOKEN": "offline-placeholder"}), contextlib.redirect_stdout(io.StringIO()):
            run_huggingface("a/b", repo_type="dataset", argv=["--local-dir", "my-data", "--revision", "v2", "--token-env", "TEST_DOWNLOAD_TOKEN", "--endpoint", "https://example.invalid"])
        options = download.call_args.kwargs
        self.assertEqual(options["token"], "offline-placeholder")
        self.assertEqual(options["revision"], "v2")
        self.assertEqual(options["repo_type"], "dataset")
        self.assertNotIn("resume_download", options)
        self.assertNotIn("local_dir_use_symlinks", options)

    def test_hf_file_uses_file_download(self):
        download = Mock(return_value="local/file")
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=download)}), contextlib.redirect_stdout(io.StringIO()):
            run_huggingface("a/b", filename="old.bin", argv=["--filename", "weights.bin"])
        self.assertEqual(download.call_args.kwargs["filename"], "weights.bin")

    def test_modelscope_dry_run_and_execution(self):
        with patch.dict(sys.modules, {"modelscope": None}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_modelscope("a/b", ["--dry-run"])["model_id"], "a/b")
        download = Mock(return_value="local/model")
        with patch.dict(sys.modules, {"modelscope": types.SimpleNamespace(snapshot_download=download)}), contextlib.redirect_stdout(io.StringIO()):
            run_modelscope("a/b", ["--revision", "test-version"])
        self.assertEqual(download.call_args.kwargs["revision"], "test-version")


if __name__ == "__main__":
    unittest.main()
