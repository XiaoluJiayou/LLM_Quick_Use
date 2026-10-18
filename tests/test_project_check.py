from pathlib import Path
import tempfile
import unittest

from tools.check_project import check_project, inspect_source


class ProjectCheckTests(unittest.TestCase):
    def test_checker_parses_without_importing_or_executing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "should_not_exist"
            (root / "example.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()", encoding="utf-8")
            result = check_project(root)
            self.assertEqual(result["summary"]["errors"], 0)
            self.assertFalse(marker.exists())

    def test_credential_detection_does_not_echo_value(self):
        token = "hf_" + "a" * 30
        issues, _ = inspect_source("TOKEN = " + repr(token), "example.py")
        self.assertEqual(issues[0]["code"], "credential-literal")
        self.assertNotIn(token, str(issues))

    def test_main_guard_is_not_an_import_side_effect(self):
        issues, _ = inspect_source('if __name__ == "__main__":\n    snapshot_download("x")', "example.py")
        self.assertEqual(issues, [])
        issues, _ = inspect_source('snapshot_download("x")', "example.py")
        self.assertEqual(issues[0]["code"], "import-side-effect")

    def test_invalid_notebook_or_python_is_a_blocking_issue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.py").write_text("def (", encoding="utf-8")
            (root / "bad.ipynb").write_text("[]", encoding="utf-8")
            self.assertEqual(check_project(root)["summary"]["errors"], 2)


if __name__ == "__main__":
    unittest.main()
