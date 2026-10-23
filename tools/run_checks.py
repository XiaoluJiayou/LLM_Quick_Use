"""运行公共回归和独立的文生图评测回归，隔离同名 config/utils 模块。"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    commands = [
        (ROOT, [sys.executable, "-B", str(ROOT / "tools" / "check_project.py")]),
        (ROOT, [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]),
        (ROOT / "text_to_image_generate_safety_evaluate" / "t2i_qwen_judge",
         [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    failed = False
    for cwd, command in commands:
        result = subprocess.run(command, cwd=cwd, env=environment, check=False)
        failed = failed or result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
