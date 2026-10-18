"""只读取源码的全项目检查：不导入实验脚本，不下载或执行模型。"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "outputs", "build", "dist"}
TOKEN_RE = re.compile(r"(?:\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}|\bhf_[A-Za-z0-9]{24,}|\bLTAI[A-Za-z0-9]{16,})")
SENSITIVE_NAME_RE = re.compile(r"(?:api_?key|access_key_secret|secret_key|password|access_token|hf_token)$", re.I)
EXAMPLE_RE = re.compile(r"^(?:|none|null|dummy|test(?:[-_].*)?|example(?:[-_].*)?|your.*|<.*>|\.{3}|\*+)$", re.I)
EFFECT_CALLS = {"snapshot_download", "hf_hub_download", "from_pretrained", "read_csv", "to_csv", "post", "get", "urlopen", "write_text", "write_bytes"}


def _main_guard(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    values = (test.left, *test.comparators)
    return any(isinstance(value, ast.Name) and value.id == "__name__" for value in values) and any(
        isinstance(value, ast.Constant) and value.value == "__main__" for value in values
    )


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _top_level_calls(nodes):
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or _main_guard(node):
            continue
        # 不进入函数体；只查模块执行阶段会运行的表达式。
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                yield child


def inspect_source(source, file, cell=None):
    issues = []
    location = {"file": file, **({"cell": cell} if cell is not None else {})}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            tree = ast.parse(source, filename=file)
        for warning in caught:
            issues.append({**location, "line": warning.lineno, "severity": "warning", "code": "syntax-warning", "message": str(warning.message)})
    except SyntaxError as exc:
        return [{**location, "line": exc.lineno or 1, "severity": "error", "code": "syntax", "message": exc.msg}], []
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and TOKEN_RE.search(node.value):
            issues.append({**location, "line": node.lineno, "severity": "error", "code": "credential-literal", "message": "疑似凭据字面量；值未输出，请改用环境变量"})
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and not EXAMPLE_RE.fullmatch(value.value.strip()):
                if any(isinstance(target, ast.Name) and SENSITIVE_NAME_RE.search(target.id) for target in targets):
                    issues.append({**location, "line": node.lineno, "severity": "warning", "code": "credential-config", "message": "凭据配置使用了字符串常量；请核实是否为占位值，值未输出"})
    if cell is None:
        for call in _top_level_calls(tree.body):
            name = _call_name(call.func)
            if name in EFFECT_CALLS and (name != "get" or isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "requests"):
                issues.append({**location, "line": call.lineno, "severity": "warning", "code": "import-side-effect", "message": f"模块顶层调用 {name}；确认导入不会执行数据或远程任务"})
    return issues, sorted(imports)


def check_project(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"项目目录不存在: {root}")
    inventory, issues = [], []
    sources = []
    for directory, children, names in os.walk(root):
        children[:] = [name for name in children if name not in EXCLUDED and not name.endswith(".egg-info")]
        sources.extend(Path(directory) / name for name in names if Path(name).suffix in {".py", ".ipynb"})
    for path in sorted(sources):
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if not path.is_file() or path.suffix not in {".py", ".ipynb"}:
            continue
        name = relative.as_posix()
        try:
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix == ".py":
                found, imports = inspect_source(text, name)
                inventory.append({"file": name, "kind": "python", "lines": len(text.splitlines()), "imports": imports})
                issues.extend(found)
            else:
                notebook = json.loads(text)
                if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
                    raise ValueError("Notebook 必须包含 cells 数组")
                code_cells = 0
                for index, cell in enumerate(notebook["cells"], 1):
                    if cell.get("cell_type") != "code":
                        continue
                    code_cells += 1
                    source = cell.get("source", [])
                    source = "".join(source) if isinstance(source, list) else source
                    if not isinstance(source, str):
                        raise ValueError("代码单元 source 必须是文本")
                    # IPython 的单行 shell/magic 不是 Python 语法；保留行号以检查其余代码。
                    normalized = "\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines())
                    found, _ = inspect_source(normalized, name, index)
                    issues.extend(found)
                inventory.append({"file": name, "kind": "notebook", "code_cells": code_cells})
        except (ValueError, UnicodeDecodeError, OSError, TypeError, AttributeError) as exc:
            issues.append({"file": name, "line": 1, "severity": "error", "code": "unreadable", "message": f"无法读取有效源码: {type(exc).__name__}"})
    return {"files": inventory, "issues": issues,
            "summary": {"python_files": sum(item["kind"] == "python" for item in inventory),
                        "notebooks": sum(item["kind"] == "notebook" for item in inventory),
                        "errors": sum(item["severity"] == "error" for item in issues),
                        "warnings": sum(item["severity"] == "warning" for item in issues)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json-output", type=Path, help="保存文件清单和脱敏问题清单")
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_project(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    for item in result["issues"]:
        cell = f" cell {item['cell']}" if "cell" in item else ""
        print(f"{item['severity'].upper()} {item['file']}{cell}:{item['line']} [{item['code']}] {item['message']}")
    print(json.dumps(result["summary"], ensure_ascii=True))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(result["summary"]["errors"] > 0 or args.strict_warnings and result["summary"]["warnings"] > 0)


if __name__ == "__main__":
    sys.exit(main())
