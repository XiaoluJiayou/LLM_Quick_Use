"""提取 Notebook 的小段逻辑，用临时图片和模拟模型验证，不运行权重单元。"""
import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
from PIL import Image

from llm_quick_use.io import atomic_write_text
from llm_quick_use.parsing import extract_json
from llm_quick_use.records import image_prompt_records

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "text_to_image_generate_safety_evaluate"


def find_cell(filename, text):
    notebook = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and text in source:
            return source
    raise AssertionError(f"Missing code cell: {filename}: {text}")


def execute_without_imports(source, scope):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "<notebook-logic>", "exec"), scope)


class NotebookTests(unittest.TestCase):
    def test_janus_loop_uses_current_row_and_stable_ids(self):
        source = find_cell("batch_text_to_img_generate.ipynb", "for index in trange")
        generate = mock.Mock()
        processor = SimpleNamespace(
            apply_sft_template_for_multi_turn_prompts=mock.Mock(return_value="encoded"),
            sft_format="format", image_start_tag="<image>",
        )
        scope = {"data": pd.DataFrame({"ID": [5, 8], "提示词": ["sky", "moon"]}),
                 "records": [{"id": 5, "prompt": "sky"}, {"id": 8, "prompt": "moon"}],
                 "trange": range, "torch": SimpleNamespace(manual_seed=mock.Mock()),
                 "generate": generate, "vl_gpt": object(), "vl_chat_processor": processor, "os": os}
        execute_without_imports(source, scope)
        self.assertEqual([call.args[-1] for call in generate.call_args_list], [5, 8])
        prompts = [call.kwargs["conversations"][0]["content"] for call in processor.apply_sft_template_for_multi_turn_prompts.call_args_list]
        self.assertEqual(prompts, ["sky", "moon"])

    def test_unit2ixl_handles_small_dataset_and_creates_output_directory(self):
        source = find_cell("img_generate_test.ipynb", "base_path = Path")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "images"
            pipeline = mock.Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (8, 8))]))
            scope = {"Path": Path, "os": os, "pd": pd, "pipeline": pipeline,
                     "records": [{"id": 2, "prompt": "sky"}, {"id": 9, "prompt": "moon"}]}
            with mock.patch.dict(os.environ, {"OUTPUT_DIR": str(output)}):
                execute_without_imports(source, scope)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["2.png", "9.png"])
            self.assertEqual(pipeline.call_count, 2)

    def test_canonical_ids_reject_leading_zero_duplicates_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "ID 重复"):
            image_prompt_records([{"ID": "1", "提示词": "sky"}, {"ID": "01", "提示词": "moon"}])

    def test_all_prompts_and_seed_ranges_are_validated_before_model_loading(self):
        for rows in ([{"ID": 1, "提示词": "  "}], [{"ID": 2 ** 64, "提示词": "sky"}],
                     [{"ID": True, "提示词": "sky"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                image_prompt_records(rows, base_seed=42)
        self.assertEqual(image_prompt_records([{"提示词": " sky "}]), [{"id": 0, "prompt": "sky"}])

    def test_llava_parser_returns_unknown_for_bad_schema_or_truncation(self):
        source = find_cell("img_evaluate_test.ipynb", "def extract_assistant_json")
        scope = {"extract_json": extract_json}
        execute_without_imports(source, scope)
        parse = scope["extract_assistant_json"]
        self.assertEqual(parse('{"rating":"Safe","category":"NA","rationale":"a } in text"}')["rating"], "Safe")
        self.assertIsNone(parse('{"rating":"Safe"}'))
        self.assertIsNone(parse('unsafe without JSON'))
        self.assertIsNone(parse('{"rating":"Safe","nested":{"rating":"Safe"}'))

    def test_llava_decodes_only_new_tokens_and_records_corrupt_images(self):
        source = find_cell("img_evaluate_test.ipynb", "image_paths = sorted")
        parser_scope = {"extract_json": extract_json}
        execute_without_imports(find_cell("img_evaluate_test.ipynb", "def extract_assistant_json"), parser_scope)

        class Input:
            shape = (1, 8)

            def to(self, device):
                return self

        class Generated:
            def __getitem__(self, key):
                self.slice_start = key[1].start
                return "new-tokens"

        generated = Generated()
        model = SimpleNamespace(device="cpu", generate=mock.Mock(return_value=generated))
        processor = mock.Mock(return_value={"input_ids": Input()})
        processor.batch_decode.return_value = ['{"rating":"Safe","category":"NA","rationale":"blue sky"}']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (8, 8)).save(images / "1.png")
            (images / "2.png").write_bytes(b"corrupt")
            output = root / "results.csv"
            scope = {"Path": Path, "os": os, "Image": Image, "pd": pd,
                     "torch": SimpleNamespace(inference_mode=nullcontext), "tqdm": lambda value: value,
                     "model": model, "processor": processor, "text_prompt": "policy", "hyperparameters": {},
                     "extract_assistant_json": parser_scope["extract_assistant_json"], "atomic_write_text": atomic_write_text}
            with mock.patch.dict(os.environ, {"IMAGE_INPUT_DIR": str(images), "IMAGE_EVAL_OUTPUT": str(output)}):
                execute_without_imports(source, scope)
            rows = pd.read_csv(output).fillna("")
            self.assertEqual(rows["status"].tolist(), ["ok", "error"])
            self.assertEqual(rows["img_rating"].tolist(), ["Safe", ""])
            self.assertEqual(generated.slice_start, 8)
            self.assertEqual(processor.batch_decode.call_args.args[0], "new-tokens")


if __name__ == "__main__":
    unittest.main()
