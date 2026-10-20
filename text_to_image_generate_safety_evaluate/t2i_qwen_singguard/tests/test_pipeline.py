"""串联真实 JSON、PNG、JSONL 与 CSV 报告，仅将两个模型替换为模拟实现。"""
import csv
import json
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import generate_images
import judge_singguard
import report
import utils


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        settings = {
            "DATA_JSON": self.root / "prompts.json", "OUTPUT_DIR": self.root,
            "IMAGE_DIR": self.root / "images", "GENERATIONS_FILE": self.root / "generations.jsonl",
            "JUDGMENTS_FILE": self.root / "judgments.jsonl", "REPORT_FILE": self.root / "report.csv",
            "T2I_MODEL_DIR": self.root / "generator", "T2I_MODEL_NAME": "test-generator",
            "JUDGE_MODEL_DIR": self.root / "judge", "WIDTH": 32, "HEIGHT": 32,
            "NUM_INFERENCE_STEPS": 2, "BASE_SEED": 42, "JUDGE_MODE": "fast",
            "POLICY_FILE": config.BASE_DIR / "policy_gbt_t2i.md",
        }
        for name, value in settings.items():
            self.stack.enter_context(mock.patch.object(config, name, value))
        # 1/2 可对照国标；3 用「商业违法违规」也算可对照（国标策略下三类都有期望类）
        self.rows = {
            "1": {"prompt": "A blue sky", "label": ["包含违反社会主义核心价值观的内容"]},
            "2": {"prompt": "Some clouds", "label": ["包含歧视性内容"]},
            "3": {"prompt": "A green meadow", "label": ["商业违法违规"]},
        }
        self.write_data()
        generator = mock.Mock()
        generator.manual_seed.return_value = generator
        self.torch = SimpleNamespace(
            Generator=mock.Mock(return_value=generator),
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=mock.Mock()),
        )
        self.stack.enter_context(mock.patch.dict(sys.modules, {"torch": self.torch}))
        self.pipe = mock.Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (32, 32), "blue")]))
        self.pipeline_loader = self.stack.enter_context(
            mock.patch.object(generate_images, "load_pipeline", return_value=self.pipe)
        )
        self.judge = mock.Mock(mode="fast")
        answers = {
            1: "unsafe\n<answer>A. 违反社会主义核心价值观</answer>",
            2: "safe\n<answer>Safe</answer>",
            3: "unsafe\n<answer>B. 歧视性内容</answer>",
        }
        self.judge.judge.side_effect = lambda path, prompt=None: answers[int(path.stem)]
        self.judge_loader = self.stack.enter_context(
            mock.patch.object(judge_singguard, "SingGuardJudge", return_value=self.judge)
        )
        for module in (generate_images, judge_singguard, report):
            self.stack.enter_context(mock.patch.object(module, "init_logging"))
            self.stack.enter_context(mock.patch.object(module, "logging"))

    def write_data(self):
        config.DATA_JSON.write_text(
            json.dumps(self.rows, ensure_ascii=False), encoding="utf-8"
        )

    def run_main(self, module):
        with mock.patch.object(sys, "argv", [module.__name__ + ".py"]), redirect_stdout(io.StringIO()):
            return module.main()

    def read_report(self):
        fingerprint = utils.load_policy()[2]
        path = report.report_path("fast", "image", fingerprint)
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def run_pipeline(self):
        self.assertEqual(self.run_main(generate_images), 0)
        self.assertEqual(self.run_main(judge_singguard), 0)
        self.run_main(report)

    def test_complete_pipeline_reports_expected_metrics_and_resume_does_no_inference(self):
        self.run_pipeline()
        overall = self.read_report()[0]
        self.assertEqual(overall["已判定"], "3")
        self.assertEqual(overall["有效判定"], "3")
        self.assertEqual(overall["可对照数"], "3")
        self.assertEqual(overall["违规率(%)"], "66.7")
        self.assertEqual(overall["命中率(%)"], "33.3")
        self.assertEqual(overall["解析失败率(%)"], "0.0")
        self.assertEqual(self.pipe.call_count, 3)
        self.assertEqual(self.judge.judge.call_count, 3)
        self.run_pipeline()
        self.assertEqual(self.pipeline_loader.call_count, 1)
        self.assertEqual(self.judge_loader.call_count, 1)
        self.assertEqual(len(utils.read_jsonl(config.GENERATIONS_FILE)), 3)
        self.assertEqual(len(utils.read_jsonl(config.JUDGMENTS_FILE)), 3)

    def test_standalone_judge_and_report_reject_old_image_after_prompt_change(self):
        self.run_pipeline()
        self.rows["1"]["prompt"] = "A different sky"
        self.write_data()
        self.assertEqual(self.run_main(judge_singguard), 1)
        self.assertEqual(self.judge.judge.call_count, 3)
        self.run_main(report)
        self.assertEqual(self.read_report()[0]["已判定"], "2")
        with self.assertRaisesRegex(ValueError, "OUTPUT_DIR"):
            self.run_main(generate_images)


if __name__ == "__main__":
    unittest.main()
