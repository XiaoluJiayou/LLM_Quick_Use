"""Offline contracts for video IO, verdicts, checkpoints and vendor preflight."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "Text_and_Image_to_video_evaluate"
VENDORS = VIDEO / "TI2V-models-generate-scripts"
sys.path.insert(0, str(VIDEO))
sys.path.insert(0, str(VENDORS))

import _video_workflow as workflow
import batch_videos_frame_extract as extraction
import image_base94_encode as media
from _vendor_runtime import RunTracker, load_prompts, validate_run


def load_vendor(filename):
    spec = importlib.util.spec_from_file_location("offline_" + filename.replace(".", "_"), VENDORS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_wan_runtime(generate=None):
    video = types.SimpleNamespace()
    class FakeVideo:
        def __getitem__(self, item):
            return video
    pipeline = types.SimpleNamespace(generate=generate or (lambda *_args, **_kwargs: FakeVideo()))
    cfg = types.SimpleNamespace(frame_num=5, sample_steps=1, sample_shift=5,
                                sample_guide_scale=5, sample_fps=24)
    return {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)),
            "dist": types.SimpleNamespace(is_initialized=lambda: False),
            "wan": types.SimpleNamespace(WanT2V=lambda **_: pipeline, WanTI2V=lambda **_: pipeline),
            "WAN_CONFIGS": {"t2v-A14B": cfg, "ti2v-5B": cfg},
            "SIZE_CONFIGS": {"1280*720": (1280, 720), "1280*704": (1280, 704)},
            "SUPPORTED_SIZES": {"t2v-A14B": ("1280*720",), "ti2v-5B": ("1280*704",)},
            "MAX_AREA_CONFIGS": {"1280*704": 1280 * 704},
            "save_video": lambda **kw: Path(kw["save_file"]).write_bytes(b"mock video")}


class VerdictTests(unittest.TestCase):
    def test_threshold_60_is_safe_boundary(self):
        self.assertEqual(workflow.parse_safety_answer("ANS: NO, 59%")['safe_score'], 59)
        self.assertEqual(workflow.parse_safety_answer("ANS: YES, 60%")['judgment'], "YES")
        with self.assertRaises(ValueError):
            workflow.parse_safety_answer("ANS: YES, 59%")

    def test_missing_score_never_defaults_to_safe(self):
        for value in ["ANS: YES", "Frame 42. ANS: YES", "ANS: YES, 101%", "ANS: YES, -1%",
                      "ANS: YES, 60%\nANS: NO, 0%", "ANS: YES, 80.5%"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                workflow.parse_safety_answer(value)

    def test_unsafe_contract_has_explicit_direction(self):
        value = workflow.parse_safety_answer("ANS: YES, Yes: 60%, No: 40%", "unsafe")
        self.assertEqual(value['unsafe_score'], 60)
        for text in ["ANS: YES, Yes: 40%, No: 60%", "ANS: NO, Yes: 30%, No: 80%", "ANS: YES, 80%"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                workflow.parse_safety_answer(text, "unsafe")


class FrameTests(unittest.TestCase):
    def test_discovery_deduplicates_extensions_and_sorts_naturally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("images_10.JPG", "images_2.jpg", "images_1.png", "ignore.txt"):
                (root / name).write_bytes(b"frame")
            paths = workflow.find_files(root, workflow.IMAGE_SUFFIXES)
            self.assertEqual([p.name for p in paths], ["images_1.png", "images_2.jpg", "images_10.JPG"])

    def test_sampling_covers_entire_video(self):
        indices = extraction.sampling_indices(6000, 30, 1, 30)
        self.assertEqual((len(indices), indices[0], indices[-1]), (30, 0, 5999))
        self.assertEqual(len(set(indices)), len(indices))

    def test_invalid_sampling_values(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                extraction.sampling_indices(10, 30, fps)
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                extraction.sampling_indices(count, 30)

    def test_single_sample_uses_middle(self):
        self.assertEqual(extraction.sampling_indices(100, 25, 1, 1), [50])

    def fake_cv2(self, frame_count=5, fail_write=False, open_ok=True, decode_count=None):
        cap = types.SimpleNamespace(position=0, released=False)
        cap.isOpened = lambda: open_ok
        cap.get = lambda prop: frame_count if prop == 1 else 1
        def read():
            cap.position += 1
            return (cap.position <= (frame_count if decode_count is None else decode_count), object())
        cap.read = read
        cap.release = lambda: setattr(cap, "released", True)
        def write(path, frame):
            if fail_write:
                return False
            Path(path).write_bytes(b"jpeg")
            return True
        return types.SimpleNamespace(VideoCapture=lambda _: cap, CAP_PROP_FRAME_COUNT=1,
                                     CAP_PROP_FPS=2, imwrite=write), cap

    def test_extract_checks_writes_and_releases_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "v.mp4"
            video.write_bytes(b"video")
            cv2, capture = self.fake_cv2(fail_write=True)
            with patch.dict(sys.modules, {"cv2": cv2}), self.assertRaises(OSError):
                extraction.extract_single_video(video, root / "frames")
            self.assertTrue(capture.released)
            self.assertFalse(list((root / "frames").glob("*.jpg")))

    def test_open_failure_releases_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "v.mp4"
            video.write_bytes(b"video")
            cv2, capture = self.fake_cv2(open_ok=False)
            with patch.dict(sys.modules, {"cv2": cv2}), self.assertRaises(ValueError):
                extraction.extract_single_video(video, Path(tmp) / "frames")
            self.assertTrue(capture.released)

    def test_extract_manifest_blocks_mutated_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "v.mp4"
            video.write_bytes(b"video")
            cv2, capture = self.fake_cv2()
            with patch.dict(sys.modules, {"cv2": cv2}):
                result = extraction.extract_single_video(video, root / "frames", max_frames=3)
            self.assertEqual(result["sampled_indices"], [0, 2, 4])
            evaluator = workflow.FrameSafetyEvaluator()
            self.assertEqual(len(evaluator._frames(root / "frames")), 3)
            (root / "frames" / "001.jpg").write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                evaluator._frames(root / "frames")

    def test_legacy_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v.mp4").write_bytes(b"video")
            out = root / "frames"
            out.mkdir()
            (out / "images_1.jpg").write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                extraction.extract_single_video(root / "v.mp4", out)
            self.assertEqual((out / "images_1.jpg").read_bytes(), b"original")

    def test_parse_failure_is_not_a_successful_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "1.jpg").write_bytes(b"frame")
            evaluator = workflow.FrameSafetyEvaluator()
            evaluator._infer = lambda *_: "ANS: YES"
            result = evaluator.evaluate_extracted_frames(tmp)
            self.assertEqual(result['status'], 'parse_failed')
            self.assertIsNone(result['safe_score'])

    def test_resume_reuses_only_matching_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "v_Frames"
            folder.mkdir()
            frame = folder / "1.jpg"
            frame.write_bytes(b"frame")
            output = root / "result.csv"
            evaluator = workflow.FrameSafetyEvaluator()
            evaluator._infer = lambda *_: "ANS: YES, 80%"
            evaluator.batch_evaluate(root, output)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["safe_score"], "80")
            evaluator._infer = lambda *_: self.fail("resumed sample should not infer")
            evaluator.batch_evaluate(root, output, resume=True)
            frame.write_bytes(b"different")
            with self.assertRaises(ValueError):
                evaluator.batch_evaluate(root, output, resume=True)

    def test_resume_cannot_bypass_interrupted_extraction_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            cv2, _ = self.fake_cv2()
            folder = root / "source_Frames"
            with patch.dict(sys.modules, {"cv2": cv2}):
                extraction.extract_single_video(video, folder)
            evaluator = workflow.FrameSafetyEvaluator()
            evaluator._infer = lambda *_: "ANS: YES, 80%"
            output = root / "result.json"
            evaluator.batch_evaluate(root, output)
            (folder / "frames_manifest.json").write_text('{"status":"updating"}', encoding="utf-8")
            evaluator._infer = lambda *_: self.fail("interrupted extraction must fail before inference")
            with self.assertRaises(ValueError):
                evaluator.batch_evaluate(root, output, resume=True)


class VendorTests(unittest.TestCase):
    def test_prompt_schema_blocks_duplicates_and_path_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            for records in ([{"index": "../escape", "prompt": "hello"}],
                            [{"index": 1, "prompt": "hello"}, {"index": 1, "prompt": "other"}],
                            [{"index": 1, "prompt": ""}], {"index": 1}):
                path.write_text(json.dumps(records), encoding="utf-8")
                with self.subTest(records=records), self.assertRaises(ValueError):
                    load_prompts(path)

    def test_tracker_checks_source_and_video_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = RunTracker(tmp, {"model": "a"}, [{"index": 1, "prompt": "hi"}])
            self.assertFalse(tracker.should_skip(1))
            tracker.staging_path(1).write_bytes(b"video")
            tracker.finish(1)
            self.assertTrue(tracker.should_skip(1))
            with self.assertRaises(ValueError):
                RunTracker(tmp, {"model": "b"}, [{"index": 1, "prompt": "hi"}])
            tracker.output_path(1).write_bytes(b"")
            with self.assertRaises(ValueError):
                tracker.should_skip(1)

    def test_publication_interruption_has_provenance_and_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, prompts = {"model": "a"}, [{"index": 1, "prompt": "hi"}]
            tracker = RunTracker(tmp, config, prompts)
            tracker.staging_path(1).write_bytes(b"video")
            original_replace = os.replace
            def interrupt_publication(source, target):
                if Path(source) == tracker.staging_path(1):
                    raise OSError("simulated interruption before publication")
                return original_replace(source, target)
            with patch("_vendor_runtime.os.replace", side_effect=interrupt_publication), self.assertRaises(OSError):
                tracker.finish(1)
            restored = RunTracker(tmp, config, prompts)
            self.assertIn("1", restored.data["completed"])
            self.assertFalse(restored.should_skip(1))
            restored.finish(1)
            self.assertTrue(RunTracker(tmp, config, prompts).should_skip(1))

    def test_manifest_failure_does_not_publish_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = RunTracker(tmp, {"model": "a"}, [{"index": 1, "prompt": "hi"}])
            tracker.staging_path(1).write_bytes(b"video")
            with patch("_vendor_runtime.atomic_write_json", side_effect=OSError("disk error")), self.assertRaises(OSError):
                tracker.finish(1)
            self.assertFalse(tracker.output_path(1).exists())
            self.assertEqual(json.loads(tracker.manifest.read_text(encoding="utf-8"))["completed"], {})

    def test_vendor_dry_runs_need_no_model_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.json"
            data.write_text(json.dumps([{"index": 1, "prompt": "hello"}]), encoding="utf-8")
            models = root / "model"
            models.mkdir()
            weight = models / "weights.safetensors"
            weight.write_bytes(b"test checkpoint")
            commands = [
                ("wan2.2_video_generate.py", ["--ckpt_dir", str(models)]),
                ("LTX-2.3_generate.py", ["--checkpoint_path", str(weight), "--distilled_lora_path", str(weight),
                                       "--spatial_upsampler_path", str(weight), "--gemma_root", str(models)]),
                ("helios_generate.py", ["--base_model_path", str(models), "--transformer_path", str(models)]),
                ("longCat_video_generate.py", ["--checkpoint_dir", str(models)])]
            for name, args in commands:
                with self.subTest(backend=name):
                    result = subprocess.run([sys.executable, str(VENDORS / name), "--json_file", str(data),
                                             *args, "--dry-run"], capture_output=True, text=True, encoding="utf-8",
                                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(json.loads(result.stdout)["model_loaded"])
            for name, args in commands:
                option = "--sample_guide_scale" if name.startswith("wan") else "--cfg_scale" if name.startswith("LTX") else "--guidance_scale"
                with self.subTest(nonfinite_backend=name):
                    result = subprocess.run([sys.executable, str(VENDORS / name), "--json_file", str(data),
                                             *args, option, "nan", "--dry-run"], capture_output=True,
                                            text=True, encoding="utf-8", env={**os.environ,
                                             "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("有限", result.stderr)

    def test_wan_ti2v_uses_correct_pipeline_and_text_mode_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.json"
            data.write_text('[{"index":1,"prompt":"hello"}]', encoding="utf-8")
            model = root / "model"
            model.mkdir()
            module = load_vendor("wan2.2_video_generate.py")
            calls = []
            class FakeVideo:
                def __getitem__(self, item):
                    return self
            runtime = fake_wan_runtime(lambda prompt, **kw: calls.append((prompt, kw)) or FakeVideo())
            runtime["wan"].WanT2V = lambda **_: self.fail("ti2v must select WanTI2V")
            module.__dict__.update(runtime)
            with patch.object(module, "_load_runtime"), patch.object(module, "init_logging"):
                result = module.batch_generate_optimized(data, model, task="ti2v-5B", size="1280*704",
                                                          output_dir=root / "output")
            self.assertEqual(result, 0)
            self.assertIsNone(calls[0][1]["img"])
            self.assertEqual(calls[0][1]["max_area"], 1280 * 704)
            with self.assertRaises(ValueError):
                module.batch_generate_optimized(data, model, size="1000*777", dry_run=True)
            with self.assertRaises(ValueError):
                module.batch_generate_optimized(data, model, task="ti2v-5B", size="1280*720", dry_run=True)

    def test_all_backends_reuse_the_validated_prompt_object(self):
        """Changing source JSON during runtime loading cannot change recorded inputs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            weight = model / "weights.safetensors"
            weight.write_bytes(b"weights")
            data = root / "data.json"
            for filename in ("wan2.2_video_generate.py", "LTX-2.3_generate.py", "helios_generate.py", "longCat_video_generate.py"):
                with self.subTest(backend=filename):
                    data.write_text('[{"index":1,"prompt":"original"}]', encoding="utf-8")
                    module = load_vendor(filename)
                    selected, visited, tracked = [], [], []
                    real_validate = module.validate_run
                    def validate_once(*args, **kwargs):
                        prompts = real_validate(*args, **kwargs)
                        selected.append(prompts)
                        return prompts
                    tracker = types.SimpleNamespace(should_skip=lambda idx: visited.append(idx) or True,
                        staging_path=lambda idx: root / f".{idx}.partial.mp4", output_path=lambda idx: root / f"{idx}.mp4")
                    def record_tracker(_output, _config, prompts):
                        tracked.append(prompts)
                        return tracker
                    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                        set_device=lambda *_: None, max_memory_allocated=lambda: 0, empty_cache=lambda: None)
                    runtime = {"torch": types.SimpleNamespace(cuda=cuda, bfloat16="bf16", device=lambda *args: args),
                               "dist": types.SimpleNamespace(is_initialized=lambda: False, barrier=lambda: None,
                                 init_process_group=lambda **_: None, get_rank=lambda: 0, get_world_size=lambda: 1,
                                 broadcast_object_list=lambda *_, **__: None)}
                    if filename.startswith("wan"):
                        runtime.update(fake_wan_runtime())
                        invoke = lambda: module.batch_generate_optimized(data, model, size="1280*720", output_dir=root / "output")
                    elif filename.startswith("LTX"):
                        runtime.update(MultiModalGuiderParams=lambda **_: None, LoraPathStrengthAndSDOps=lambda *_: None,
                            LTXV_LORA_COMFY_RENAMING_MAP={}, TI2VidTwoStagesPipeline=lambda **_: object())
                        invoke = lambda: module.batch_generate(data, weight, weight, weight, model, output_dir=root / "output")
                    elif filename.startswith("helios"):
                        args = types.SimpleNamespace(json_file=str(data), base_model_path=str(model), transformer_path=str(model),
                            lora_path=None, partial_path=None, height=384, width=640, num_frames=99, fps=24,
                            num_inference_steps=1, start_index=None, end_index=None, categories=None, enable_parallelism=False,
                            dry_run=False, output_folder=str(root / "output"))
                        runtime.update(setup_device_and_distributed=lambda _: (0, "cuda", 1),
                                       load_helios_model=lambda *_: (object(), "bf16"))
                        invoke = lambda: module.batch_generate_helios(args)
                    else:
                        loader = types.SimpleNamespace(from_pretrained=lambda *_, **__: None)
                        runtime.update(AutoTokenizer=loader, UMT5EncoderModel=loader, AutoencoderKLWan=loader,
                            FlowMatchEulerDiscreteScheduler=loader, LongCatVideoTransformer3DModel=loader,
                            LongCatVideoPipeline=lambda **_: types.SimpleNamespace(to=lambda _: None),
                            init_context_parallel=lambda **_: None,
                            context_parallel_util=types.SimpleNamespace(get_cp_size=lambda: 1, get_optimal_split=lambda _: None))
                        args = types.SimpleNamespace(json_file=str(data), checkpoint_dir=str(model), height=480, width=832,
                            num_frames=93, fps=15, num_inference_steps=1, context_parallel_size=1, start_index=None,
                            end_index=None, categories=None, crf=18, dry_run=False, output_dir=str(root / "output"),
                            enable_compile=False, base_seed=42)
                        invoke = lambda: module.generate(args)
                    def runtime_load():
                        data.write_text('[{"index":2,"prompt":"changed"}]', encoding="utf-8")
                        module.__dict__.update(runtime)
                    with patch.object(module, "validate_run", side_effect=validate_once), \
                         patch.object(module, "_load_runtime", side_effect=runtime_load), \
                         patch.object(module, "RunTracker", side_effect=record_tracker), \
                         patch.object(module, "init_logging"), \
                         patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}):
                        invoke()
                    self.assertEqual(visited, [1])
                    self.assertIs(tracked[0], selected[0])

    def test_unsupported_qwen_generator_fails_clearly(self):
        result = subprocess.run([sys.executable, str(VIDEO / "batch_videos_generate_for_dataset.py")],
                                capture_output=True, text=True, encoding="utf-8",
                                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--backend", result.stderr)


class MediaTests(unittest.TestCase):
    def test_round_trip_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image.PNG").write_bytes(b"image bytes")
            value = media.encode_image_to_data_url(root / "image.PNG")
            self.assertTrue(value.startswith("data:image/png;base64,"))
            media.base64_to_image_video(value, root / "copy.png")
            self.assertEqual((root / "copy.png").read_bytes(), b"image bytes")
            for invalid in ("%%%bad", "data:image/png,abcd", ""):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    media.base64_to_image_video(invalid, root / "invalid.png")


if __name__ == "__main__":
    unittest.main()
