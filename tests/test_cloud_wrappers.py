"""Mock-only cloud wrapper verification: no real credentials or network."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ALIYUN = ROOT / "video-content-safety-evaluate" / "aliyun-content-safety-evaluate"
BAIDU = ROOT / "video-content-safety-evaluate" / "baidu-video-content-safety-evaluate"
sys.path.insert(0, str(ALIYUN))
import _aliyun_video as aliyun


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baidu = load_file("baidu_video_wrapper", BAIDU / "video-content-safety-baidu1-api-evaluate.py")
diagnostics = load_file("video_response_analyzer", BAIDU / "response-analyze.py")


class CloudTests(unittest.TestCase):
    def test_all_wrappers_import_without_sdk_or_requests(self):
        with patch.dict(sys.modules, {"requests": None, "alibabacloud_green20220302": None}):
            for path in [*ALIYUN.glob("*.py"), *BAIDU.glob("*.py")]:
                with self.subTest(path=path):
                    runpy.run_path(str(path), run_name="offline_import")

    def test_aliyun_url_and_task_validation(self):
        self.assertEqual(aliyun.service_parameters(task_id="task1"), {"taskId": "task1"})
        for value in ("", "file:///tmp/video", "https://user:password@example.com/video"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                aliyun.service_parameters(url=value)

    def test_aliyun_submission_sets_service_and_parameters(self):
        requests = []
        response = types.SimpleNamespace(status_code=200, body={"Code": 200, "Data": {"TaskId": "t1"}})
        client = types.SimpleNamespace(video_moderation=lambda req: requests.append(req) or response,
            video_moderation_result=lambda req: requests.append(req) or response)
        sdk = types.ModuleType("alibabacloud_green20220302")
        sdk.models = types.SimpleNamespace(VideoModerationRequest=lambda **kw: types.SimpleNamespace(**kw),
            VideoModerationResultRequest=lambda **kw: types.SimpleNamespace(**kw))
        with patch.dict(sys.modules, {"alibabacloud_green20220302": sdk}):
            aliyun.submit_video("https://example.com/live", "liveStreamDetection", client)
            self.assertEqual(requests[-1].service, "liveStreamDetection")
            self.assertEqual(json.loads(requests[-1].service_parameters), {"url": "https://example.com/live"})
            aliyun.pull_result("t1", client=client)
            self.assertEqual(json.loads(requests[-1].service_parameters), {"taskId": "t1"})

    def test_aliyun_http_success_business_failure_is_error(self):
        with self.assertRaises(RuntimeError):
            aliyun.normalize_response(types.SimpleNamespace(status_code=200, body={"Code": 400}))

    def test_missing_baidu_credentials_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            baidu.get_access_token(session=object())

    def test_baidu_auth_rejects_missing_token(self):
        response = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"error": "invalid"})
        session = types.SimpleNamespace(post=lambda *a, **kw: response)
        with self.assertRaises(RuntimeError):
            baidu.get_access_token("test-ak", "test-sk", session)

    def test_baidu_uses_encoded_form_and_bounded_timeouts(self):
        calls = []
        responses = [types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"access_token": "test-token"}),
                     types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"log_id": 123})]
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return responses.pop(0)
        session = types.SimpleNamespace(post=post)
        with patch.dict(os.environ, {"BAIDU_API_KEY": "test-ak", "BAIDU_SECRET_KEY": "test-sk"}):
            result = baidu.evaluate_video("https://example.com/video?a=1&b=2", "test", "id", session, 12)
        self.assertEqual(result, {"log_id": 123})
        self.assertEqual(calls[0][1]['data']['client_secret'], 'test-sk')
        self.assertNotIn('params', calls[0][1])
        self.assertEqual(calls[1][1]['data']['videoUrl'], 'https://example.com/video?a=1&b=2')
        self.assertEqual(calls[1][1]['timeout'], (5, 12))

    def test_url_diagnostics_remove_userinfo_tokens_and_fragments(self):
        url = diagnostics.sanitized_url("https://user:secret@example.com/video?access_token=secret#hidden")
        self.assertNotIn("secret", url)
        self.assertNotIn("user", url)
        self.assertNotIn("hidden", url)


if __name__ == "__main__":
    unittest.main()
