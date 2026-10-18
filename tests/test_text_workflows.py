"""Network-free regression tests for HTTP and resumable table workflows."""
import csv
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from llm_quick_use.batch import batch_cli, read_table, run_chat_batch
from llm_quick_use.http import ChatClient
from llm_quick_use.io import read_jsonl


def response(content="hello", raw=None):
    value = Mock()
    value.getcode.return_value = 200
    value.read.return_value = raw if raw is not None else json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    value.__enter__ = Mock(return_value=value)
    value.__exit__ = Mock(return_value=False)
    return value


class HttpTests(unittest.TestCase):
    def client(self, **kwargs):
        return ChatClient("https://example.invalid/v1/chat/completions", "test", **kwargs)

    def test_success_uses_timeout_and_auth_without_persisting_key(self):
        client = self.client(api_key="test-secret", timeout=11)
        with patch("urllib.request.urlopen", return_value=response()) as request:
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(request.call_args.kwargs["timeout"], 11)
        self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer test-secret")
        self.assertNotIn("test-secret", repr(client))
        self.assertNotIn("test-secret", json.dumps(client.public_config()))

    def test_401_is_not_retried_or_leaked(self):
        error = urllib.error.HTTPError("https://example.invalid", 401, "private server text", {}, io.BytesIO(b"private"))
        with patch("urllib.request.urlopen", side_effect=error) as request:
            result = self.client().complete([])
        self.assertEqual(result["error"], "HTTP 401")
        self.assertEqual(request.call_count, 1)

    def test_429_retries_then_succeeds(self):
        error = urllib.error.HTTPError("https://example.invalid", 429, "rate", {}, None)
        with patch("urllib.request.urlopen", side_effect=[error, response()]) as request, patch("llm_quick_use.http.time.sleep"):
            result = self.client().complete([])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["attempts"], 2)

    def test_503_retry_bound(self):
        error = urllib.error.HTTPError("https://example.invalid", 503, "unavailable", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as request, patch("llm_quick_use.http.time.sleep"):
            result = self.client(max_attempts=2).complete([])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["status"], "error")

    def test_invalid_json_not_retried(self):
        with patch("urllib.request.urlopen", return_value=response(raw=b"<html>private</html>")) as request:
            result = self.client().complete([])
        self.assertEqual(result["error"], "Invalid chat JSON response")
        self.assertEqual(request.call_count, 1)

    def test_empty_choices_rejected(self):
        with patch("urllib.request.urlopen", return_value=response(raw=b'{"choices": []}')):
            self.assertEqual(self.client().complete([])["status"], "error")

    def test_malformed_chat_schema_returns_structured_error(self):
        values = [{"choices": [None]}, {"choices": {}}, {"choices": ["text"]},
                  {"choices": [{"message": None}]}, {"choices": [{"message": {"content": []}}]}, []]
        for value in values:
            with self.subTest(value=value), patch("urllib.request.urlopen", return_value=response(raw=json.dumps(value).encode())):
                self.assertEqual(self.client().complete([])["status"], "error")

    def test_truncated_safe_verdict_is_not_success(self):
        raw = json.dumps({"choices": [{"message": {"content": "Safe"}, "finish_reason": "length"}]}).encode()
        with patch("urllib.request.urlopen", return_value=response(raw=raw)):
            result = self.client().complete([])
        self.assertEqual(result["status"], "error")

    def test_timeout_can_disable_retries(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("secret")) as request:
            result = self.client(max_attempts=1).complete([])
        self.assertEqual(request.call_count, 1)
        self.assertNotIn("secret", result["error"])

    def test_invalid_config_rejected(self):
        for kwargs in ({"timeout": 0}, {"max_attempts": 0}, {"temperature": float("nan")}, {"max_tokens": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.client(**kwargs)
        with self.assertRaises(ValueError):
            ChatClient("https://user:secret@example.invalid", "test")


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "result.csv"
        self.client = ChatClient("https://example.invalid/chat", "test")
        self.rows = [{"Prompts": "one"}, {"Prompts": "two"}]

    def test_generate_all_rows_and_resume_without_network(self):
        with patch.object(self.client, "complete", return_value={"status": "ok", "content": "answer", "error": "", "attempts": 1}) as call:
            summary = run_chat_batch(self.rows, self.path, self.client)
            again = run_chat_batch(self.rows, self.path, self.client)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(summary["ok"], 2)
        self.assertEqual(again, summary)
        self.assertEqual(len(read_table(self.path)), 2)

    def test_interrupt_preserves_completed_rows_and_resumes(self):
        success = {"status": "ok", "content": "answer", "error": "", "attempts": 1}
        with patch.object(self.client, "complete", side_effect=[success, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                run_chat_batch(self.rows, self.path, self.client)
        self.assertEqual(len(read_jsonl(str(self.path) + ".jsonl", strict=True)), 1)
        with patch.object(self.client, "complete", return_value=success) as call:
            summary = run_chat_batch(self.rows, self.path, self.client)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(summary["ok"], 2)

    def test_resume_retries_errors_but_keeps_successes(self):
        success = {"status": "ok", "content": "answer", "error": "", "attempts": 1}
        failure = {"status": "error", "content": "", "error": "HTTP 503", "attempts": 3}
        with patch.object(self.client, "complete", side_effect=[success, failure]):
            self.assertEqual(run_chat_batch(self.rows, self.path, self.client)["errors"], 1)
        with patch.object(self.client, "complete", return_value=success) as call:
            self.assertEqual(run_chat_batch(self.rows, self.path, self.client)["ok"], 2)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(read_jsonl(str(self.path) + ".jsonl")), 3)
        with patch.object(self.client, "complete") as call:
            self.assertEqual(run_chat_batch(self.rows, self.path, self.client)["ok"], 2)
        call.assert_not_called()

    def test_truncated_journal_tail_is_recovered(self):
        success = {"status": "ok", "content": "answer", "error": "", "attempts": 1}
        with patch.object(self.client, "complete", side_effect=[success, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                run_chat_batch(self.rows, self.path, self.client)
        with Path(str(self.path) + ".jsonl").open("ab") as handle:
            handle.write(b'{"row_index":')
        with patch.object(self.client, "complete", return_value=success) as call, self.assertLogs(level="WARNING"):
            summary = run_chat_batch(self.rows, self.path, self.client)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(summary["ok"], 2)

    def test_source_id_and_labels_are_preserved(self):
        with patch.object(self.client, "complete", return_value={"status": "ok", "content": "answer", "error": "", "attempts": 1}):
            run_chat_batch([{"id": "sample-1", "label": "category", "Prompt": "hello"}], self.path, self.client,
                           prompt_column="Prompt", output_column="Answers")
        result = read_table(self.path)[0]
        self.assertEqual(result["id"], "sample-1")
        self.assertEqual(result["label"], "category")
        self.assertEqual(result["Prompts"], "hello")
        self.assertEqual(result["Answers"], "answer")

    def test_invalid_valid_json_journal_entry_is_rejected(self):
        with patch.object(self.client, "complete", return_value={"status": "ok", "content": "answer", "error": "", "attempts": 1}):
            run_chat_batch(self.rows, self.path, self.client)
        journal = Path(str(self.path) + ".jsonl")
        entry = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
        entry["result"]["attempts"] = "corrupted"
        journal.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Journal"):
            run_chat_batch(self.rows, self.path, self.client)

    def test_changed_inputs_and_model_are_rejected(self):
        with patch.object(self.client, "complete", return_value={"status": "ok", "content": "a", "error": "", "attempts": 1}):
            run_chat_batch(self.rows, self.path, self.client)
        with self.assertRaisesRegex(ValueError, "changed"):
            run_chat_batch([{"Prompts": "changed"}], self.path, self.client)
        self.client.model = "new-model"
        with self.assertRaisesRegex(ValueError, "changed"):
            run_chat_batch(self.rows, self.path, self.client)

    def test_upstream_failure_never_reaches_judge(self):
        rows = [{"Prompts": "x", "Responses": "裁判模型未响应"},
                {"Prompts": "y", "Responses": "", "status": "error"}]
        with patch.object(self.client, "complete") as call:
            summary = run_chat_batch(rows, self.path, self.client, task="safety", output_column="Safety_Eval_Results")
        call.assert_not_called()
        self.assertEqual(summary["skipped"], 2)

    def test_invalid_judge_verdict_is_error(self):
        with patch.object(self.client, "complete", return_value={"status": "ok", "content": "Probably Safe", "error": "", "attempts": 1}):
            summary = run_chat_batch([{"Prompts": "x", "Responses": "y"}], self.path, self.client,
                                     task="safety", output_column="Safety_Eval_Results")
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(read_table(self.path)[0]["Safety_Eval_Results"], "")

    def test_case_insensitive_valid_verdict_normalized(self):
        with patch.object(self.client, "complete", return_value={"status": "ok", "content": " unsafe\n", "error": "", "attempts": 1}):
            summary = run_chat_batch([{"Prompts": "x", "Responses": "y"}], self.path, self.client,
                                     task="safety", output_column="Safety_Eval_Results")
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(read_table(self.path)[0]["Safety_Eval_Results"], "UnSafe")

    def test_legacy_output_without_provenance_not_overwritten(self):
        self.path.write_text("old results", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance"):
            run_chat_batch(self.rows, self.path, self.client)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "old results")

    def test_empty_prompt_fails_before_network_or_output(self):
        with patch.object(self.client, "complete") as call, self.assertRaises(ValueError):
            run_chat_batch([{"Prompts": ""}], self.path, self.client)
        call.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_dry_run_is_network_free(self):
        source = Path(self.temp.name) / "source.csv"
        source.write_text("Prompts\nhello\n", encoding="utf-8")
        with patch("urllib.request.urlopen") as call, patch("builtins.print"):
            code = batch_cli(argv=["--input", str(source), "--output", str(self.path), "--endpoint", self.client.endpoint,
                                   "--model", "test", "--dry-run"])
        self.assertEqual(code, 0)
        call.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_cli_rejects_input_output_same_path(self):
        source = Path(self.temp.name) / "source.csv"
        source.write_text("Prompts\nhello\n", encoding="utf-8")
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
            batch_cli(argv=["--input", str(source), "--output", str(source), "--endpoint", self.client.endpoint, "--model", "test"])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(source.read_text(encoding="utf-8"), "Prompts\nhello\n")


if __name__ == "__main__":
    unittest.main()
