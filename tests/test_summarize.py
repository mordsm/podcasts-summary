import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from src import summarize


class RunLockTests(unittest.TestCase):
    def test_run_lock_prevents_concurrent_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = Path(tmp_dir) / "pipeline.lock"
            with patch.object(main, "LOCK_PATH", lock_path):
                self.assertTrue(main.acquire_run_lock())
                self.assertFalse(main.acquire_run_lock())
                main.release_run_lock()
                self.assertTrue(main.acquire_run_lock())
                main.release_run_lock()


class OpenAIApiKeyTests(unittest.TestCase):
    def test_ignores_github_token_outside_actions(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "actions-token"}, clear=True):
            self.assertEqual(summarize._openai_api_key(), "")

    def test_uses_openai_api_key_when_present(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            self.assertEqual(summarize._openai_api_key(), "openai-key")

    def test_ignores_github_token_in_actions(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_TOKEN": "actions-token",
            },
            clear=True,
        ):
            self.assertEqual(summarize._openai_api_key(), "")


class TelegramDeliveryTests(unittest.TestCase):
    def test_send_telegram_treats_missing_config_as_skipped_success(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(main.send_telegram("hello"))

    def test_send_telegram_returns_false_on_unauthorized_response(self):
        class Response:
            ok = False
            status_code = 401
            text = '{"ok":false,"error_code":401,"description":"Unauthorized"}'

        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "bad-token", "TELEGRAM_CHAT_ID": "chat"},
            clear=True,
        ), patch("requests.post", return_value=Response()):
            self.assertFalse(main.send_telegram("hello"))


class ResendHistoryTests(unittest.TestCase):
    def test_result_block_date_reads_generated_date(self):
        block = "Title\n20/07/2026 10:00 UTC\n26/07/2026 13:59 UTC [Generated]"

        self.assertEqual(main._result_block_date(block).date().isoformat(), "2026-07-26")

    def test_resend_history_since_filters_old_entries(self):
        content = """----
old
19/07/2026 09:00 UTC [Generated]
----
new
20/07/2026 09:00 UTC [Generated]
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "results.txt.md"
            results_path.write_text(content, encoding="utf-8")
            sent = []
            with patch.object(main, "RESULTS_PATH", results_path), patch.object(
                main, "send_telegram", side_effect=lambda text: sent.append(text) or True
            ):
                main.resend_history_since(main.datetime(2026, 7, 20, tzinfo=main.timezone.utc))

        self.assertEqual(sent, ["new\n20/07/2026 09:00 UTC [Generated]"])

    def test_resend_history_since_arg_triggers_resend_mode(self):
        called = []
        with patch("sys.argv", ["main.py", "--resend-history-since", "2026-07-20"]), patch.object(
            main, "resend_history_since", side_effect=lambda since: called.append(since)
        ):
            main.main()

        self.assertEqual(called[0].date().isoformat(), "2026-07-20")


if __name__ == "__main__":
    unittest.main()
