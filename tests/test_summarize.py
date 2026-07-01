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


class GitHubModelsTokenTests(unittest.TestCase):
    def test_ignores_github_token_outside_actions(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "actions-token"}, clear=True):
            self.assertEqual(summarize._github_models_token(), "")

    def test_uses_models_token_when_present(self):
        with patch.dict(os.environ, {"MODELS_TOKEN": "models-token"}, clear=True):
            self.assertEqual(summarize._github_models_token(), "models-token")

    def test_ignores_github_token_in_actions(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_TOKEN": "actions-token",
            },
            clear=True,
        ):
            self.assertEqual(summarize._github_models_token(), "")


if __name__ == "__main__":
    unittest.main()
