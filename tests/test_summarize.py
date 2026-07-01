import os
import unittest
from unittest.mock import patch

from src import summarize


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
