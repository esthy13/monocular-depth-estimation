import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.depth_models import _huggingface_token


class HuggingFaceTokenTests(unittest.TestCase):
    def test_reads_hf_token_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("OTHER=value\nHF_TOKEN='token-from-file'\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_huggingface_token(env_file), "token-from-file")
                self.assertEqual(os.environ["HF_TOKEN"], "token-from-file")

    def test_process_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("HF_TOKEN=token-from-file\n")
            with patch.dict(os.environ, {"HF_TOKEN": "token-from-environment"}, clear=True):
                self.assertEqual(_huggingface_token(env_file), "token-from-environment")


if __name__ == "__main__":
    unittest.main()
