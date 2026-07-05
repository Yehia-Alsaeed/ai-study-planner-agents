import os
import tempfile
import unittest
from pathlib import Path

from study_planner.config import MissingEnvironmentVariable, configure_paths, load_api_keys


class ConfigTests(unittest.TestCase):
    def test_configure_paths_defaults_to_local_outputs_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                paths = configure_paths(create=True)
            finally:
                os.chdir(original_cwd)

            expected_root = Path(temp_dir) / "outputs"
            self.assertEqual(paths.project_dir, expected_root)
            self.assertEqual(paths.results_dir, expected_root / "variation_outputs")
            self.assertEqual(paths.log_dir, expected_root / "tensorboard_logs")
            self.assertTrue(paths.results_dir.is_dir())
            self.assertTrue(paths.log_dir.is_dir())

    def test_configure_paths_uses_explicit_project_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_dir = Path(temp_dir) / "planner-artifacts"
            paths = configure_paths(project_dir=explicit_dir, create=True)

            self.assertEqual(paths.project_dir, explicit_dir)
            self.assertTrue((explicit_dir / "variation_outputs").is_dir())
            self.assertTrue((explicit_dir / "tensorboard_logs").is_dir())

    def test_load_api_keys_reads_dotenv_before_prompting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=test-openai\nSERPER_API_KEY=test-serper\n",
                encoding="utf-8",
            )

            with self._clean_env("OPENAI_API_KEY", "SERPER_API_KEY"):
                keys = load_api_keys(env_file=env_file, prompt_missing=False)

            self.assertEqual(keys["OPENAI_API_KEY"], "test-openai")
            self.assertEqual(keys["SERPER_API_KEY"], "test-serper")

    def test_load_api_keys_raises_for_missing_values_when_prompting_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("OPENAI_API_KEY=test-openai\n", encoding="utf-8")

            with self._clean_env("OPENAI_API_KEY", "SERPER_API_KEY"):
                with self.assertRaises(MissingEnvironmentVariable) as context:
                    load_api_keys(env_file=env_file, prompt_missing=False)

            self.assertIn("SERPER_API_KEY", str(context.exception))

    class _clean_env:
        def __init__(self, *names):
            self.names = names
            self.original_values = {}

        def __enter__(self):
            for name in self.names:
                self.original_values[name] = os.environ.pop(name, None)

        def __exit__(self, exc_type, exc, traceback):
            for name, value in self.original_values.items():
                if value is not None:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
