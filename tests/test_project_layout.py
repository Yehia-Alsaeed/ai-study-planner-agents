import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_notebook_lives_under_notebooks_and_uses_package_helpers(self):
        root_notebook = REPO_ROOT / "study_planner.ipynb"
        demo_notebook = REPO_ROOT / "notebooks" / "study_planner_demo.ipynb"

        self.assertFalse(root_notebook.exists())
        self.assertTrue(demo_notebook.exists())

        notebook = json.loads(demo_notebook.read_text(encoding="utf-8-sig"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )

        self.assertIn("configure_paths", source)
        self.assertIn("load_api_keys", source)
        self.assertNotIn("/content/drive/MyDrive/NLP Assessment II", source)

    def test_examples_use_consistent_deep_learning_name(self):
        sample_input_path = REPO_ROOT / "examples" / "sample_input.json"
        sample_output_path = REPO_ROOT / "examples" / "sample_output.md"

        self.assertTrue(sample_input_path.exists())
        self.assertTrue(sample_output_path.exists())

        sample_input = json.loads(sample_input_path.read_text(encoding="utf-8"))
        subject_names = [subject["name"] for subject in sample_input["subjects"]]
        sample_output = sample_output_path.read_text(encoding="utf-8")

        self.assertIn("Deep Learning", subject_names)
        self.assertNotIn("Deep", subject_names)
        self.assertIn("Deep Learning", sample_output)


if __name__ == "__main__":
    unittest.main()
