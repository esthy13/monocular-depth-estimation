import ast
import json
import unittest
from pathlib import Path


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"

    def load_notebook(self, name):
        return json.loads((self.notebook_dir / name).read_text())

    def test_unidac_uses_current_branch_and_remaining_recordings(self):
        notebook = self.load_notebook("unidac_colab.ipynb")
        sources = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"]
        }

        self.assertIn('PROJECT_REF = "main"', sources["experiment-settings"])
        self.assertIn(
            'BATCH_RECORDINGS = ["recording2", "recording3", "recording4"]',
            sources["resumable-batch"],
        )
        self.assertIn('BATCH_SENSORS = ["G1_A"]', sources["resumable-batch"])
        self.assertIn("FRAME_STEP = 1", sources["resumable-batch"])
        self.assertIn("MAX_FRAMES_PER_SENSOR = None", sources["resumable-batch"])
        self.assertIn("for recording in EVALUATION_RECORDINGS", sources["person-evaluation"])

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"unidac_colab.ipynb:cell-{index}",
                )

    def test_dac_loops_remaining_evaluations_and_comparisons(self):
        notebook = self.load_notebook("dac_colab.ipynb")
        sources = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"]
        }

        self.assertIn("for recording in EVALUATION_RECORDINGS", sources["person-evaluation"])
        self.assertIn("for recording in COMPARISON_RECORDINGS", sources["comparison"])
        self.assertIn("compare_evaluation_suite.py", sources["comparison"])

    def test_benchmark_notebook_cells_compile_and_protocol_is_controlled(self):
        notebook = self.load_notebook("benchmark_colab.ipynb")
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"benchmark_colab.ipynb:cell-{index}",
                )

        settings = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["id"] == "settings"
        )
        self.assertIn("WARMUP_RUNS = 5", settings)
        self.assertIn("TIMED_RUNS_PER_FRAME = 10", settings)
        self.assertIn('MODEL_ORDER = ["DAC", "UniDAC"]', settings)


if __name__ == "__main__":
    unittest.main()
