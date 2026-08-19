import ast
import json
import unittest
from pathlib import Path


class DACNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "notebooks" / "dac_colab.ipynb"
        cls.notebook = json.loads(cls.path.read_text())

    def test_cell_ids_are_unique(self):
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]

        self.assertEqual(len(cell_ids), len(set(cell_ids)))

    def test_all_python_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            ast.parse(
                "".join(cell["source"]),
                filename=f"{self.path}:cell-{index}",
            )

    def test_batch_defaults_cover_remaining_full_sequences(self):
        batch_cell = next(
            cell for cell in self.notebook["cells"]
            if cell["id"] == "resumable-batch"
        )
        batch_source = "".join(batch_cell["source"])

        self.assertIn(
            'BATCH_RECORDINGS = ["recording2", "recording3", "recording4"]',
            batch_source,
        )
        self.assertIn('BATCH_SENSORS = ["G1_A"]', batch_source)
        self.assertIn("FRAME_STEP = 1", batch_source)
        self.assertIn("MAX_FRAMES_PER_SENSOR = None", batch_source)


if __name__ == "__main__":
    unittest.main()
