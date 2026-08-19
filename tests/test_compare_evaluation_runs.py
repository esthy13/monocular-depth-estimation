import json
import tempfile
import unittest
from pathlib import Path

from compare_evaluation_runs import (
    comparability_warnings,
    load_comparison_row,
    markdown_table,
    parse_run_spec,
)


class CompareEvaluationRunsTests(unittest.TestCase):
    def test_run_spec_preserves_paths_with_equals_characters(self):
        label, path = parse_run_spec("UniDAC=/tmp/results=a/summary.json")

        self.assertEqual(label, "UniDAC")
        self.assertEqual(path, Path("/tmp/results=a/summary.json"))

    def test_summary_fields_are_loaded_for_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "prediction_models": ["UniDAC"],
                        "recording": "recording1",
                        "processed_frames": 10,
                        "mean_zed_mae_m": 0.25,
                        "configuration": {"frame_step": 10},
                    }
                )
            )

            row = load_comparison_row("unidac", path)

        self.assertEqual(row["prediction_models"], "UniDAC")
        self.assertEqual(row["mean_zed_mae_m"], 0.25)
        self.assertIn("ZED MAE m", markdown_table([row]))

    def test_mismatched_protocol_is_reported(self):
        rows = [
            {
                "recording": "recording1",
                "processed_frames": 10,
                "configuration": {"frame_step": 1},
            },
            {
                "recording": "recording1",
                "processed_frames": 5,
                "configuration": {"frame_step": 2},
            },
        ]

        warnings = comparability_warnings(rows)

        self.assertIn("Runs contain different numbers of processed frames.", warnings)
        self.assertIn("Runs use different frame_step settings.", warnings)


if __name__ == "__main__":
    unittest.main()
