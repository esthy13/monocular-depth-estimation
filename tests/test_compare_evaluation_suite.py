import json
import tempfile
import unittest
from pathlib import Path

from compare_evaluation_suite import (
    aggregate_rows,
    load_suite_rows,
    suite_warnings,
)


class CompareEvaluationSuiteTests(unittest.TestCase):
    def write_summary(self, root, recording, model, frames, zed_error):
        directory = root / recording
        directory.mkdir(parents=True)
        (directory / "evaluation_summary.json").write_text(json.dumps({
            "recording": recording,
            "prediction_models": [model],
            "processed_frames": frames,
            "person_detections": frames,
            "detections_with_zed_reference": frames,
            "detections_with_lidar_reference": frames,
            "mean_depth_inference_ms": float(frames),
            "mean_zed_mae_m": zed_error,
            "mean_zed_rmse_m": zed_error,
            "mean_zed_abs_rel": zed_error,
            "mean_zed_3d_error_m": zed_error,
            "mean_lidar_3d_error_m": zed_error,
            "configuration": {"frame_step": 1},
        }))

    def test_aggregate_is_weighted_by_valid_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dac = root / "dac"
            unidac = root / "unidac"
            for model_root, model in ((dac, "DAC"), (unidac, "UniDAC")):
                self.write_summary(model_root, "recording1", model, 1, 1.0)
                self.write_summary(model_root, "recording2", model, 3, 3.0)

            rows = load_suite_rows(
                [("DAC", dac), ("UniDAC", unidac)],
                ["recording1", "recording2"],
            )
            aggregates = aggregate_rows(rows, ["DAC", "UniDAC"])

            self.assertEqual(aggregates[0]["processed_frames"], 4)
            self.assertEqual(aggregates[0]["mean_zed_3d_error_m"], 2.5)
            self.assertEqual(suite_warnings(rows, ["recording1", "recording2"]), [])


if __name__ == "__main__":
    unittest.main()
