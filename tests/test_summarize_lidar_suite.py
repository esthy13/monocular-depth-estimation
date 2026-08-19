import json
import math
import tempfile
import unittest
from pathlib import Path

from summarize_lidar_suite import (
    aggregate_row,
    load_summaries,
    recording_row,
)


def fixture(recording: str, count: int, mae: float, rmse: float) -> dict:
    return {
        "recording": recording,
        "camera_sensor": "ZED_B",
        "lidar_sensor": "E1_A",
        "evaluated_frames": count,
        "alignment_method": "none",
        "evaluation_role": "metric_evaluation",
        "timestamp_dt_min_s": -0.01,
        "timestamp_dt_max_s": 0.02,
        "timestamp_dt_mean_s": 0.005,
        "timestamp_dt_std_s": 0.01,
        "metrics": {
            "count": count,
            "mae_m": mae,
            "rmse_m": rmse,
            "abs_rel": mae / 10,
            "sq_rel": mae / 5,
            "rmse_log": rmse / 10,
            "delta_1": 0.7,
            "delta_2": 0.8,
            "delta_3": 0.9,
        },
    }


class SummarizeLidarSuiteTests(unittest.TestCase):
    def test_aggregate_uses_point_weighting_and_squared_rmse(self):
        summaries = [
            fixture("recording1", count=1, mae=1.0, rmse=2.0),
            fixture("recording2", count=3, mae=3.0, rmse=4.0),
        ]
        result = aggregate_row(summaries)
        self.assertEqual(result["evaluated_frames"], 4)
        self.assertEqual(result["lidar_points"], 4)
        self.assertAlmostEqual(result["mae_m"], 2.5)
        self.assertAlmostEqual(result["rmse_m"], math.sqrt(13.0))

    def test_recording_row_maps_global_metrics(self):
        row = recording_row(fixture("recording1", count=10, mae=0.5, rmse=1.0))
        self.assertEqual(row["scope"], "recording1")
        self.assertEqual(row["lidar_points"], 10)
        self.assertEqual(row["alignment_method"], "none")

    def test_loading_rejects_nonmetric_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "recording1"
            result_dir.mkdir()
            payload = fixture("recording1", count=1, mae=1.0, rmse=1.0)
            payload["alignment_method"] = "inverse_least_squares"
            path = result_dir / "recording1_ZED_B_E1_A_lidar_global_metrics.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "alignment_method"):
                load_summaries(
                    root,
                    ["recording1"],
                    "ZED_B",
                    "E1_A",
                    "none",
                    "metric_evaluation",
                )


if __name__ == "__main__":
    unittest.main()
