import unittest
from pathlib import Path

import cv2
import numpy as np

from analyze_lidar_outliers import (
    frame_index,
    laplacian_variance,
    masked_laplacian_variance,
    summarize_rows,
)


class AnalyzeLidarOutliersTests(unittest.TestCase):
    def test_frame_index_is_read_from_metrics_filename(self):
        path = Path("recording3_ZED_B_0123_lidar_metrics.json")
        self.assertEqual(frame_index(path), 123)

    def test_laplacian_variance_increases_with_visible_edges(self):
        uniform = np.zeros((32, 32, 3), dtype=np.uint8)
        edged = uniform.copy()
        cv2.rectangle(edged, (8, 8), (24, 24), (255, 255, 255), -1)
        self.assertEqual(laplacian_variance(uniform), 0.0)
        self.assertGreater(laplacian_variance(edged), 0.0)

    def test_masked_laplacian_variance_uses_selected_region(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, 16:] = 255
        left_mask = np.zeros((32, 32), dtype=bool)
        left_mask[:, :12] = True
        edge_mask = np.zeros((32, 32), dtype=bool)
        edge_mask[:, 10:22] = True
        self.assertEqual(masked_laplacian_variance(image, left_mask), 0.0)
        self.assertGreater(masked_laplacian_variance(image, edge_mask), 0.0)

    def test_summary_ranks_largest_rmse(self):
        rows = []
        for index, rmse in enumerate((1.0, 3.0, 2.0)):
            rows.append(
                {
                    "recording": "recording1",
                    "frame_index": index,
                    "image_file": f"{index}.jpg",
                    "image_path": f"/{index}.jpg",
                    "lidar_points": 10,
                    "mae_m": 1.0,
                    "rmse_m": rmse,
                    "rmse_mae_ratio": rmse,
                    "abs_rel": 0.1,
                    "laplacian_variance": float(index + 1),
                    "person_detections": 0,
                    "person_mask_pixels": 0,
                    "person_laplacian_variance": None,
                }
            )
        summary = summarize_rows(rows, top_k=1)
        self.assertEqual(summary["top_rmse_frames"][0]["frame_index"], 1)


if __name__ == "__main__":
    unittest.main()
