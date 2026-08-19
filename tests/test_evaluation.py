import unittest

import numpy as np

from src.evaluation import (
    fisheye_unit_ray,
    lidar_reference_for_instance,
    paired_depth_metrics,
    project_lidar_to_fisheye,
    project_perspective_depth_to_fisheye,
    select_foreground_reference_mask,
)


class EvaluationGeometryTests(unittest.TestCase):
    def test_central_perspective_depth_reprojects_to_central_fisheye_range(self):
        depth = np.full((3, 3), np.nan, dtype=np.float32)
        depth[1, 1] = 2.0
        camera_matrix = np.array(
            [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
        )

        projected = project_perspective_depth_to_fisheye(
            depth,
            camera_matrix,
            np.eye(4),
            camera_matrix,
            np.zeros(4),
            output_shape=(3, 3),
        )

        self.assertAlmostEqual(float(projected[1, 1]), 2.0)
        self.assertEqual(int(np.isfinite(projected).sum()), 1)

    def test_fisheye_principal_point_is_forward_unit_ray(self):
        camera_matrix = np.array(
            [[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]]
        )

        ray = fisheye_unit_ray((20.0, 30.0), camera_matrix, np.zeros(4))

        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-8)

    def test_forward_lidar_point_projects_to_fisheye_principal_point(self):
        camera_matrix = np.array(
            [[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]]
        )
        pixels, ranges = project_lidar_to_fisheye(
            np.array([[0.0, 0.0, 2.0, 0.5]]),
            np.eye(4),
            camera_matrix,
            np.zeros(4),
            output_shape=(60, 40),
        )

        np.testing.assert_allclose(pixels, [[20.0, 30.0]], atol=1e-6)
        np.testing.assert_allclose(ranges, [2.0], atol=1e-6)

    def test_foreground_reference_selection_removes_far_background_mode(self):
        reference = np.r_[np.full(60, 1.5), np.full(40, 3.0)].reshape(10, 10)
        selected, metadata = select_foreground_reference_mask(
            reference,
            np.ones_like(reference, dtype=bool),
            minimum_samples=20,
        )

        self.assertTrue(metadata["foreground_cluster_used"])
        self.assertEqual(int(selected.sum()), 60)
        self.assertTrue(np.all(reference[selected] == 1.5))

    def test_paired_metrics_use_only_common_valid_pixels(self):
        prediction = np.array([[2.0, 4.0], [np.nan, 1.0]], dtype=np.float32)
        reference = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)

        metrics = paired_depth_metrics(
            prediction, reference, np.ones((2, 2), dtype=bool)
        )

        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["mae_m"], 1.5)
        self.assertAlmostEqual(metrics["abs_rel"], 1.0)

    def test_lidar_reference_rejects_points_inconsistent_with_stereo(self):
        pixels = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
        ranges = np.array([1.5, 3.0], dtype=np.float32)
        instance = np.ones((5, 5), dtype=bool)
        stereo = np.full((5, 5), np.nan, dtype=np.float32)
        stereo[1, 1] = 1.6
        stereo[3, 3] = 1.5

        result = lidar_reference_for_instance(
            pixels,
            ranges,
            instance,
            dense_reference=stereo,
            consistency_tolerance_metres=0.3,
            minimum_points=1,
        )

        self.assertTrue(result["valid"])
        self.assertTrue(result["stereo_consistency_used"])
        self.assertEqual(result["points_in_mask"], 2)
        self.assertEqual(result["consistent_points"], 1)
        self.assertAlmostEqual(result["pixel_median_u"], 1.0)
        self.assertAlmostEqual(result["pixel_median_v"], 1.0)
        self.assertAlmostEqual(result["median_m"], 1.5)


if __name__ == "__main__":
    unittest.main()
