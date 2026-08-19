import unittest

import numpy as np

from src.utils import (
    colorize_depth,
    create_common_valid_depth_mask,
    depth_visualization_limits,
    depth_visualization_scale_label,
)


class DepthVisualizationTests(unittest.TestCase):
    def test_invalid_pixels_are_gray_not_far_depth_black(self):
        depth = np.array([[1.0, 10.0], [4.0, np.nan]], dtype=np.float32)
        valid_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)

        colored = colorize_depth(
            depth,
            valid_mask=valid_mask,
            invert=True,
            invalid_color="#808080",
        )

        np.testing.assert_array_equal(colored[1, 0], [128, 128, 128])
        np.testing.assert_array_equal(colored[1, 1], [128, 128, 128])
        self.assertFalse(np.array_equal(colored[0, 1], [128, 128, 128]))

    def test_common_valid_mask_intersects_depths_and_sensor_masks(self):
        model_depth = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        sensor_depth = np.array([[1.1, 0.0], [3.0, 4.1]], dtype=np.float32)
        lens_mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)

        common = create_common_valid_depth_mask(
            model_depth,
            sensor_depth,
            masks=[lens_mask],
        )

        np.testing.assert_array_equal(common, [[1, 0], [0, 0]])

    def test_all_invalid_depth_is_rendered_gray(self):
        depth = np.full((2, 3), np.nan, dtype=np.float32)

        colored = colorize_depth(depth, invalid_color="#808080")

        expected = np.full((2, 3, 3), 128, dtype=np.uint8)
        np.testing.assert_array_equal(colored, expected)

    def test_fixed_visualization_range_is_independent_of_image_values(self):
        depth = np.array([[1.0, 3.0], [8.0, np.nan]], dtype=np.float32)

        limits = depth_visualization_limits(depth, value_range=(0.5, 10.0))

        self.assertEqual(limits, (0.5, 10.0))

    def test_automatic_visualization_range_uses_only_valid_pixels(self):
        depth = np.array([[1.0, 3.0], [100.0, 9.0]], dtype=np.float32)
        valid_mask = np.array([[1, 1], [0, 1]], dtype=np.uint8)

        limits = depth_visualization_limits(
            depth,
            valid_mask=valid_mask,
            robust_percentiles=(0.0, 100.0),
        )

        self.assertEqual(limits, (1.0, 9.0))

    def test_invalid_fixed_visualization_range_is_rejected(self):
        with self.assertRaises(ValueError):
            depth_visualization_limits(
                np.ones((2, 2), dtype=np.float32), value_range=(5.0, 5.0)
            )

    def test_fixed_metric_scale_label_states_exact_range(self):
        label = depth_visualization_scale_label(
            0.5,
            10.0,
            depth_unit="m",
            fixed_range=True,
            quantity_label="Metric depth",
        )

        self.assertEqual(
            label,
            "Metric depth (m)\nFixed display range: 0.5 to 10 m",
        )

    def test_relative_scale_label_states_per_image_range(self):
        label = depth_visualization_scale_label(
            0.44781,
            5.7986,
            depth_unit="a.u.",
            robust_percentiles=(2.0, 98.0),
            quantity_label="Relative inverse depth",
        )

        self.assertEqual(
            label,
            (
                "Relative inverse depth (a.u.)\n"
                "Per-image display range (2-98%): 0.4478 to 5.799 a.u."
            ),
        )


if __name__ == "__main__":
    unittest.main()
