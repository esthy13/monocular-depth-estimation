import unittest

import numpy as np

from src.utils import colorize_depth, create_common_valid_depth_mask


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


if __name__ == "__main__":
    unittest.main()
