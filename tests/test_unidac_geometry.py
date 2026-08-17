import unittest

from src.depth_models import UniDACDepth


class UniDACCropGeometryTests(unittest.TestCase):
    def test_rectangular_model_input_covers_full_circular_fisheye(self):
        crop_height, crop_width = UniDACDepth._erp_crop_shape(
            canonical_height=1400,
            forward_size=(512, 704),
            crop_wfov=180.0,
            is_fisheye=True,
        )

        self.assertEqual(crop_height, 1400)
        self.assertEqual(crop_width, 1925)
        self.assertAlmostEqual(180.0 * crop_height / 1400, 180.0)
        self.assertAlmostEqual(180.0 * crop_width / 1400, 247.5)

    def test_perspective_crop_keeps_requested_horizontal_fov(self):
        crop_height, crop_width = UniDACDepth._erp_crop_shape(
            canonical_height=1400,
            forward_size=(512, 704),
            crop_wfov=90.0,
            is_fisheye=False,
        )

        self.assertEqual(crop_width, 700)
        self.assertEqual(crop_height, 509)
        self.assertAlmostEqual(180.0 * crop_width / 1400, 90.0)

    def test_crop_rejects_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            UniDACDepth._erp_crop_shape(
                canonical_height=1400,
                forward_size=(0, 704),
                crop_wfov=180.0,
                is_fisheye=True,
            )


if __name__ == "__main__":
    unittest.main()
