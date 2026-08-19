import unittest

from src.depth_models import DepthAnyCamera, UniDACDepth


class DACCropGeometryTests(unittest.TestCase):
    def test_official_indoor_input_covers_full_circular_fisheye(self):
        crop_height, crop_width = DepthAnyCamera._erp_crop_shape(
            canonical_height=1400,
            forward_size=(500, 750),
            crop_wfov=180.0,
            is_fisheye=True,
        )

        self.assertEqual((crop_height, crop_width), (1400, 2100))
        self.assertAlmostEqual(180.0 * crop_width / 1400, 270.0)

    def test_square_dac_input_covers_full_circular_fisheye(self):
        crop_height, crop_width = DepthAnyCamera._erp_crop_shape(
            canonical_height=1400,
            forward_size=(700, 700),
            crop_wfov=180.0,
            is_fisheye=True,
        )

        self.assertEqual((crop_height, crop_width), (1400, 1400))

    def test_camera_can_change_without_reloading_weights(self):
        first_camera = {"camera_model": "OPENCV_FISHEYE", "fx": 1.0}
        second_camera = {"camera_model": "PINHOLE", "fx": 2.0}
        model = DepthAnyCamera(cam_params=first_camera)
        model._grid_cache[(10, 10)] = object()
        model.last_projection_metadata = {"old": True}

        model.set_camera(second_camera, 92.0)

        self.assertEqual(model._cam_params, second_camera)
        self.assertEqual(model._crop_wfov, 92.0)
        self.assertEqual(model._grid_cache, {})
        self.assertEqual(model.last_projection_metadata, {})


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
