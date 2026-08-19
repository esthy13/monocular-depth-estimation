import unittest
from pathlib import Path

from project_lidar_fisheye import selected_indices, validate_existing_metadata


class ProjectLidarFisheyeTests(unittest.TestCase):
    def test_single_frame_selection(self):
        self.assertEqual(selected_indices(10, 4, False, 1, 0), [4])

    def test_all_frame_selection_honours_step_and_limit(self):
        self.assertEqual(selected_indices(10, 0, True, 3, 3), [0, 3, 6])

    def test_invalid_frame_index_is_rejected(self):
        with self.assertRaises(IndexError):
            selected_indices(3, 3, False, 1, 0)

    def test_invalid_frame_step_is_rejected(self):
        with self.assertRaises(ValueError):
            selected_indices(3, 0, True, 0, 0)

    def test_existing_output_must_use_the_same_protocol(self):
        with self.assertRaisesRegex(ValueError, "rerun with --overwrite"):
            validate_existing_metadata(
                {"lidar_sensors": ["E1_A"]},
                {"lidar_sensors": ["E1_A", "E1_B"]},
                Path("metadata.json"),
            )


if __name__ == "__main__":
    unittest.main()
