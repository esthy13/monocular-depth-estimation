import unittest
from unittest.mock import patch

from run_depth import parse_args


class RunDepthCliTests(unittest.TestCase):
    def test_device_defaults_to_auto(self):
        with patch("sys.argv", ["run_depth.py", "--data_dir", "/tmp/data"]):
            self.assertEqual(parse_args().device, "auto")

    def test_apple_metal_device_can_be_selected(self):
        with patch(
            "sys.argv",
            ["run_depth.py", "--data_dir", "/tmp/data", "--device", "mps"],
        ):
            self.assertEqual(parse_args().device, "mps")


if __name__ == "__main__":
    unittest.main()
