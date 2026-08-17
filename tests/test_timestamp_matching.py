import unittest
from pathlib import Path

from src.utils import match_by_timestamp


class TimestampMatchingTests(unittest.TestCase):
    def test_time_offset_is_applied_to_target_timestamp(self):
        source = Path("0000000000_10.100.jpg")
        target = Path("0000000000_10.000.npy")
        self.assertEqual(match_by_timestamp([source], [target], max_dt=0.01), [])
        self.assertEqual(match_by_timestamp([source], [target], max_dt=0.01, time_offset=0.1), [(source, target)])


if __name__ == "__main__":
    unittest.main()
