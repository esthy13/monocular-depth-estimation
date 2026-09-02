import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.lidar_evaluation import align_depth, evaluate_depth, save_point_samples


class LeastSquaresAlignmentTests(unittest.TestCase):
    def test_scale_only_alignment_is_applied_before_metrics(self):
        predicted = np.array([1.0, 2.0, 4.0])
        ground_truth = predicted * 3.5

        aligned, scale, shift = align_depth(predicted, ground_truth, "least_squares")
        result = evaluate_depth(predicted, ground_truth, "least_squares")

        np.testing.assert_allclose(aligned, ground_truth)
        self.assertAlmostEqual(scale, 3.5)
        self.assertEqual(shift, 0.0)
        self.assertAlmostEqual(result.abs_rel, 0.0)

    def test_alignment_ignores_invalid_pairs_when_fitting_metrics(self):
        predicted = np.array([1.0, 2.0, np.nan, 0.0])
        ground_truth = np.array([2.0, 4.0, 99.0, 99.0])

        result = evaluate_depth(predicted, ground_truth, "least_squares")

        self.assertEqual(result.count, 2)
        self.assertAlmostEqual(result.scale, 2.0)
        self.assertAlmostEqual(result.abs_rel, 0.0)

    def test_inverse_depth_alignment_does_not_invert_raw_near_zero_values(self):
        raw_inverse_depth = np.array([0.1, 0.2, 0.4, 1e-10])
        ground_truth = 1.0 / (2.0 * raw_inverse_depth + 0.05)

        result = evaluate_depth(raw_inverse_depth, ground_truth, "inverse_least_squares")

        self.assertAlmostEqual(result.scale, 2.0)
        self.assertAlmostEqual(result.shift, 0.05)
        self.assertAlmostEqual(result.abs_rel, 0.0)

    def test_inverse_alignment_rejects_degenerate_predictions(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            align_depth(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]), "inverse_least_squares")

    def test_inverse_alignment_marks_nonpositive_fitted_inverse_depth_invalid(self):
        raw = np.array([1.0, 2.0, 3.0])
        gt = 1.0 / (raw - 0.5)
        aligned, scale, shift = align_depth(raw, gt, "inverse_least_squares")
        self.assertGreater(scale, 0)
        self.assertAlmostEqual(shift, -0.5)
        np.testing.assert_allclose(aligned, gt)

    def test_point_samples_include_signed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            save_point_samples(
                path,
                np.array([[10.0, 20.0]]),
                np.array([4.0]),
                np.array([3.0]),
                np.array([3.5]),
            )

            with path.open(newline="") as file:
                row = next(csv.DictReader(file))
            self.assertAlmostEqual(float(row["signed_error_m"]), -0.5)
            self.assertAlmostEqual(float(row["absolute_error_m"]), 0.5)


if __name__ == "__main__":
    unittest.main()
