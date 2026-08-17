import unittest

import numpy as np

from src.lidar_evaluation import align_depth, evaluate_depth


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


if __name__ == "__main__":
    unittest.main()
