import unittest

import numpy as np

from calibrate_extrinsics import ransac_rigid, transform_points as calibrate_transform_points
from src.lidar_evaluation import project_perspective_points
from src.scientific_evaluation import (
    AlignmentParameters,
    ValidityRules,
    apply_inverse_alignment,
    calculate_metrics,
    deterministic_calibration_split,
    evaluate_mode,
    fit_fixed_inverse_alignment,
    rejection_counts,
    stable_configuration_hash,
)
from src.utils import camera_from_lidar_transform


class ScientificEvaluationTests(unittest.TestCase):
    def test_transform_composition_sensor_to_reference(self):
        camera = np.eye(4); camera[0, 3] = 2
        lidar = np.eye(4); lidar[1, 3] = 3
        result = camera_from_lidar_transform({"cam": camera.tolist(), "lidar": lidar.tolist()}, "cam", "lidar")
        np.testing.assert_allclose(result, np.linalg.inv(camera) @ lidar)

    def test_projection_uses_positive_camera_z(self):
        camera = {"K": [[100, 0, 50], [0, 100, 40], [0, 0, 1]], "dist": []}
        pixels, depth, indices = project_perspective_points(np.array([[0, 0, 2], [0, 0, -2], [2, 0, 1]]), camera, (80, 100))
        np.testing.assert_allclose(pixels, [[50, 40]])
        np.testing.assert_allclose(depth, [2]); np.testing.assert_array_equal(indices, [0])

    def test_fixed_alignment_recovers_synthetic_parameters(self):
        raw = np.linspace(.1, 4, 100); gt = 1 / (.3 * raw + .04)
        fitted = fit_fixed_inverse_alignment(raw, gt, "cal", [1, 2])
        self.assertAlmostEqual(fitted.scale, .3); self.assertAlmostEqual(fitted.shift, .04)
        np.testing.assert_allclose(apply_inverse_alignment(raw, fitted), gt)

    def test_split_is_deterministic_disjoint_and_nonempty(self):
        first = deterministic_calibration_split(list(range(20)), .2, 7)
        second = deterministic_calibration_split(list(range(20)), .2, 7)
        self.assertEqual(first, second); self.assertFalse(set(first[0]) & set(first[1])); self.assertTrue(first[0]); self.assertTrue(first[1])

    def test_metrics_match_definitions(self):
        metrics = calculate_metrics(np.array([1., 4.]), np.array([1., 2.]))
        self.assertEqual(metrics["count"], 2); self.assertAlmostEqual(metrics["mae_m"], 1.0)
        self.assertAlmostEqual(metrics["rmse_m"], np.sqrt(2)); self.assertAlmostEqual(metrics["abs_rel"], .5)

    def test_invalid_reasons_are_mutually_exclusive(self):
        raw=np.array([1,np.nan,0,2]);gt=np.array([1,1,1,30.]);valid,reasons=rejection_counts(raw,gt,ValidityRules())
        np.testing.assert_array_equal(valid,[True,False,False,False]);self.assertEqual(sum(v for k,v in reasons.items() if k!="accepted_input"),3)

    def test_ill_conditioned_fit_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"ill-conditioned"):
            fit_fixed_inverse_alignment(np.ones(10),np.arange(1,11),"cal",[0])

    def test_outlier_is_reported_not_removed_as_a_frame(self):
        params=AlignmentParameters(1,0,1,3,"cal",(0,));frame={"recording":"test","frame":56,"raw_prediction":np.array([1.,1.,1.]),"ground_truth_m":np.array([1.,1.,1000.]),"boundary":np.zeros(3,bool),"timestamp_delta_s":0.}
        result=evaluate_mode([frame],"fixed_held_out",ValidityRules(max_prediction_m=2000,min_correspondences_per_frame=2),params)
        self.assertEqual(result["valid_frame_count"],1);self.assertEqual(result["excluded_frames"],[]);self.assertGreater(result["pooled_metrics"]["rmse_m"],500)

    def test_ransac_rigid_recovers_transform_with_outlier(self):
        rng=np.random.default_rng(2);source=rng.normal(size=(30,3));rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]]);target=source@rotation.T+np.array([1,2,3]);target[-1]=100
        transform,inliers=ransac_rigid(source,target,.01,500,3);self.assertEqual(inliers.sum(),29);np.testing.assert_allclose(calibrate_transform_points(source[:-1],transform),target[:-1],atol=1e-10)

    def test_configuration_hash_ignores_dictionary_order(self):
        self.assertEqual(stable_configuration_hash({"a":1,"b":2}),stable_configuration_hash({"b":2,"a":1}))


if __name__ == "__main__":
    unittest.main()
