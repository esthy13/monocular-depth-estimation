import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from plot_evaluation_diagnostics import (
    binned_medians,
    load_lidar_residual_samples,
    load_person_localization_samples,
    save_lidar_residual_overlay,
    save_localization_error_plot,
)


class EvaluationDiagnosticPlotTests(unittest.TestCase):
    def test_loads_signed_prediction_minus_lidar_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "u_px",
                        "v_px",
                        "lidar_depth_m",
                        "aligned_prediction_m",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "u_px": 10,
                        "v_px": 20,
                        "lidar_depth_m": 4.0,
                        "aligned_prediction_m": 3.5,
                    }
                )

            pixels, residuals = load_lidar_residual_samples(path)

            np.testing.assert_allclose(pixels, [[10.0, 20.0]])
            np.testing.assert_allclose(residuals, [-0.5])

    def test_person_distance_prefers_reference_xyz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person_measurements.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "lidar_x_m",
                        "lidar_y_m",
                        "lidar_z_m",
                        "lidar_median_m",
                        "lidar_3d_error_m",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "lidar_x_m": 3,
                        "lidar_y_m": 4,
                        "lidar_z_m": 0,
                        "lidar_median_m": 99,
                        "lidar_3d_error_m": 0.25,
                    }
                )

            distances, errors = load_person_localization_samples(path)

            np.testing.assert_allclose(distances, [5.0])
            np.testing.assert_allclose(errors, [0.25])

    def test_binned_medians_include_value_on_integer_upper_edge(self):
        centers, medians, counts = binned_medians(
            np.array([0.5, 1.2, 4.0]),
            np.array([0.1, 0.4, 0.8]),
            1.0,
        )

        np.testing.assert_allclose(centers, [0.5, 1.5, 4.5])
        np.testing.assert_allclose(medians, [0.1, 0.4, 0.8])
        np.testing.assert_array_equal(counts, [1, 1, 1])

    def test_plot_commands_create_nonempty_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "rgb.jpg"
            cv2.imwrite(str(image_path), np.full((60, 80, 3), 200, dtype=np.uint8))
            samples_path = root / "samples.csv"
            samples_path.write_text(
                "u_px,v_px,lidar_depth_m,aligned_prediction_m\n"
                "10,20,2.0,1.5\n"
                "30,40,3.0,3.75\n"
            )
            measurements_path = root / "person_measurements.csv"
            measurements_path.write_text(
                "lidar_x_m,lidar_y_m,lidar_z_m,lidar_3d_error_m\n"
                "0,0,2,0.2\n"
                "0,0,4,0.8\n"
            )
            overlay_path = root / "overlay.png"
            localization_path = root / "localization.png"

            save_lidar_residual_overlay(image_path, samples_path, overlay_path)
            save_localization_error_plot(
                [("Model", measurements_path)], localization_path
            )

            self.assertGreater(overlay_path.stat().st_size, 0)
            self.assertGreater(localization_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
