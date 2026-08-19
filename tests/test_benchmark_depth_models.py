import unittest

from benchmark_depth_models import (
    build_comparison,
    markdown_report,
    percentile,
    summarize_model_timings,
    validate_frame_indices,
)


class BenchmarkDepthModelsTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

    def test_frame_indices_are_unique_and_validated(self):
        self.assertEqual(validate_frame_indices([0, 2, 2, 4], 5), [0, 2, 4])
        with self.assertRaises(ValueError):
            validate_frame_indices([5], 5)

    def test_summary_and_comparison_use_all_repetitions(self):
        rows = [
            {"model": "DAC", "time_ms": value}
            for value in (8.0, 10.0, 12.0)
        ] + [
            {"model": "UniDAC", "time_ms": value}
            for value in (18.0, 20.0, 22.0)
        ]
        summaries = [
            summarize_model_timings(model, rows)
            for model in ("DAC", "UniDAC")
        ]
        comparison = build_comparison(summaries)

        self.assertEqual(summaries[0]["measurements"], 3)
        self.assertEqual(summaries[0]["median_ms"], 10.0)
        self.assertEqual(comparison["faster_model"], "DAC")
        self.assertEqual(comparison["median_speedup_ratio"], 2.0)

    def test_markdown_contains_protocol_and_result(self):
        summary = {
            "protocol": {
                "recording": "recording1",
                "sensor": "G1_A",
                "frame_indices": [0],
                "warmup_runs": 2,
                "timed_runs_per_frame": 3,
                "timing_scope": "predict",
            },
            "environment": {
                "gpu": "Test GPU",
                "torch_version": "1.0",
                "cuda_version": "1.0",
            },
            "models": [
                {
                    "model": "DAC", "measurements": 3, "mean_ms": 10.0,
                    "median_ms": 10.0, "p10_ms": 9.0, "p90_ms": 11.0,
                    "throughput_fps_from_median": 100.0,
                },
                {
                    "model": "UniDAC", "measurements": 3, "mean_ms": 20.0,
                    "median_ms": 20.0, "p10_ms": 19.0, "p90_ms": 21.0,
                    "throughput_fps_from_median": 50.0,
                },
            ],
            "comparison": {
                "faster_model": "DAC",
                "slower_model": "UniDAC",
                "median_speedup_ratio": 2.0,
                "median_latency_reduction_percent": 50.0,
            },
        }

        report = markdown_report(summary)

        self.assertIn("same GPU", report)
        self.assertIn("DAC is 2.000× faster", report)


if __name__ == "__main__":
    unittest.main()
