import unittest

from evaluate_person_tracking import summarize_tracks


class TrackSummaryTests(unittest.TestCase):
    def test_summary_marks_fallback_ids_as_unassigned(self):
        rows = [
            {
                "track_id": 1,
                "tracker_assigned": True,
                "timestamp_seconds": 1.0,
                "pred_median_m": 2.0,
            },
            {
                "track_id": 1,
                "tracker_assigned": True,
                "timestamp_seconds": 2.0,
                "pred_median_m": 1.8,
            },
            {
                "track_id": 1_000_000,
                "tracker_assigned": False,
                "timestamp_seconds": 3.0,
                "pred_median_m": 1.5,
            },
        ]

        summaries = {row["track_id"]: row for row in summarize_tracks(rows)}

        self.assertTrue(summaries[1]["tracker_assigned"])
        self.assertEqual(summaries[1]["detections"], 2)
        self.assertFalse(summaries[1_000_000]["tracker_assigned"])


if __name__ == "__main__":
    unittest.main()
