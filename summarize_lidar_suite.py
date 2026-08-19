#!/usr/bin/env python3
"""Validate and combine recording-level camera-versus-LiDAR metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MEAN_METRICS = ("mae_m", "abs_rel", "sq_rel", "delta_1", "delta_2", "delta_3")
ROOT_MEAN_SQUARE_METRICS = ("rmse_m", "rmse_log")
CSV_FIELDS = (
    "scope",
    "evaluated_frames",
    "lidar_points",
    "alignment_method",
    "evaluation_role",
    "mae_m",
    "rmse_m",
    "abs_rel",
    "sq_rel",
    "rmse_log",
    "delta_1",
    "delta_2",
    "delta_3",
    "timestamp_dt_min_s",
    "timestamp_dt_max_s",
    "timestamp_dt_mean_s",
    "timestamp_dt_std_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched recording-level LiDAR evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["recording1", "recording2", "recording3", "recording4"],
    )
    parser.add_argument("--sensor", default="ZED_B")
    parser.add_argument("--lidar_sensor", default="E1_A")
    parser.add_argument("--model", default="UniDAC")
    parser.add_argument("--expected_alignment", default="none")
    parser.add_argument("--expected_role", default="metric_evaluation")
    return parser.parse_args()


def summary_path(
    results_root: Path, recording: str, sensor: str, lidar_sensor: str
) -> Path:
    return (
        results_root
        / recording
        / f"{recording}_{sensor}_{lidar_sensor}_lidar_global_metrics.json"
    )


def project_commit(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def load_summaries(
    results_root: Path,
    recordings: list[str],
    sensor: str,
    lidar_sensor: str,
    expected_alignment: str,
    expected_role: str,
) -> list[dict]:
    summaries = []
    for recording in recordings:
        path = summary_path(results_root, recording, sensor, lidar_sensor)
        payload = json.loads(path.read_text())
        expected_values = {
            "recording": recording,
            "camera_sensor": sensor,
            "lidar_sensor": lidar_sensor,
            "alignment_method": expected_alignment,
            "evaluation_role": expected_role,
        }
        for field, expected in expected_values.items():
            actual = payload.get(field)
            if actual != expected:
                raise ValueError(
                    f"{path}: expected {field}={expected!r}, got {actual!r}"
                )
        summaries.append(payload)
    return summaries


def weighted_mean(summaries: list[dict], field: str) -> float:
    total_points = sum(int(item["metrics"]["count"]) for item in summaries)
    return sum(
        float(item["metrics"][field]) * int(item["metrics"]["count"])
        for item in summaries
    ) / total_points


def weighted_root_mean_square(summaries: list[dict], field: str) -> float:
    return math.sqrt(weighted_mean_of_squares(summaries, field))


def weighted_mean_of_squares(summaries: list[dict], field: str) -> float:
    total_points = sum(int(item["metrics"]["count"]) for item in summaries)
    return sum(
        float(item["metrics"][field]) ** 2 * int(item["metrics"]["count"])
        for item in summaries
    ) / total_points


def pooled_timestamp_stats(summaries: list[dict]) -> dict[str, float]:
    """Pool recording means and population standard deviations by frame count."""
    total_frames = sum(int(item["evaluated_frames"]) for item in summaries)
    mean = sum(
        int(item["evaluated_frames"]) * float(item["timestamp_dt_mean_s"])
        for item in summaries
    ) / total_frames
    variance = sum(
        int(item["evaluated_frames"])
        * (
            float(item["timestamp_dt_std_s"]) ** 2
            + (float(item["timestamp_dt_mean_s"]) - mean) ** 2
        )
        for item in summaries
    ) / total_frames
    return {
        "timestamp_dt_min_s": min(
            float(item["timestamp_dt_min_s"]) for item in summaries
        ),
        "timestamp_dt_max_s": max(
            float(item["timestamp_dt_max_s"]) for item in summaries
        ),
        "timestamp_dt_mean_s": mean,
        "timestamp_dt_std_s": math.sqrt(variance),
    }


def recording_row(summary: dict) -> dict:
    metrics = summary["metrics"]
    return {
        "scope": summary["recording"],
        "evaluated_frames": int(summary["evaluated_frames"]),
        "lidar_points": int(metrics["count"]),
        "alignment_method": summary["alignment_method"],
        "evaluation_role": summary["evaluation_role"],
        **{field: float(metrics[field]) for field in (*MEAN_METRICS, *ROOT_MEAN_SQUARE_METRICS)},
        **{
            field: float(summary[field])
            for field in (
                "timestamp_dt_min_s",
                "timestamp_dt_max_s",
                "timestamp_dt_mean_s",
                "timestamp_dt_std_s",
            )
        },
    }


def aggregate_row(summaries: list[dict]) -> dict:
    row = {
        "scope": "all recordings",
        "evaluated_frames": sum(int(item["evaluated_frames"]) for item in summaries),
        "lidar_points": sum(int(item["metrics"]["count"]) for item in summaries),
        "alignment_method": summaries[0]["alignment_method"],
        "evaluation_role": summaries[0]["evaluation_role"],
        **{field: weighted_mean(summaries, field) for field in MEAN_METRICS},
        **{
            field: weighted_root_mean_square(summaries, field)
            for field in ROOT_MEAN_SQUARE_METRICS
        },
        **pooled_timestamp_stats(summaries),
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)


def main() -> None:
    args = parse_args()
    recordings = list(dict.fromkeys(args.recordings))
    summaries = load_summaries(
        args.results_root,
        recordings,
        args.sensor,
        args.lidar_sensor,
        args.expected_alignment,
        args.expected_role,
    )
    rows = [recording_row(summary) for summary in summaries]
    aggregate = aggregate_row(summaries)
    rows.append(aggregate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "perspective_lidar_suite_summary.csv"
    json_path = args.output_dir / "perspective_lidar_suite_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "project_commit": project_commit(Path(__file__).resolve().parent),
                "model": args.model,
                "results_root": str(args.results_root.resolve()),
                "camera_sensor": args.sensor,
                "lidar_sensor": args.lidar_sensor,
                "aggregation": (
                    "Point-weighted means; RMSE and log-RMSE reconstructed from "
                    "point-weighted squared errors. Timestamp moments are "
                    "frame-weighted."
                ),
                "recordings": rows[:-1],
                "all_recordings": aggregate,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validated {len(summaries)} recording summaries")
    print(f"Suite CSV: {csv_path}")
    print(f"Suite JSON: {json_path}")


if __name__ == "__main__":
    main()
