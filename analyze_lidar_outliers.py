#!/usr/bin/env python3
"""Rank perspective LiDAR-evaluation frames and test a blur/error hypothesis."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr


FRAME_PATTERN = re.compile(r"_(\d{4})_lidar_metrics\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-frame LiDAR metrics, rank outliers, and measure whether "
            "lower image sharpness is associated with higher error."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["recording1", "recording2", "recording3", "recording4"],
    )
    parser.add_argument("--sensor", default="ZED_B")
    parser.add_argument("--lidar_sensor", default="E1_A")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--person_blur",
        action="store_true",
        help="Run person segmentation and measure sharpness inside person masks.",
    )
    parser.add_argument("--detector_weights", default="yolo26n-seg.pt")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def frame_index(path: Path) -> int:
    match = FRAME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot read a frame index from {path.name}")
    return int(match.group(1))


def laplacian_variance(image: np.ndarray) -> float:
    """Return a simple global sharpness score; lower values indicate more blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def masked_laplacian_variance(image: np.ndarray, mask: np.ndarray) -> float | None:
    """Measure sharpness inside an eroded region without counting its boundary."""
    mask = mask.astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    selected = eroded.astype(bool) if eroded.any() else mask.astype(bool)
    if np.count_nonzero(selected) < 2:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian[selected].var())


def project_commit(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def load_rows(
    data_dir: Path,
    results_root: Path,
    recordings: list[str],
    sensor: str,
    lidar_sensor: str,
    person_tracker=None,
) -> list[dict]:
    rows: list[dict] = []
    for recording in recordings:
        result_dir = results_root / recording
        pattern = f"{recording}_{sensor}_[0-9][0-9][0-9][0-9]_lidar_metrics.json"
        paths = sorted(result_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No per-frame metrics found under {result_dir}")

        for path in paths:
            payload = json.loads(path.read_text())
            if payload.get("lidar_sensor") != lidar_sensor:
                continue
            index = frame_index(path)
            image_file = Path(payload["image"]).name
            image_path = data_dir / recording / "data" / sensor / sensor / image_file
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"OpenCV could not read {image_path}")
            metrics = payload["metrics"]
            mae = float(metrics["mae_m"])
            rmse = float(metrics["rmse_m"])
            person_count = 0
            person_mask_pixels = 0
            person_sharpness = None
            if person_tracker is not None:
                detections, _speed = person_tracker.track(image)
                person_count = len(detections)
                if detections:
                    person_mask = np.logical_or.reduce(
                        [detection.mask for detection in detections]
                    )
                    person_mask_pixels = int(np.count_nonzero(person_mask))
                    person_sharpness = masked_laplacian_variance(image, person_mask)
            rows.append(
                {
                    "recording": recording,
                    "frame_index": index,
                    "image_file": image_file,
                    "image_path": str(image_path.resolve()),
                    "lidar_points": int(metrics["count"]),
                    "mae_m": mae,
                    "rmse_m": rmse,
                    "rmse_mae_ratio": rmse / mae if mae > 0 else None,
                    "abs_rel": float(metrics["abs_rel"]),
                    "delta_1": float(metrics["delta_1"]),
                    "delta_2": float(metrics["delta_2"]),
                    "delta_3": float(metrics["delta_3"]),
                    "laplacian_variance": laplacian_variance(image),
                    "person_detections": person_count,
                    "person_mask_pixels": person_mask_pixels,
                    "person_laplacian_variance": person_sharpness,
                }
            )
    return rows


def finite_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def compact_row(row: dict) -> dict:
    keys = (
        "recording",
        "frame_index",
        "image_file",
        "image_path",
        "lidar_points",
        "mae_m",
        "rmse_m",
        "rmse_mae_ratio",
        "abs_rel",
        "laplacian_variance",
        "person_detections",
        "person_mask_pixels",
        "person_laplacian_variance",
    )
    return {key: row[key] for key in keys}


def summarize_rows(rows: list[dict], top_k: int) -> dict:
    blur = np.asarray([row["laplacian_variance"] for row in rows], dtype=float)
    mae = np.asarray([row["mae_m"] for row in rows], dtype=float)
    rmse = np.asarray([row["rmse_m"] for row in rows], dtype=float)
    ratio = np.asarray([row["rmse_mae_ratio"] for row in rows], dtype=float)
    low_blur = blur <= np.quantile(blur, 0.25)
    high_rmse = rmse >= np.quantile(rmse, 0.75)
    overlap = low_blur & high_rmse

    person_blur = np.asarray(
        [
            row["person_laplacian_variance"]
            if row["person_laplacian_variance"] is not None
            else np.nan
            for row in rows
        ],
        dtype=float,
    )
    person_valid = np.isfinite(person_blur)
    person_overlap = np.zeros(len(rows), dtype=bool)
    if np.count_nonzero(person_valid) >= 4:
        low_person_blur_threshold = np.quantile(person_blur[person_valid], 0.25)
        high_person_rmse_threshold = np.quantile(rmse[person_valid], 0.75)
        person_overlap = (
            person_valid
            & (person_blur <= low_person_blur_threshold)
            & (rmse >= high_person_rmse_threshold)
        )

    return {
        "frames": len(rows),
        "median_laplacian_variance": float(np.median(blur)),
        "median_mae_m": float(np.median(mae)),
        "median_rmse_m": float(np.median(rmse)),
        "median_rmse_mae_ratio": float(np.median(ratio)),
        "sharpness_vs_mae_spearman": finite_correlation(blur, mae),
        "sharpness_vs_rmse_spearman": finite_correlation(blur, rmse),
        "low_sharpness_and_high_rmse_frames": int(np.count_nonzero(overlap)),
        "low_sharpness_and_high_rmse_fraction": float(np.mean(overlap)),
        "person_frames": int(np.count_nonzero(person_valid)),
        "median_person_laplacian_variance": (
            float(np.median(person_blur[person_valid]))
            if np.any(person_valid)
            else None
        ),
        "person_sharpness_vs_mae_spearman": finite_correlation(person_blur, mae),
        "person_sharpness_vs_rmse_spearman": finite_correlation(person_blur, rmse),
        "low_person_sharpness_and_high_rmse_frames": int(
            np.count_nonzero(person_overlap)
        ),
        "low_person_sharpness_and_high_rmse_fraction_of_person_frames": (
            float(np.count_nonzero(person_overlap) / np.count_nonzero(person_valid))
            if np.any(person_valid)
            else None
        ),
        "top_rmse_frames": [
            compact_row(row)
            for row in sorted(rows, key=lambda row: row["rmse_m"], reverse=True)[
                :top_k
            ]
        ],
        "top_rmse_mae_ratio_frames": [
            compact_row(row)
            for row in sorted(
                rows, key=lambda row: row["rmse_mae_ratio"], reverse=True
            )[:top_k]
        ],
        "lowest_sharpness_frames": [
            compact_row(row)
            for row in sorted(rows, key=lambda row: row["laplacian_variance"])[
                :top_k
            ]
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1")
    person_tracker = None
    if args.person_blur:
        from src.person_tracker import YOLOPersonTracker

        person_tracker = YOLOPersonTracker(
            weights=args.detector_weights,
            device=(int(args.device) if args.device and args.device.isdigit() else args.device),
        )
    rows = load_rows(
        args.data_dir,
        args.results_root,
        args.recordings,
        args.sensor,
        args.lidar_sensor,
        person_tracker,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "perspective_lidar_frame_analysis.csv"
    json_path = args.output_dir / "perspective_lidar_outlier_summary.json"
    write_csv(csv_path, rows)

    by_recording = {
        recording: summarize_rows(
            [row for row in rows if row["recording"] == recording], args.top_k
        )
        for recording in args.recordings
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": project_commit(Path(__file__).resolve().parent),
        "data_dir": str(args.data_dir.resolve()),
        "results_root": str(args.results_root.resolve()),
        "sensor": args.sensor,
        "lidar_sensor": args.lidar_sensor,
        "blur_metric": (
            "Variance of the grayscale Laplacian over the complete image; lower "
            "values indicate less high-frequency detail. This is a blur proxy, "
            "not a person-specific blur annotation."
        ),
        "person_blur_metric": (
            "Variance of the grayscale Laplacian inside the eroded union of "
            "YOLO person masks. Lower values indicate less high-frequency "
            "detail on detected people. Frames without a detection are excluded."
            if args.person_blur
            else "Not computed; rerun with --person_blur."
        ),
        "interpretation": (
            "A negative sharpness/error Spearman rho supports an association "
            "between blur and higher error. It does not establish causality."
        ),
        "all_recordings": summarize_rows(rows, args.top_k),
        "recordings": by_recording,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Analyzed {len(rows)} frames")
    print(f"Frame table: {csv_path}")
    print(f"Outlier summary: {json_path}")


if __name__ == "__main__":
    main()
