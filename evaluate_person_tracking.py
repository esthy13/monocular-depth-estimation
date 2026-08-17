#!/usr/bin/env python3
"""Evaluate tracked G1_A people against calibrated ZED and LiDAR references."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.evaluation import (
    depth_statistics,
    erode_instance_mask,
    fisheye_unit_ray,
    lidar_reference_for_instance,
    mask_centroid,
    nearest_timestamp_file,
    paired_depth_metrics,
    point_from_euclidean_depth,
    project_lidar_to_fisheye,
    project_perspective_depth_to_fisheye,
    select_foreground_reference_mask,
)
from src.person_tracker import YOLOPersonTracker
from src.utils import find_rgb_images, load_extrinsics, load_intrinsics, parse_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track people in G1_A, localize them with metric monocular depth, "
            "and compare with synchronized ZED/LiDAR measurements."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--depth_output_dir",
        type=Path,
        required=True,
        help="UniDAC output root containing <recording>/G1_A/*_depth_raw.npy.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--recording", default="recording1")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum frames to evaluate; 0 evaluates every available prediction.",
    )
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument(
        "--prediction_label",
        default=None,
        help=(
            "Name shown for the monocular prediction. By default it is read "
            "from each prediction metadata file."
        ),
    )
    parser.add_argument("--detector_weights", default="yolo26n-seg.pt")
    parser.add_argument("--detector_confidence", type=float, default=0.25)
    parser.add_argument("--detector_iou", type=float, default=0.7)
    parser.add_argument("--detector_image_size", type=int, default=640)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0 or cpu. Omit for automatic selection.",
    )
    parser.add_argument("--zed_max_dt", type=float, default=0.20)
    parser.add_argument("--lidar_max_dt", type=float, default=0.05)
    parser.add_argument("--lidar_consistency", type=float, default=0.30)
    parser.add_argument("--minimum_lidar_points", type=int, default=10)
    return parser.parse_args()


def parse_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def predicted_depth_path(
    depth_output_dir: Path,
    recording: str,
    image_index: int,
) -> Path:
    return (
        depth_output_dir
        / recording
        / "G1_A"
        / f"{recording}_G1_A_{image_index:06d}_depth_raw.npy"
    )


def prediction_metadata_path(depth_path: Path) -> Path:
    return depth_path.with_name(depth_path.name.replace("_depth_raw.npy", "_metadata.json"))


def numeric_color(track_id: int) -> tuple[int, int, int]:
    return (
        int(60 + (track_id * 67) % 180),
        int(60 + (track_id * 109) % 180),
        int(60 + (track_id * 149) % 180),
    )


def annotate_detection(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    track_id: int,
    confidence: float,
    prediction_label: str,
    prediction_m: float | None,
    zed_m: float | None,
    lidar_m: float | None,
) -> None:
    color = numeric_color(track_id)
    overlay = image.copy()
    overlay[mask.astype(bool)] = color
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0, dst=image)

    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    parts = [f"ID {track_id}", f"conf {confidence:.2f}"]
    if prediction_m is not None:
        parts.append(f"{prediction_label} {prediction_m:.2f}m")
    if zed_m is not None:
        parts.append(f"ZED {zed_m:.2f}m")
    if lidar_m is not None:
        parts.append(f"LiDAR {lidar_m:.2f}m")
    label = " | ".join(parts)
    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    top = max(0, y1 - text_size[1] - baseline - 8)
    cv2.rectangle(
        image,
        (x1, top),
        (min(image.shape[1] - 1, x1 + text_size[0] + 8), y1),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 4, y1 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def prefixed(target: dict, prefix: str, values: dict) -> None:
    for key, value in values.items():
        target[f"{prefix}_{key}"] = value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    preferred = [
        "recording",
        "image_index",
        "image_file",
        "timestamp_seconds",
        "track_id",
        "detector_confidence",
    ]
    available = {key for row in rows for key in row}
    keys = [key for key in preferred if key in available]
    keys += sorted(available - set(keys))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_tracks(rows: list[dict]) -> list[dict]:
    tracks: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        tracks[int(row["track_id"])].append(row)

    summaries: list[dict] = []
    for track_id, track_rows in sorted(tracks.items()):
        output: dict = {
            "track_id": track_id,
            "detections": len(track_rows),
            "first_timestamp_seconds": min(row["timestamp_seconds"] for row in track_rows),
            "last_timestamp_seconds": max(row["timestamp_seconds"] for row in track_rows),
        }
        for key in [
            "pred_median_m",
            "zed_reference_median_m",
            "zed_mae_m",
            "zed_rmse_m",
            "zed_abs_rel",
            "zed_3d_error_m",
            "lidar_median_m",
            "lidar_3d_error_m",
        ]:
            values = [row.get(key) for row in track_rows if row.get(key) is not None]
            output[f"mean_{key}"] = float(np.mean(values)) if values else None
            output[f"median_{key}"] = float(np.median(values)) if values else None
        summaries.append(output)
    return summaries


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame_step must be at least 1")

    intrinsics = load_intrinsics(args.data_dir / "intrinsic.json")
    extrinsics = load_extrinsics(args.data_dir / "extrinsics.json")
    if extrinsics.get("G1_A") is not None:
        raise ValueError("Expected G1_A to be the reference sensor")

    images = find_rgb_images(args.data_dir, "G1_A", args.recording)
    zed_files = sorted(
        (args.data_dir / args.recording / "data" / "ZED_B_depth" / "ZED_B_depth").glob(
            "*.npy"
        )
    )
    lidar_files = {
        sensor: sorted(
            (args.data_dir / args.recording / "data" / sensor / sensor).glob("*.npy")
        )
        for sensor in ("E1_A", "E1_B")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    tracker = YOLOPersonTracker(
        weights=args.detector_weights,
        confidence=args.detector_confidence,
        iou=args.detector_iou,
        image_size=args.detector_image_size,
        tracker=args.tracker,
        device=parse_device(args.device),
    )

    g1 = intrinsics["G1_A"]
    target_k = np.asarray(g1["K"], dtype=np.float64)
    target_distortion = np.asarray(g1["dist"], dtype=np.float64)
    zed_k = np.asarray(intrinsics["ZED_B"]["K"], dtype=np.float64)
    zed_transform = np.asarray(extrinsics["ZED_B"], dtype=np.float64)

    indices = list(range(args.start_index, len(images), args.frame_step))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    rows: list[dict] = []
    missing_predictions = 0
    processed_frames = 0
    detector_speeds: list[float] = []
    depth_inference_times: list[float] = []
    prediction_models: set[str] = set()
    project_commits: set[str] = set()

    for image_index in indices:
        image_path = images[image_index]
        depth_path = predicted_depth_path(
            args.depth_output_dir, args.recording, image_index
        )
        if not depth_path.is_file():
            missing_predictions += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")
        prediction = np.load(depth_path).squeeze().astype(np.float32)
        if prediction.shape != image.shape[:2]:
            raise ValueError(
                f"Prediction {depth_path} has shape {prediction.shape}; "
                f"expected {image.shape[:2]}"
            )

        metadata_path = prediction_metadata_path(depth_path)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if metadata.get("project_commit"):
            project_commits.add(str(metadata["project_commit"]))
        prediction_label = str(
            args.prediction_label or metadata.get("model") or "Prediction"
        )
        prediction_models.add(prediction_label)
        depth_inference_ms = metadata.get("median_time_ms")
        if depth_inference_ms is not None:
            depth_inference_times.append(float(depth_inference_ms))
        timestamp = float(metadata.get("timestamp_seconds") or parse_timestamp(image_path))

        detections, speed = tracker.track(image)
        if speed.get("inference") is not None:
            detector_speeds.append(float(speed["inference"]))
        processed_frames += 1
        annotated = image.copy()

        zed_match = nearest_timestamp_file(zed_files, timestamp, args.zed_max_dt)
        zed_reference = None
        if zed_match is not None:
            zed_reference = project_perspective_depth_to_fisheye(
                np.load(zed_match.path),
                zed_k,
                zed_transform,
                target_k,
                target_distortion,
                image.shape[:2],
            )

        lidar_matches = {
            sensor: nearest_timestamp_file(
                lidar_files[sensor], timestamp, args.lidar_max_dt
            )
            for sensor in ("E1_A", "E1_B")
        }
        lidar_pixels: list[np.ndarray] = []
        lidar_ranges: list[np.ndarray] = []
        for sensor, match in lidar_matches.items():
            if match is None:
                continue
            pixels, ranges = project_lidar_to_fisheye(
                np.load(match.path),
                np.asarray(extrinsics[sensor], dtype=np.float64),
                target_k,
                target_distortion,
                image.shape[:2],
            )
            lidar_pixels.append(pixels)
            lidar_ranges.append(ranges)
        combined_lidar_pixels = (
            np.concatenate(lidar_pixels, axis=0)
            if lidar_pixels
            else np.empty((0, 2), dtype=np.float32)
        )
        combined_lidar_ranges = (
            np.concatenate(lidar_ranges)
            if lidar_ranges
            else np.empty(0, dtype=np.float32)
        )

        for detection in detections:
            sampling_mask = erode_instance_mask(
                detection.mask, detection.bbox_xyxy
            )
            if not sampling_mask.any():
                sampling_mask = detection.mask.astype(bool)
            prediction_stats = depth_statistics(prediction, sampling_mask)
            centroid = mask_centroid(sampling_mask)
            predicted_median = prediction_stats["median_m"]
            predicted_point = (
                point_from_euclidean_depth(
                    centroid, predicted_median, target_k, target_distortion
                )
                if predicted_median is not None
                else np.full(3, np.nan)
            )

            row: dict = {
                "recording": args.recording,
                "image_index": image_index,
                "image_file": image_path.name,
                "timestamp_seconds": timestamp,
                "track_id": detection.track_id,
                "detector_confidence": detection.confidence,
                "prediction_model": prediction_label,
                "depth_inference_ms": float(depth_inference_ms)
                if depth_inference_ms is not None
                else None,
                "bbox_x1": detection.bbox_xyxy[0],
                "bbox_y1": detection.bbox_xyxy[1],
                "bbox_x2": detection.bbox_xyxy[2],
                "bbox_y2": detection.bbox_xyxy[3],
                "mask_pixels": int(detection.mask.sum()),
                "sampling_mask_pixels": int(sampling_mask.sum()),
                "centroid_u": centroid[0],
                "centroid_v": centroid[1],
                "pred_x_m": float(predicted_point[0]),
                "pred_y_m": float(predicted_point[1]),
                "pred_z_m": float(predicted_point[2]),
                "zed_file": zed_match.path.name if zed_match else None,
                "zed_dt_ms": zed_match.delta_seconds * 1000 if zed_match else None,
                "E1_A_file": lidar_matches["E1_A"].path.name
                if lidar_matches["E1_A"]
                else None,
                "E1_A_dt_ms": lidar_matches["E1_A"].delta_seconds * 1000
                if lidar_matches["E1_A"]
                else None,
                "E1_B_file": lidar_matches["E1_B"].path.name
                if lidar_matches["E1_B"]
                else None,
                "E1_B_dt_ms": lidar_matches["E1_B"].delta_seconds * 1000
                if lidar_matches["E1_B"]
                else None,
            }
            prefixed(row, "pred", prediction_stats)

            zed_median = None
            lidar_dense_reference = None
            if zed_reference is not None:
                common = (
                    sampling_mask
                    & np.isfinite(prediction)
                    & (prediction > 0)
                    & np.isfinite(zed_reference)
                    & (zed_reference > 0)
                )
                zed_mask, cluster = select_foreground_reference_mask(
                    zed_reference, common
                )
                zed_metrics = paired_depth_metrics(
                    prediction, zed_reference, zed_mask
                )
                lidar_dense_reference = np.where(
                    zed_mask, zed_reference, np.nan
                ).astype(np.float32)
                zed_median = zed_metrics["reference_median_m"]
                prefixed(row, "zed", zed_metrics)
                prefixed(row, "zed_cluster", cluster)
                if zed_median is not None:
                    zed_point = point_from_euclidean_depth(
                        centroid, zed_median, target_k, target_distortion
                    )
                    row["zed_x_m"] = float(zed_point[0])
                    row["zed_y_m"] = float(zed_point[1])
                    row["zed_z_m"] = float(zed_point[2])
                    row["zed_3d_error_m"] = float(
                        np.linalg.norm(predicted_point - zed_point)
                    )

            lidar = lidar_reference_for_instance(
                combined_lidar_pixels,
                combined_lidar_ranges,
                sampling_mask,
                dense_reference=lidar_dense_reference,
                consistency_tolerance_metres=args.lidar_consistency,
                minimum_points=args.minimum_lidar_points,
            )
            prefixed(row, "lidar", lidar)
            if lidar["valid"] and predicted_median is not None:
                row["lidar_median_error_m"] = float(
                    predicted_median - lidar["median_m"]
                )
                row["lidar_absolute_error_m"] = abs(row["lidar_median_error_m"])
                lidar_point = point_from_euclidean_depth(
                    centroid,
                    float(lidar["median_m"]),
                    target_k,
                    target_distortion,
                )
                row["lidar_x_m"] = float(lidar_point[0])
                row["lidar_y_m"] = float(lidar_point[1])
                row["lidar_z_m"] = float(lidar_point[2])
                row["lidar_3d_error_m"] = float(
                    np.linalg.norm(predicted_point - lidar_point)
                )
            rows.append(row)

            annotate_detection(
                annotated,
                detection.mask,
                detection.bbox_xyxy,
                detection.track_id,
                detection.confidence,
                prediction_label,
                predicted_median,
                zed_median,
                lidar["median_m"] if lidar["valid"] else None,
            )

        cv2.imwrite(
            str(annotated_dir / f"{args.recording}_G1_A_{image_index:06d}.jpg"),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        print(
            f"[{image_index + 1}/{len(images)}] {image_path.name}: "
            f"{len(detections)} person(s)"
        )

    measurement_path = args.output_dir / "person_measurements.csv"
    write_csv(measurement_path, rows)
    track_rows = summarize_tracks(rows)
    write_csv(args.output_dir / "track_summary.csv", track_rows)

    zed_rows = [row for row in rows if row.get("zed_count", 0)]
    lidar_rows = [row for row in rows if row.get("lidar_valid")]
    lidar_3d_rows = [
        row for row in lidar_rows if row.get("lidar_3d_error_m") is not None
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recording": args.recording,
        "depth_output_dir": str(args.depth_output_dir),
        "prediction_models": sorted(prediction_models),
        "detector_weights": args.detector_weights,
        "ultralytics_version": tracker.ultralytics_version,
        "tracker": args.tracker,
        "project_commits": sorted(project_commits),
        "processed_frames": processed_frames,
        "missing_depth_predictions": missing_predictions,
        "person_detections": len(rows),
        "unique_tracks": len({row["track_id"] for row in rows}),
        "detections_with_zed_reference": len(zed_rows),
        "detections_with_lidar_reference": len(lidar_rows),
        "mean_detector_inference_ms": float(np.mean(detector_speeds))
        if detector_speeds
        else None,
        "mean_depth_inference_ms": float(np.mean(depth_inference_times))
        if depth_inference_times
        else None,
        "median_depth_inference_ms": float(np.median(depth_inference_times))
        if depth_inference_times
        else None,
        "mean_zed_mae_m": float(np.mean([row["zed_mae_m"] for row in zed_rows]))
        if zed_rows
        else None,
        "mean_zed_rmse_m": float(np.mean([row["zed_rmse_m"] for row in zed_rows]))
        if zed_rows
        else None,
        "mean_zed_abs_rel": float(np.mean([row["zed_abs_rel"] for row in zed_rows]))
        if zed_rows
        else None,
        "mean_zed_3d_error_m": float(
            np.mean([row["zed_3d_error_m"] for row in zed_rows])
        )
        if zed_rows
        else None,
        "mean_lidar_3d_error_m": float(
            np.mean([row["lidar_3d_error_m"] for row in lidar_3d_rows])
        )
        if lidar_3d_rows
        else None,
        "configuration": {
            "start_index": args.start_index,
            "max_frames": args.max_frames,
            "frame_step": args.frame_step,
            "detector_confidence": args.detector_confidence,
            "detector_iou": args.detector_iou,
            "detector_image_size": args.detector_image_size,
            "zed_max_dt_seconds": args.zed_max_dt,
            "lidar_max_dt_seconds": args.lidar_max_dt,
            "lidar_consistency_metres": args.lidar_consistency,
            "minimum_lidar_points": args.minimum_lidar_points,
        },
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Measurements: {measurement_path}")
    print(f"Summary:      {args.output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
