"""Leakage-controlled, report-oriented sparse LiDAR depth evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .lidar_evaluation import (
    keep_nearest_per_pixel,
    load_lidar_points,
    project_perspective_points,
    sample_bilinear,
    transform_points,
)
from .utils import (
    camera_from_lidar_transform,
    find_rgb_images,
    find_sensor_npy_files,
    match_by_timestamp,
    parse_timestamp,
)


METRIC_NAMES = ("mae_m", "rmse_m", "abs_rel", "sq_rel", "rmse_log", "delta_1", "delta_2", "delta_3")


@dataclass(frozen=True)
class AlignmentParameters:
    scale: float
    shift: float
    condition_number: float
    fit_points: int
    source_recording: str
    source_frames: tuple[int, ...]
    method: str = "inverse_depth_affine"


@dataclass(frozen=True)
class ValidityRules:
    min_ground_truth_m: float = 0.1
    max_ground_truth_m: float = 20.0
    min_prediction_m: float = 0.1
    max_prediction_m: float = 100.0
    min_correspondences_per_frame: int = 100
    max_alignment_condition: float = 1e6


def deterministic_calibration_split(frame_indices: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Return deterministic disjoint calibration/evaluation frame lists."""
    if not 0 < fraction < 1:
        raise ValueError("calibration fraction must be strictly between zero and one")
    if len(frame_indices) < 2:
        raise ValueError("at least two frames are needed for a calibration/evaluation split")
    rng = np.random.default_rng(seed)
    count = max(1, min(len(frame_indices) - 1, int(round(len(frame_indices) * fraction))))
    calibration = sorted(rng.choice(frame_indices, size=count, replace=False).tolist())
    calibration_set = set(calibration)
    evaluation = [frame for frame in frame_indices if frame not in calibration_set]
    return calibration, evaluation


def fit_fixed_inverse_alignment(
    raw_prediction: np.ndarray,
    ground_truth_m: np.ndarray,
    source_recording: str,
    source_frames: list[int],
    max_condition: float = 1e6,
) -> AlignmentParameters:
    """Fit ``1/Z = scale * raw + shift`` and report numerical conditioning."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    gt = np.asarray(ground_truth_m, dtype=np.float64)
    valid = np.isfinite(raw) & (raw > 0) & np.isfinite(gt) & (gt > 0)
    raw, gt = raw[valid], gt[valid]
    if len(raw) < 2:
        raise ValueError("alignment fit has fewer than two valid correspondences")
    design = np.column_stack((raw, np.ones(len(raw))))
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > max_condition or np.linalg.matrix_rank(design) < 2:
        raise ValueError(f"ill-conditioned alignment fit: condition_number={condition:.6g}")
    scale, shift = np.linalg.lstsq(design, 1.0 / gt, rcond=None)[0]
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        raise ValueError(f"invalid alignment parameters: scale={scale}, shift={shift}")
    return AlignmentParameters(
        float(scale), float(shift), condition, len(raw), source_recording, tuple(source_frames)
    )


def apply_inverse_alignment(raw_prediction: np.ndarray, parameters: AlignmentParameters) -> np.ndarray:
    inverse_depth = parameters.scale * np.asarray(raw_prediction, dtype=np.float64) + parameters.shift
    return np.divide(1.0, inverse_depth, out=np.full(inverse_depth.shape, np.nan), where=inverse_depth > 0)


def calculate_metrics(prediction_m: np.ndarray, ground_truth_m: np.ndarray) -> dict[str, float | int]:
    """Calculate standard depth metrics without changing or clipping predictions."""
    pred = np.asarray(prediction_m, dtype=np.float64)
    gt = np.asarray(ground_truth_m, dtype=np.float64)
    valid = np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > 0)
    pred, gt = pred[valid], gt[valid]
    if not len(pred):
        return {"count": 0, **{name: float("nan") for name in METRIC_NAMES}}
    error = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "count": int(len(pred)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(np.abs(error) / gt)),
        "sq_rel": float(np.mean(error**2 / gt)),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "delta_2": float(np.mean(ratio < 1.25**2)),
        "delta_3": float(np.mean(ratio < 1.25**3)),
    }


def summarize_frame_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    valid_rows = [row for row in rows if row.get("status") == "valid"]
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in valid_rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values)), "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)), "min": float(np.min(values)),
            "max": float(np.max(values)),
        } if len(values) else {}
    return summary


def depth_binned_metrics(prediction_m: np.ndarray, ground_truth_m: np.ndarray, edges: list[float]) -> list[dict]:
    rows = []
    pred, gt = np.asarray(prediction_m), np.asarray(ground_truth_m)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (gt >= low) & (gt < high)
        rows.append({"range_m": [low, high], **calculate_metrics(pred[selected], gt[selected])})
    return rows


def rejection_counts(raw: np.ndarray, gt: np.ndarray, rules: ValidityRules) -> tuple[np.ndarray, dict[str, int]]:
    """Apply ordered, mutually exclusive input rejection rules."""
    remaining = np.ones(len(raw), dtype=bool)
    counts: dict[str, int] = {}
    reasons = {
        "lidar_nonfinite": ~np.isfinite(gt),
        "lidar_nonpositive": np.isfinite(gt) & (gt <= 0),
        "lidar_below_min": np.isfinite(gt) & (gt > 0) & (gt < rules.min_ground_truth_m),
        "lidar_above_max": np.isfinite(gt) & (gt > rules.max_ground_truth_m),
        "prediction_nonfinite": ~np.isfinite(raw),
        "prediction_nonpositive": np.isfinite(raw) & (raw <= 0),
    }
    for name, mask in reasons.items():
        rejected = remaining & mask
        counts[name] = int(np.count_nonzero(rejected))
        remaining &= ~mask
    counts["accepted_input"] = int(np.count_nonzero(remaining))
    return remaining, counts


def _prediction_path(output_dir: Path, recording: str, camera: str, frame: int) -> Path:
    return output_dir / f"{recording}_{camera}_{frame:04d}_depth_raw.npy"


def collect_recording_samples(
    data_dir: Path,
    output_dir: Path,
    recording: str,
    camera: str,
    lidar: str,
    intrinsics: dict,
    transform: np.ndarray,
    max_dt: float,
    time_offset: float,
    rules: ValidityRules,
) -> tuple[list[dict], dict]:
    images = find_rgb_images(data_dir, camera, recording)
    lidar_files = find_sensor_npy_files(data_dir, lidar, recording)
    depth_files = find_sensor_npy_files(data_dir, f"{camera}_depth", recording)
    lidar_pairs = dict(match_by_timestamp(images, lidar_files, max_dt, time_offset))
    stereo_pairs = dict(match_by_timestamp(images, depth_files, 0.001))
    frames: list[dict] = []
    unmatched = []
    missing_predictions = []
    previous_gray = None
    for frame_index, image_path in enumerate(images):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        motion_score = float("nan") if previous_gray is None else float(np.mean(cv2.absdiff(gray, previous_gray)))
        previous_gray = gray
        if image_path not in lidar_pairs:
            unmatched.append({"frame": frame_index, "image": image_path.name, "reason": "no_lidar_within_threshold"})
            continue
        prediction_path = _prediction_path(output_dir, recording, camera, frame_index)
        if not prediction_path.exists():
            missing_predictions.append({"frame": frame_index, "path": str(prediction_path)})
            continue
        raw_image = np.load(prediction_path).squeeze()
        lidar_path = lidar_pairs[image_path]
        stored_count = int(len(np.load(lidar_path, mmap_mode="r")))
        lidar_points = load_lidar_points(lidar_path)
        camera_points = transform_points(lidar_points, transform)
        front = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0)
        pixels, gt, _ = project_perspective_points(camera_points, intrinsics[camera], raw_image.shape)
        projected_count = len(pixels)
        nearest = keep_nearest_per_pixel(pixels, gt)
        pixels, gt = pixels[nearest], gt[nearest]
        raw = sample_bilinear(raw_image, pixels)
        accepted, rejected = rejection_counts(raw, gt, rules)
        stereo_boundary = np.zeros(len(pixels), dtype=bool)
        stereo_projection_median_abs_error_m = float("nan")
        if image_path in stereo_pairs:
            stereo = np.load(stereo_pairs[image_path]).squeeze().astype(np.float32)
            gradient_x = cv2.Sobel(stereo, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(stereo, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.hypot(gradient_x, gradient_y)
            sampled_gradient = sample_bilinear(np.where(np.isfinite(gradient), gradient + 1e-12, np.nan), pixels)
            stereo_boundary = sampled_gradient > 0.5
            sampled_stereo = sample_bilinear(stereo, pixels)
            stereo_valid = np.isfinite(sampled_stereo) & (sampled_stereo > rules.min_ground_truth_m) & (sampled_stereo < rules.max_ground_truth_m)
            if np.any(stereo_valid):
                stereo_projection_median_abs_error_m = float(np.median(np.abs(gt[stereo_valid] - sampled_stereo[stereo_valid])))
        frame = {
            "recording": recording, "frame": frame_index, "image_path": str(image_path),
            "lidar_path": str(lidar_path), "prediction_path": str(prediction_path),
            "camera_timestamp": parse_timestamp(image_path), "lidar_timestamp": parse_timestamp(lidar_path),
            "timestamp_delta_s": parse_timestamp(lidar_path) + time_offset - parse_timestamp(image_path),
            "motion_score_mean_abs_gray": motion_score,
            "stereo_projection_median_abs_error_m": stereo_projection_median_abs_error_m,
            "stored_lidar_points": stored_count, "finite_lidar_points": len(lidar_points),
            "front_camera_points": int(np.count_nonzero(front)), "projected_points": projected_count,
            "unique_projected_points": len(pixels), "rejections": rejected,
            "camera_xyz_range": {
                "x": [float(np.nanmin(camera_points[:, 0])), float(np.nanmax(camera_points[:, 0]))],
                "y": [float(np.nanmin(camera_points[:, 1])), float(np.nanmax(camera_points[:, 1]))],
                "z": [float(np.nanmin(camera_points[:, 2])), float(np.nanmax(camera_points[:, 2]))],
            },
            "pixels": pixels[accepted], "ground_truth_m": gt[accepted], "raw_prediction": raw[accepted],
            "boundary": stereo_boundary[accepted], "raw_image": raw_image,
        }
        frames.append(frame)
    sync = {
        "total_camera_frames": len(images), "matched_frames": len(lidar_pairs),
        "unmatched_frames": unmatched, "missing_prediction_frames": missing_predictions,
    }
    return frames, sync


def evaluate_mode(
    frames: list[dict],
    mode: str,
    rules: ValidityRules,
    fixed_parameters: AlignmentParameters | None = None,
) -> dict:
    frame_rows, all_pred, all_gt, all_boundary = [], [], [], []
    point_rejections: dict[str, int] = {}
    exclusions = []
    for frame in frames:
        raw, gt = frame["raw_prediction"], frame["ground_truth_m"]
        condition = float("nan")
        scale = shift = float("nan")
        try:
            if mode == "unaligned_relative_proxy":
                pred = np.divide(1.0, raw, out=np.full(raw.shape, np.nan), where=raw > 0)
            elif mode == "fixed_held_out":
                if fixed_parameters is None:
                    raise ValueError("fixed parameters were not supplied")
                pred = apply_inverse_alignment(raw, fixed_parameters)
                scale, shift, condition = fixed_parameters.scale, fixed_parameters.shift, fixed_parameters.condition_number
            elif mode == "oracle_diagnostic":
                parameters = fit_fixed_inverse_alignment(raw, gt, frame["recording"], [frame["frame"]], rules.max_alignment_condition)
                pred = apply_inverse_alignment(raw, parameters)
                scale, shift, condition = parameters.scale, parameters.shift, parameters.condition_number
            else:
                raise ValueError(f"unknown mode {mode}")
        except ValueError as error:
            exclusions.append({"recording": frame["recording"], "frame": frame["frame"], "reason": str(error)})
            frame_rows.append({"recording": frame["recording"], "frame": frame["frame"], "status": "excluded", "reason": str(error)})
            continue
        valid = np.isfinite(pred) & (pred >= rules.min_prediction_m) & (pred <= rules.max_prediction_m)
        local_rejections = {
            "aligned_prediction_nonfinite_or_nonpositive": int(np.count_nonzero(~np.isfinite(pred) | (pred <= 0))),
            "aligned_prediction_below_min": int(np.count_nonzero(np.isfinite(pred) & (pred > 0) & (pred < rules.min_prediction_m))),
            "aligned_prediction_above_max": int(np.count_nonzero(np.isfinite(pred) & (pred > rules.max_prediction_m))),
        }
        for name, count in local_rejections.items():
            point_rejections[name] = point_rejections.get(name, 0) + count
        if np.count_nonzero(valid) < rules.min_correspondences_per_frame:
            reason = f"insufficient_correspondences:{np.count_nonzero(valid)}<{rules.min_correspondences_per_frame}"
            exclusions.append({"recording": frame["recording"], "frame": frame["frame"], "reason": reason})
            frame_rows.append({"recording": frame["recording"], "frame": frame["frame"], "status": "excluded", "reason": reason})
            continue
        pred_valid, gt_valid = pred[valid], gt[valid]
        metrics = calculate_metrics(pred_valid, gt_valid)
        residual = pred_valid - gt_valid
        frame_rows.append({
            "recording": frame["recording"], "frame": frame["frame"], "status": "valid",
            "timestamp_delta_s": frame["timestamp_delta_s"], "alignment_scale": scale,
            "alignment_shift": shift, "alignment_condition_number": condition,
            "prediction_min_m": float(np.min(pred_valid)), "prediction_max_m": float(np.max(pred_valid)),
            "residual_mean_m": float(np.mean(residual)), "residual_std_m": float(np.std(residual)),
            **metrics,
        })
        frame[f"{mode}_prediction_m"] = pred
        frame[f"{mode}_valid"] = valid
        all_pred.append(pred_valid); all_gt.append(gt_valid); all_boundary.append(frame["boundary"][valid])
    pred = np.concatenate(all_pred) if all_pred else np.empty(0)
    gt = np.concatenate(all_gt) if all_gt else np.empty(0)
    boundary = np.concatenate(all_boundary) if all_boundary else np.empty(0, dtype=bool)
    return {
        "mode": mode,
        "interpretation": (
            "reciprocal relative proxy in arbitrary units; errors are not metric-depth accuracy"
            if mode == "unaligned_relative_proxy" else
            "single alignment fitted only on the recorded calibration subset and frozen for these frames"
            if mode == "fixed_held_out" else
            "per-frame oracle fitted and scored on the same LiDAR samples; optimistic diagnostic only"
        ),
        "valid_frame_count": sum(row.get("status") == "valid" for row in frame_rows),
        "excluded_frames": exclusions, "point_rejections": point_rejections,
        "pooled_metrics": calculate_metrics(pred, gt),
        "frame_statistics": summarize_frame_metrics(frame_rows),
        "depth_bins": depth_binned_metrics(pred, gt, [0.1, 1, 2, 4, 6, 10, 20]),
        "boundary_metrics": calculate_metrics(pred[boundary], gt[boundary]),
        "non_boundary_metrics": calculate_metrics(pred[~boundary], gt[~boundary]),
        "frame_rows": frame_rows, "all_prediction": pred, "all_ground_truth": gt,
    }


def stable_configuration_hash(configuration: dict) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items() if not key.startswith("all_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
