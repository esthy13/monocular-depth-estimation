"""LiDAR-to-camera projection and sparse monocular-depth evaluation helpers."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of depth errors at valid, visible LiDAR returns."""

    count: int
    alignment: str
    scale: float
    shift: float
    mae_m: float
    rmse_m: float
    abs_rel: float
    sq_rel: float
    rmse_log: float
    delta_1: float
    delta_2: float
    delta_3: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_lidar_points(path: Path) -> np.ndarray:
    """Load an ``N×3`` (or ``N×4+``) LiDAR array and return XYZ coordinates."""
    points = np.asarray(np.load(path))
    if points.dtype.names:
        names = points.dtype.names
        if not all(name in names for name in ("x", "y", "z")):
            raise ValueError(f"Structured point cloud {path} needs x, y, z fields; has {names}.")
        points = np.column_stack([points["x"], points["y"], points["z"]])
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an N×3 (or wider) point cloud in {path}; got {points.shape}.")
    points = points[:, :3].astype(np.float64, copy=False)
    return points[np.isfinite(points).all(axis=1)]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous transform to an ``N×3`` point array."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4×4 transform; got {transform.shape}.")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3:4]


def project_perspective_points(
    points_camera: np.ndarray,
    camera: dict,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points with OpenCV intrinsics/distortion.

    Returns floating-point pixels, optical-axis depths, and indices into
    ``points_camera``. Points behind the camera or outside the image are removed.
    """
    h, w = image_shape[:2]
    front = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0)
    indices = np.flatnonzero(front)
    if not len(indices):
        return np.empty((0, 2)), np.empty(0), indices
    K = np.asarray(camera["K"], dtype=np.float64)
    dist = np.asarray(camera.get("dist", camera.get("d", [])), dtype=np.float64)
    # OpenCV rejects integer point arrays; projection is intrinsically floating point.
    visible_points = np.asarray(points_camera[indices], dtype=np.float64)
    pixels, _ = cv2.projectPoints(visible_points, np.zeros(3), np.zeros(3), K, dist)
    pixels = pixels.reshape(-1, 2)
    depths = points_camera[indices, 2]
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    return pixels[inside], depths[inside], indices[inside]


def keep_nearest_per_pixel(pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Return indices retaining only the nearest LiDAR return in each integer pixel."""
    if not len(pixels):
        return np.empty(0, dtype=np.int64)
    integer_pixels = np.floor(pixels).astype(np.int64)
    # Sort on pixel first and depth second. The first item per pixel is visible.
    order = np.lexsort((depths, integer_pixels[:, 1], integer_pixels[:, 0]))
    ordered = integer_pixels[order]
    first = np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)]
    return order[first]


def sample_bilinear(depth: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Sample a depth image at sub-pixel locations; invalid neighbours yield NaN."""
    h, w = depth.shape
    x = np.clip(pixels[:, 0], 0, w - 1)
    y = np.clip(pixels[:, 1], 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    values = np.stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]), axis=1)
    weights = np.stack(((1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy), axis=1)
    valid = np.isfinite(values) & (values > 0)
    numerator = np.where(valid, values * weights, 0).sum(axis=1)
    denominator = np.where(valid, weights, 0).sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(x), np.nan), where=denominator > 0)


def align_depth(predicted: np.ndarray, ground_truth: np.ndarray, method: str) -> tuple[np.ndarray, float, float]:
    """Align a positive relative-depth proxy to metric LiDAR depths.

    ``none`` is for already-metric models. ``median`` estimates one scale from
    the medians. ``least_squares`` estimates the single scale that minimises
    ``sum((scale * predicted - ground_truth) ** 2)``. Parameters are fitted on
    these points, so they must be reported with every result.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    valid = np.isfinite(predicted) & (predicted > 0) & np.isfinite(ground_truth) & (ground_truth > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Alignment needs at least two finite, positive prediction/ground-truth pairs.")
    fit_prediction, fit_ground_truth = predicted[valid], ground_truth[valid]
    if method == "none":
        return predicted.copy(), 1.0, 0.0
    if method == "median":
        scale = float(np.median(fit_ground_truth) / np.median(fit_prediction))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Median alignment produced a non-positive or non-finite scale.")
        return predicted * scale, scale, 0.0
    if method == "least_squares":
        denominator = float(np.dot(fit_prediction, fit_prediction))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Least-squares alignment needs finite, non-zero predicted depths.")
        scale = float(np.dot(fit_prediction, fit_ground_truth) / denominator)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Least-squares alignment produced a non-positive or non-finite scale.")
        return predicted * scale, scale, 0.0
    if method == "inverse_least_squares":
        # Depth Anything V2's relative head produces an inverse-depth-like
        # quantity. Fit 1 / Z = a * prediction + b, then invert only the
        # fitted positive inverse depths. Inverting raw values first is
        # unstable: a valid raw value close to zero becomes an enormous outlier.
        inverse_ground_truth = 1.0 / fit_ground_truth
        design = np.column_stack((fit_prediction, np.ones(len(fit_prediction))))
        if np.linalg.matrix_rank(design) < 2:
            raise ValueError("Inverse-depth alignment is degenerate (prediction has no usable variation).")
        scale, shift = np.linalg.lstsq(design, inverse_ground_truth, rcond=None)[0]
        if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
            raise ValueError("Inverse-depth alignment produced invalid parameters.")
        inverse_predicted = predicted * scale + shift
        aligned = np.divide(
            1.0, inverse_predicted, out=np.full(len(predicted), np.nan), where=inverse_predicted > 0
        )
        return aligned, float(scale), float(shift)
    raise ValueError(f"Unknown alignment method {method!r}.")


def evaluate_depth(predicted: np.ndarray, ground_truth: np.ndarray, alignment: str) -> EvaluationResult:
    """Align and calculate standard sparse depth-estimation metrics."""
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    valid_input = (
        np.isfinite(predicted) & (predicted > 0)
        & np.isfinite(ground_truth) & (ground_truth > 0)
    )
    predicted, ground_truth = predicted[valid_input], ground_truth[valid_input]
    if len(predicted) < 2:
        raise ValueError("At least two valid projected LiDAR points are required for evaluation.")
    metric_predicted, scale, shift = align_depth(predicted, ground_truth, alignment)
    valid = np.isfinite(metric_predicted) & (metric_predicted > 0) & np.isfinite(ground_truth) & (ground_truth > 0)
    metric_predicted, ground_truth = metric_predicted[valid], ground_truth[valid]
    if len(metric_predicted) < 2:
        raise ValueError("Alignment produced fewer than two positive depth estimates.")
    error = metric_predicted - ground_truth
    ratio = np.maximum(metric_predicted / ground_truth, ground_truth / metric_predicted)
    return EvaluationResult(
        count=len(metric_predicted), alignment=alignment, scale=scale, shift=shift,
        mae_m=float(np.mean(np.abs(error))), rmse_m=float(np.sqrt(np.mean(error**2))),
        abs_rel=float(np.mean(np.abs(error) / ground_truth)),
        sq_rel=float(np.mean(error**2 / ground_truth)),
        rmse_log=float(np.sqrt(np.mean((np.log(metric_predicted) - np.log(ground_truth)) ** 2))),
        delta_1=float(np.mean(ratio < 1.25)), delta_2=float(np.mean(ratio < 1.25**2)),
        delta_3=float(np.mean(ratio < 1.25**3)),
    )


def save_point_samples(path: Path, pixels: np.ndarray, lidar_depth: np.ndarray, predicted: np.ndarray, aligned: np.ndarray) -> None:
    """Save pointwise values in a portable CSV for plotting and audit."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("u_px", "v_px", "lidar_depth_m", "prediction_proxy", "aligned_prediction_m", "signed_error_m", "absolute_error_m", "relative_error"))
        for pixel, gt, raw, metric in zip(pixels, lidar_depth, predicted, aligned):
            writer.writerow((pixel[0], pixel[1], gt, raw, metric, metric - gt, abs(metric - gt), abs(metric - gt) / gt))


def save_metrics(path: Path, result: EvaluationResult, **metadata: object) -> None:
    """Write metrics plus provenance as JSON."""
    payload = {**metadata, "metrics": result.to_dict()}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def save_projection_overlay(image: np.ndarray, pixels: np.ndarray, depths: np.ndarray, path: Path) -> None:
    """Draw projected LiDAR points coloured by their camera-frame distance."""
    overlay = image.copy()
    if len(pixels):
        lo, hi = np.percentile(depths, (2, 98))
        normalized = np.clip((depths - lo) / max(hi - lo, 1e-6), 0, 1)
        colours = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        for (u, v), colour in zip(np.rint(pixels).astype(int), colours):
            cv2.circle(overlay, (u, v), 2, tuple(map(int, colour[0])), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), overlay)


def save_evaluation_visualization(
    image: np.ndarray,
    raw_prediction: np.ndarray,
    pixels: np.ndarray,
    lidar_depth: np.ndarray,
    aligned_prediction: np.ndarray,
    path: Path,
) -> None:
    """Save RGB, LiDAR projection, prediction, and prediction/LiDAR overlay panels."""
    from src.utils import colorize_depth

    prediction_rgb = colorize_depth(raw_prediction)
    prediction_bgr = cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2BGR)
    lidar_overlay = image.copy()
    prediction_overlay = prediction_bgr.copy()
    if len(pixels):
        lo, hi = np.percentile(lidar_depth, (2, 98))
        normalized = np.clip((lidar_depth - lo) / max(hi - lo, 1e-6), 0, 1)
        colours = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        for (u, v), colour in zip(np.rint(pixels).astype(int), colours):
            point_colour = tuple(map(int, colour[0]))
            cv2.circle(lidar_overlay, (u, v), 2, point_colour, -1, lineType=cv2.LINE_AA)
            cv2.circle(prediction_overlay, (u, v), 2, point_colour, -1, lineType=cv2.LINE_AA)
    top = np.hstack((image, lidar_overlay))
    bottom = np.hstack((prediction_bgr, prediction_overlay))
    cv2.imwrite(str(path), np.vstack((top, bottom)))
