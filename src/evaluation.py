"""Calibrated geometry and metrics for person-depth evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .utils import parse_timestamp


@dataclass(frozen=True)
class TimestampMatch:
    """Nearest sensor file and its signed offset from the requested timestamp."""

    path: Path
    delta_seconds: float


def nearest_timestamp_file(
    files: Iterable[Path],
    timestamp: float,
    max_delta_seconds: float,
) -> TimestampMatch | None:
    """Return the closest timestamped file within ``max_delta_seconds``."""
    best: TimestampMatch | None = None
    for path in files:
        delta = parse_timestamp(path) - timestamp
        if best is None or abs(delta) < abs(best.delta_seconds):
            best = TimestampMatch(Path(path), delta)
    if best is None or abs(best.delta_seconds) > max_delta_seconds:
        return None
    return best


def fisheye_unit_ray(
    pixel: tuple[float, float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Convert a fisheye image pixel to an OpenCV camera-frame unit ray."""
    points = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.fisheye.undistortPoints(
        points,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )[0, 0]
    ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def point_from_euclidean_depth(
    pixel: tuple[float, float],
    depth_metres: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Back-project Euclidean ray distance to a 3D camera-frame point."""
    return fisheye_unit_ray(pixel, camera_matrix, distortion) * float(depth_metres)


def project_camera_points_to_fisheye(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame XYZ points and return in-frame pixels and ranges."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_front = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)
    points = points[finite_front]
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

    ranges = np.linalg.norm(points, axis=1)
    pixels, _ = cv2.fisheye.projectPoints(
        points[:, None, :],
        np.zeros(3),
        np.zeros(3),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    pixels = pixels[:, 0]
    height, width = output_shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[inside].astype(np.float32), ranges[inside].astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous sensor-to-reference transform."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def zbuffer_range_map(
    pixels: np.ndarray,
    ranges: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize projected ranges, keeping the nearest sample per pixel."""
    height, width = output_shape
    output = np.full((height, width), np.nan, dtype=np.float32)
    if len(ranges) == 0:
        return output

    rounded = np.rint(pixels).astype(np.int64)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
        & np.isfinite(ranges)
        & (ranges > 0)
    )
    rounded = rounded[inside]
    ranges = np.asarray(ranges, dtype=np.float32)[inside]
    if len(ranges) == 0:
        return output

    linear = rounded[:, 1] * width + rounded[:, 0]
    order = np.lexsort((ranges, linear))
    sorted_linear = linear[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    selected = order[first]
    output[rounded[selected, 1], rounded[selected, 0]] = ranges[selected]
    return output


def project_perspective_depth_to_fisheye(
    depth: np.ndarray,
    source_camera_matrix: np.ndarray,
    sensor_to_reference: np.ndarray,
    target_camera_matrix: np.ndarray,
    target_distortion: np.ndarray,
    output_shape: tuple[int, int],
    max_depth_metres: float = 30.0,
) -> np.ndarray:
    """Reproject a perspective Z-depth image into a fisheye Euclidean-range map."""
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image; got shape {depth.shape}")

    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth_metres)
    rows, cols = np.nonzero(valid)
    z = depth[valid].astype(np.float64)
    source_k = np.asarray(source_camera_matrix, dtype=np.float64)
    x = (cols - source_k[0, 2]) * z / source_k[0, 0]
    y = (rows - source_k[1, 2]) * z / source_k[1, 1]
    source_points = np.column_stack((x, y, z))
    reference_points = transform_points(source_points, sensor_to_reference)
    pixels, ranges = project_camera_points_to_fisheye(
        reference_points,
        target_camera_matrix,
        target_distortion,
        output_shape,
    )
    return zbuffer_range_map(pixels, ranges, output_shape)


def project_lidar_to_fisheye(
    point_cloud: np.ndarray,
    sensor_to_reference: np.ndarray,
    target_camera_matrix: np.ndarray,
    target_distortion: np.ndarray,
    output_shape: tuple[int, int],
    max_range_metres: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an XYZ[I] point cloud and project it into the fisheye image."""
    point_cloud = np.asarray(point_cloud)
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
        raise ValueError(f"Expected an Nx3/Nx4 point cloud; got {point_cloud.shape}")
    points = point_cloud[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    reference_points = transform_points(points, sensor_to_reference)
    pixels, ranges = project_camera_points_to_fisheye(
        reference_points,
        target_camera_matrix,
        target_distortion,
        output_shape,
    )
    keep = ranges <= max_range_metres
    return pixels[keep], ranges[keep]


def erode_instance_mask(
    mask: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    fraction: float = 0.02,
) -> np.ndarray:
    """Erode an object mask relative to its box size to suppress boundary mixing."""
    x1, y1, x2, y2 = bbox_xyxy
    radius = max(1, int(round(min(x2 - x1, y2 - y1) * fraction)))
    radius = min(radius, 8)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return an instance-mask centroid as ``(u, v)``."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("Cannot compute the centroid of an empty mask")
    return float(np.mean(cols)), float(np.mean(rows))


def depth_statistics(depth: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    """Return robust summary statistics for positive finite depth samples."""
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    values = np.asarray(depth, dtype=np.float32)[valid]
    if values.size == 0:
        return {
            "count": 0,
            "median_m": None,
            "mean_m": None,
            "p10_m": None,
            "p90_m": None,
        }
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    return {
        "count": int(values.size),
        "median_m": float(median),
        "mean_m": float(np.mean(values)),
        "p10_m": float(p10),
        "p90_m": float(p90),
    }


def select_foreground_reference_mask(
    reference_depth: np.ndarray,
    base_mask: np.ndarray,
    minimum_samples: int = 100,
    minimum_cluster_fraction: float = 0.15,
    minimum_separation_metres: float = 0.35,
) -> tuple[np.ndarray, dict[str, float | bool | int | None]]:
    """Remove a farther background mode caused by cross-camera parallax.

    A deterministic two-means split is accepted only when both clusters are
    substantial and clearly separated. Otherwise the complete common mask is
    retained, avoiding an artificial split across the front/back of one person.
    """
    base_valid = base_mask.astype(bool) & np.isfinite(reference_depth) & (
        reference_depth > 0
    )
    values = reference_depth[base_valid].astype(np.float64)
    metadata: dict[str, float | bool | int | None] = {
        "foreground_cluster_used": False,
        "near_center_m": None,
        "far_center_m": None,
        "input_count": int(values.size),
        "selected_count": int(values.size),
    }
    if values.size < minimum_samples:
        return base_valid, metadata

    centers = np.percentile(values, [25, 75]).astype(np.float64)
    labels = np.zeros(values.size, dtype=bool)
    for _ in range(30):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1).astype(bool)
        if new_labels.all() or (~new_labels).all():
            break
        new_centers = np.array(
            [values[~new_labels].mean(), values[new_labels].mean()]
        )
        if np.allclose(new_centers, centers, atol=1e-6):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    order = np.argsort(centers)
    near_index, far_index = int(order[0]), int(order[1])
    near_labels = labels == bool(near_index)
    near_fraction = float(np.mean(near_labels))
    far_fraction = 1.0 - near_fraction
    separation = float(centers[far_index] - centers[near_index])
    required_separation = max(
        minimum_separation_metres, 0.2 * float(centers[near_index])
    )

    metadata.update(
        {
            "near_center_m": float(centers[near_index]),
            "far_center_m": float(centers[far_index]),
        }
    )
    if (
        near_fraction < minimum_cluster_fraction
        or far_fraction < minimum_cluster_fraction
        or separation < required_separation
    ):
        return base_valid, metadata

    selected = np.zeros(reference_depth.shape, dtype=bool)
    selected_values = np.zeros(values.size, dtype=bool)
    selected_values[:] = near_labels
    selected[base_valid] = selected_values
    metadata["foreground_cluster_used"] = True
    metadata["selected_count"] = int(np.sum(selected))
    return selected, metadata


def paired_depth_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute common-pixel metric-depth errors."""
    common = (
        mask.astype(bool)
        & np.isfinite(prediction)
        & (prediction > 0)
        & np.isfinite(reference)
        & (reference > 0)
    )
    predicted = prediction[common].astype(np.float64)
    measured = reference[common].astype(np.float64)
    if measured.size == 0:
        return {
            "count": 0,
            "prediction_median_m": None,
            "reference_median_m": None,
            "median_error_m": None,
            "bias_m": None,
            "mae_m": None,
            "rmse_m": None,
            "abs_rel": None,
        }

    error = predicted - measured
    return {
        "count": int(measured.size),
        "prediction_median_m": float(np.median(predicted)),
        "reference_median_m": float(np.median(measured)),
        "median_error_m": float(np.median(error)),
        "bias_m": float(np.mean(error)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(np.abs(error) / measured)),
    }


def sample_depth_near_pixels(
    depth: np.ndarray,
    pixels: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Sample the local median depth near each sparse projected point."""
    height, width = depth.shape
    rounded = np.rint(pixels).astype(int)
    samples = np.full(len(rounded), np.nan, dtype=np.float32)
    for index, (u, v) in enumerate(rounded):
        x1, x2 = max(0, u - radius), min(width, u + radius + 1)
        y1, y2 = max(0, v - radius), min(height, v + radius + 1)
        window = depth[y1:y2, x1:x2]
        values = window[np.isfinite(window) & (window > 0)]
        if values.size:
            samples[index] = float(np.median(values))
    return samples


def lidar_reference_for_instance(
    projected_pixels: np.ndarray,
    projected_ranges: np.ndarray,
    instance_mask: np.ndarray,
    dense_reference: np.ndarray | None = None,
    consistency_tolerance_metres: float = 0.5,
    minimum_points: int = 10,
) -> dict[str, float | int | bool | None]:
    """Summarize LiDAR returns that are visible on one person instance.

    When stereo is available, a LiDAR return must agree with its local stereo
    range. This rejects background points that project through a person because
    the LiDAR and camera have different viewpoints.
    """
    if len(projected_ranges) == 0:
        return {
            "valid": False,
            "stereo_consistency_used": dense_reference is not None,
            "points_in_mask": 0,
            "consistent_points": 0,
            "pixel_median_u": None,
            "pixel_median_v": None,
            "median_m": None,
            "p10_m": None,
            "p90_m": None,
        }

    rounded = np.rint(projected_pixels).astype(int)
    height, width = instance_mask.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    rounded = rounded[inside]
    pixels = projected_pixels[inside]
    ranges = projected_ranges[inside]
    in_mask = instance_mask[rounded[:, 1], rounded[:, 0]].astype(bool)
    pixels = pixels[in_mask]
    ranges = ranges[in_mask]
    points_in_mask = int(len(ranges))

    if dense_reference is not None and len(ranges):
        nearby_reference = sample_depth_near_pixels(dense_reference, pixels)
        consistent = np.isfinite(nearby_reference) & (
            np.abs(ranges - nearby_reference) <= consistency_tolerance_metres
        )
        pixels = pixels[consistent]
        ranges = ranges[consistent]

    if len(ranges) < minimum_points:
        return {
            "valid": False,
            "stereo_consistency_used": dense_reference is not None,
            "points_in_mask": points_in_mask,
            "consistent_points": int(len(ranges)),
            "pixel_median_u": None,
            "pixel_median_v": None,
            "median_m": None,
            "p10_m": None,
            "p90_m": None,
        }

    p10, median, p90 = np.percentile(ranges, [10, 50, 90])
    return {
        "valid": True,
        "stereo_consistency_used": dense_reference is not None,
        "points_in_mask": points_in_mask,
        "consistent_points": int(len(ranges)),
        "pixel_median_u": float(np.median(pixels[:, 0])),
        "pixel_median_v": float(np.median(pixels[:, 1])),
        "median_m": float(median),
        "p10_m": float(p10),
        "p90_m": float(p90),
    }
