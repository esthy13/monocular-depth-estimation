"""Utility functions for loading calibration data, finding images, and saving results."""
from __future__ import annotations

import bisect
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


# ---------------------------------------------------------------------------
# Calibration loaders
# ---------------------------------------------------------------------------

def load_intrinsics(path: Path) -> dict:
    """Load camera intrinsics from intrinsic.json (OpenCV convention, K + dist)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    with path.open() as f:
        return json.load(f)


def load_extrinsics(path: Path) -> dict:
    """Load sensor extrinsics from extrinsics.json (4x4 homogeneous transforms, ref=G1_A)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Extrinsics file not found: {path}")
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_rgb_images(
    data_dir: Path,
    sensor_name: str = "ZED_B",
    recording: str = "recording1",
) -> list[Path]:
    """Return sorted list of .jpg image paths for a sensor under a recording.

    Directory layout: <data_dir>/<recording>/data/<sensor>/<sensor>/*.jpg
    RGB sensors available: 'ZED_B' (perspective), 'G1_A' (fisheye).
    """
    data_dir = Path(data_dir)
    sensor_dir = data_dir / recording / "data" / sensor_name / sensor_name

    if not sensor_dir.exists():
        recordings = sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("recording")
        )
        raise FileNotFoundError(
            f"Sensor directory not found: {sensor_dir}\n"
            f"Available recordings: {recordings}"
        )

    images = sorted(sensor_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in {sensor_dir}")
    return images


def find_depth_files(
    data_dir: Path,
    sensor_name: str = "ZED_B_depth",
    recording: str = "recording1",
) -> list[Path]:
    """Return sorted list of .npy depth files for a sensor under a recording."""
    data_dir = Path(data_dir)
    sensor_dir = data_dir / recording / "data" / sensor_name / sensor_name

    if not sensor_dir.exists():
        raise FileNotFoundError(f"Depth sensor directory not found: {sensor_dir}")

    files = sorted(sensor_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy depth files found in {sensor_dir}")
    return files


def find_sensor_npy_files(
    data_dir: Path,
    sensor_name: str,
    recording: str = "recording1",
) -> list[Path]:
    """Return sorted NumPy files for any sensor stored as ``data/<sensor>/<sensor>``."""
    sensor_dir = Path(data_dir) / recording / "data" / sensor_name / sensor_name
    if not sensor_dir.exists():
        available = sorted(p.name for p in (Path(data_dir) / recording / "data").iterdir())
        raise FileNotFoundError(f"Sensor directory not found: {sensor_dir}. Available sensors: {available}")
    files = sorted(sensor_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {sensor_dir}")
    return files


def sensor_to_reference_transform(extrinsics: dict, sensor_name: str) -> np.ndarray:
    """Extract a sensor-to-reference 4×4 calibration matrix from common JSON layouts."""
    if sensor_name not in extrinsics:
        raise KeyError(f"No extrinsic calibration for {sensor_name!r}; keys: {list(extrinsics)}")
    entry = extrinsics[sensor_name]
    if isinstance(entry, dict):
        for key in ("T", "transform", "matrix", "extrinsic", "extrinsics"):
            if key in entry:
                entry = entry[key]
                break
    matrix = np.asarray(entry, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Extrinsic for {sensor_name!r} must be 4×4; got {matrix.shape}.")
    return matrix


def camera_from_lidar_transform(
    extrinsics: dict,
    camera_sensor: str,
    lidar_sensor: str,
    convention: str = "sensor_to_reference",
) -> np.ndarray:
    """Return ``T_camera_lidar`` from calibrations expressed relative to a reference.

    ``sensor_to_reference`` means each stored matrix maps that sensor's frame to
    the reference frame; choose ``reference_to_sensor`` if the JSON uses inverse
    matrices. Keeping this choice explicit avoids silently evaluating with a
    reversed calibration.
    """
    camera = sensor_to_reference_transform(extrinsics, camera_sensor)
    lidar = sensor_to_reference_transform(extrinsics, lidar_sensor)
    if convention == "sensor_to_reference":
        return np.linalg.inv(camera) @ lidar
    if convention == "reference_to_sensor":
        return camera @ np.linalg.inv(lidar)
    raise ValueError("convention must be 'sensor_to_reference' or 'reference_to_sensor'.")


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def parse_timestamp(filepath: Path) -> float:
    """Extract float timestamp from filenames like 0000000012_1779291269.232229471.jpg."""
    # Filename stem: 0000000012_1779291269.232229471
    match = re.search(r"_(\d+\.\d+)$", Path(filepath).stem)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from filename: {Path(filepath).name}")
    return float(match.group(1))


def match_by_timestamp(
    source_files: list[Path],
    target_files: list[Path],
    max_dt: float = 0.05,
    time_offset: float = 0.0,
) -> list[tuple[Path, Path]]:
    """Pair each source file with the nearest-timestamp target file.

    ``time_offset`` is added to target (LiDAR) timestamps before matching.
    Returns only pairs where ``|ts_source - (ts_target + time_offset)| <= max_dt``.
    """
    target_ts_pairs = sorted(
        (parse_timestamp(f) + time_offset, f) for f in target_files
    )
    target_ts = [t for t, _ in target_ts_pairs]
    target_fs = [f for _, f in target_ts_pairs]

    pairs: list[tuple[Path, Path]] = []
    for src in source_files:
        src_ts = parse_timestamp(src)
        idx = bisect.bisect_left(target_ts, src_ts)

        candidates: list[tuple[float, Path]] = []
        if idx < len(target_ts):
            candidates.append((abs(target_ts[idx] - src_ts), target_fs[idx]))
        if idx > 0:
            candidates.append((abs(target_ts[idx - 1] - src_ts), target_fs[idx - 1]))

        if candidates:
            best_dt, best_match = min(candidates, key=lambda x: x[0])
            if best_dt <= max_dt:
                pairs.append((src, best_match))

    return pairs


# ---------------------------------------------------------------------------
# DAC camera parameter conversion
# ---------------------------------------------------------------------------

def intrinsics_to_dac_cam_params(sensor_name: str, intrinsics: dict) -> dict:
    """Convert our intrinsics.json format to the cam_params dict expected by DAC.

    DAC's cam_to_erp_patch_fast distinguishes two camera models:
      - OPENCV_FISHEYE  (k1-k4 radial coefficients, fl_x/fl_y focal lengths)
      - PINHOLE         (standard perspective, fx/fy)

    Args:
        sensor_name: key in the intrinsics dict, e.g. 'G1_A' or 'ZED_B'.
        intrinsics: loaded from intrinsic.json.
    """
    cam = intrinsics[sensor_name]
    K = cam["K"]
    dist = cam.get("dist", [0.0, 0.0, 0.0, 0.0])
    model = cam.get("model", "perspective")

    if model == "fisheye":
        return {
            "camera_model": "OPENCV_FISHEYE",
            "fl_x": float(K[0][0]),
            "fl_y": float(K[1][1]),
            "cx":   float(K[0][2]),
            "cy":   float(K[1][2]),
            "k1": float(dist[0]),
            "k2": float(dist[1]),
            "k3": float(dist[2]),
            "k4": float(dist[3]),
        }
    else:
        return {
            "camera_model": "PINHOLE",
            "fx": float(K[0][0]),
            "fy": float(K[1][1]),
            "cx": float(K[0][2]),
            "cy": float(K[1][2]),
        }


# ---------------------------------------------------------------------------
# Fisheye mask
# ---------------------------------------------------------------------------

def create_fisheye_valid_mask(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    black_threshold: int = 20,
    erode_frac: float = 0.01,
) -> np.ndarray:
    """Detect the valid circular lens region of a fisheye image as a clean circle.

    Outside the fisheye lens circle the border is near-black — it is invalid
    lens area, NOT real scene content. This mask isolates the valid circle so
    the black border cannot distort depth normalization or colormap range.

    This is a lens-boundary mask, NOT object segmentation.

    The lens circle is fixed camera geometry. Its centre is the principal point
    (pass via `center` from the intrinsics); its radius is found from where the
    dark outer region begins, measured outward from that centre. Interior dark
    objects are excluded by keeping only the border-connected dark region, so the
    estimate is independent of scene content and the same on every frame.

    Args:
        image: BGR uint8 image from a fisheye camera.
        center: (cx, cy) lens centre in pixels — the camera principal point.
                Defaults to the image centre if not provided.
        black_threshold: pixels with grayscale value ≤ this are treated as
                         invalid border. Default 20 is robust to JPEG/vignette.
        erode_frac: fraction of the radius to trim, removing the vignette ring.

    Returns:
        uint8 (H, W) mask — 1 = valid scene pixel, 0 = invalid fisheye border.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = center if center is not None else (w / 2.0, h / 2.0)

    dark = (gray <= black_threshold).astype(np.uint8)

    # Keep only dark pixels connected to the image corners — the true outside-lens
    # region. Dark objects inside the lens are surrounded by bright scene and stay
    # excluded, so they cannot pull the radius estimate inward.
    ff = dark.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[sy, sx]:
            cv2.floodFill(ff, ff_mask, (sx, sy), 2)
    outside = ff == 2

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if outside.sum() < 0.001 * h * w:
        # No dark border found (not a circular fisheye) — whole frame is valid
        return np.ones((h, w), dtype=np.uint8)

    # Radius = innermost reach of the outer dark region (1st percentile is robust
    # to stray pixels), then trimmed slightly to drop the dark vignette edge.
    radius = float(np.percentile(dist[outside], 1.0)) * (1.0 - erode_frac)

    mask = (dist <= radius).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Depth colorization
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    colormap: str = "inferno",
    invert: bool = False,
) -> np.ndarray:
    """Normalize and colorize a depth map, computing min/max from valid pixels only.

    Args:
        depth: float32 (H, W) depth array. May contain NaN for invalid pixels.
        valid_mask: optional uint8/bool (H, W) mask — 1=valid, 0=invalid.
                    Invalid pixels are rendered black (0, 0, 0).
        colormap: matplotlib colormap name.
        invert: reverse the colormap so near pixels are bright (use for metric
                depth, where near = small value).

    Returns:
        uint8 (H, W, 3) RGB colorized depth image.
    """
    effective_mask = np.isfinite(depth)
    if valid_mask is not None:
        effective_mask &= valid_mask.astype(bool)

    d_valid = depth[effective_mask]
    if d_valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d_min, d_max = float(d_valid.min()), float(d_valid.max())
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    if d_max > d_min:
        depth_norm[effective_mask] = (depth[effective_mask] - d_min) / (d_max - d_min)

    cmap = plt.get_cmap(colormap)
    if invert:
        cmap = cmap.reversed()
    colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)  # RGB
    colored[~effective_mask] = 0  # black for invalid/masked pixels
    return colored


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_depth_visualization(
    depth: np.ndarray,
    output_path: Path,
    rgb_image: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    colormap: str = "inferno",
    invert: bool = False,
    robust_percentiles: tuple[float, float] = (2.0, 98.0),
) -> None:
    """Save a colorized depth map as a PNG, optionally side-by-side with the RGB image.

    The colormap range is computed from valid pixels only, so an invalid fisheye
    border cannot skew the scale. Invalid pixels are rendered black.

    Args:
        depth: float32 (H, W) depth array. May contain NaN.
        output_path: destination .png path.
        rgb_image: optional BGR uint8 image to place left of the depth panel.
        valid_mask: optional uint8/bool (H, W) mask — 0=invalid (e.g. fisheye border).
        colormap: matplotlib colormap name (default 'inferno').
        invert: reverse the colormap so near pixels are bright (use for metric
                depth, where near = small value).
        robust_percentiles: (low, high) percentiles of valid depth used as the
                colormap range. Clipping outliers (a few very-near/very-far
                pixels) spreads the colours over the bulk of the scene, giving
                much better contrast than raw min/max.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a masked array so matplotlib computes colorbar range from valid pixels only
    combined_invalid = ~np.isfinite(depth)
    if valid_mask is not None:
        combined_invalid |= ~valid_mask.astype(bool)
    depth_display = ma.array(depth, mask=combined_invalid)

    # Robust colour range from valid pixels → better contrast than raw min/max
    valid_vals = depth[~combined_invalid]
    if valid_vals.size:
        vmin, vmax = np.percentile(valid_vals, robust_percentiles)
        if vmax <= vmin:
            vmin, vmax = float(valid_vals.min()), float(valid_vals.max()) or vmin + 1
    else:
        vmin, vmax = None, None

    cmap = plt.get_cmap(colormap)
    if invert:
        cmap = cmap.reversed()
    cmap = cmap.copy()
    cmap.set_bad(color="black")  # invalid fisheye border → black

    depth_title = "Predicted Depth (valid region)" if valid_mask is not None else "Predicted Depth"

    if rgb_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("RGB Image", fontsize=13)
        axes[0].axis("off")
        im = axes[1].imshow(depth_display, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(depth_title, fontsize=13)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(depth_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(depth_title, fontsize=13)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mask_visualization(mask: np.ndarray, output_path: Path) -> None:
    """Save a binary valid-region mask as a grayscale PNG (white=valid, black=invalid)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask * 255)
