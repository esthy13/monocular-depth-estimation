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
) -> list[tuple[Path, Path]]:
    """Pair each source file with the nearest-timestamp target file.

    Returns only pairs where |ts_source - ts_target| <= max_dt seconds.
    """
    target_ts_pairs = sorted(
        (parse_timestamp(f), f) for f in target_files
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
    black_threshold: int = 15,
) -> np.ndarray:
    """Detect the valid circular lens region of a fisheye image.

    Outside the fisheye lens circle the border is near-black — it is invalid
    lens area, NOT real scene content. This mask isolates the valid circle so
    the black border cannot distort depth normalization or colormap range.

    This is a lens-boundary mask, NOT object segmentation.

    Strategy:
      1. Threshold near-black pixels → coarse binary map of potentially valid pixels.
      2. Morphological closing fills small dark patches inside the valid circle.
      3. Keep the largest connected component, which is the fisheye circle.

    Args:
        image: BGR uint8 image from a fisheye camera.
        black_threshold: pixels with grayscale value ≤ this are treated as
                         invalid border. Default 15 is robust to JPEG artifacts.

    Returns:
        uint8 (H, W) mask — 1 = valid scene pixel, 0 = invalid fisheye border.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pixels above the threshold might be real scene content
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    # Closing kernel sized ~1/30 of image fills dark objects/textures inside the
    # circle without connecting the valid circle to the outer black border
    k = max(image.shape[0], image.shape[1]) // 30
    k += 1 - (k % 2)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Retain only the largest connected component (the fisheye circle)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    # Label 0 is the background; find the largest non-background component
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest_label).astype(np.uint8)

    return mask


# ---------------------------------------------------------------------------
# Depth colorization
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    colormap: str = "inferno",
) -> np.ndarray:
    """Normalize and colorize a depth map, computing min/max from valid pixels only.

    Args:
        depth: float32 (H, W) depth array. May contain NaN for invalid pixels.
        valid_mask: optional uint8/bool (H, W) mask — 1=valid, 0=invalid.
                    Invalid pixels are rendered black (0, 0, 0).
        colormap: matplotlib colormap name.

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
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a masked array so matplotlib computes colorbar range from valid pixels only
    combined_invalid = ~np.isfinite(depth)
    if valid_mask is not None:
        combined_invalid |= ~valid_mask.astype(bool)
    depth_display = ma.array(depth, mask=combined_invalid)

    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(color="black")  # invalid fisheye border → black

    depth_title = "Predicted Depth (valid region)" if valid_mask is not None else "Predicted Depth"

    if rgb_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("RGB Image", fontsize=13)
        axes[0].axis("off")
        im = axes[1].imshow(depth_display, cmap=cmap)
        axes[1].set_title(depth_title, fontsize=13)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(depth_display, cmap=cmap)
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
