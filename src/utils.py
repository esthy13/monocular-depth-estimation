"""Utility functions for loading calibration data, finding images, and saving results."""
from __future__ import annotations

import bisect
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
# Visualization
# ---------------------------------------------------------------------------

def save_depth_visualization(
    depth: np.ndarray,
    output_path: Path,
    rgb_image: np.ndarray | None = None,
    colormap: str = "inferno",
) -> None:
    """Save a colorized depth map as a PNG, optionally side-by-side with the RGB image.

    Args:
        depth: float32 depth array (H, W).
        output_path: destination .png file path.
        rgb_image: optional BGR uint8 image (H, W, 3) to place left of the depth.
        colormap: matplotlib colormap name (default 'inferno').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rgb_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("RGB Image", fontsize=13)
        axes[0].axis("off")
        im = axes[1].imshow(depth, cmap=colormap)
        axes[1].set_title("Predicted Depth (relative)", fontsize=13)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(depth, cmap=colormap)
        ax.set_title("Predicted Depth (relative)", fontsize=13)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
