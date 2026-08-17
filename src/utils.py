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
from matplotlib.colors import to_rgb


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


def create_common_valid_depth_mask(
    *depth_maps: np.ndarray,
    masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    min_depth: float = 0.0,
) -> np.ndarray:
    """Return the intersection where every aligned depth source is valid.

    This prevents model comparisons from using different pixel populations.
    A pixel is retained only when every depth map is finite and greater than
    ``min_depth``, and when every optional sensor/model mask is non-zero.
    """
    if not depth_maps:
        raise ValueError("At least one depth map is required")

    shape = depth_maps[0].shape
    common = np.ones(shape, dtype=bool)
    for index, depth in enumerate(depth_maps):
        if depth.shape != shape:
            raise ValueError(
                f"Depth map {index} has shape {depth.shape}; expected {shape}"
            )
        common &= np.isfinite(depth) & (depth > min_depth)

    for index, mask in enumerate(masks or ()):
        if mask.shape != shape:
            raise ValueError(f"Mask {index} has shape {mask.shape}; expected {shape}")
        common &= mask.astype(bool)

    return common.astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth colorization
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    colormap: str = "inferno",
    invert: bool = False,
    invalid_color: str = "#808080",
) -> np.ndarray:
    """Normalize and colorize a depth map, computing min/max from valid pixels only.

    Args:
        depth: float32 (H, W) depth array. May contain NaN for invalid pixels.
        valid_mask: optional uint8/bool (H, W) mask — 1=valid, 0=invalid.
                    Invalid pixels are rendered black (0, 0, 0).
        colormap: matplotlib colormap name.
        invert: reverse the colormap so near pixels are bright (use for metric
                depth, where near = small value).
        invalid_color: color for masked/no-prediction pixels. Neutral gray is
                       deliberately distinct from valid far depth, which is black
                       in the reversed inferno colormap.

    Returns:
        uint8 (H, W, 3) RGB colorized depth image.
    """
    effective_mask = np.isfinite(depth)
    if valid_mask is not None:
        effective_mask &= valid_mask.astype(bool)

    invalid_rgb = np.rint(np.asarray(to_rgb(invalid_color)) * 255).astype(np.uint8)
    d_valid = depth[effective_mask]
    if d_valid.size == 0:
        return np.broadcast_to(invalid_rgb, (*depth.shape, 3)).copy()

    d_min, d_max = float(d_valid.min()), float(d_valid.max())
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    if d_max > d_min:
        depth_norm[effective_mask] = (depth[effective_mask] - d_min) / (d_max - d_min)

    cmap = plt.get_cmap(colormap)
    if invert:
        cmap = cmap.reversed()
    colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)  # RGB
    colored[~effective_mask] = invalid_rgb
    return colored


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def depth_visualization_limits(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    robust_percentiles: tuple[float, float] = (2.0, 98.0),
    value_range: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """Resolve a reproducible color range for a depth visualization.

    ``value_range`` should be supplied for camera/model comparisons so the
    same color always represents the same depth. When omitted, robust limits
    are estimated from the current image for higher-contrast exploration.
    """
    if value_range is not None:
        vmin, vmax = (float(value) for value in value_range)
        if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
            raise ValueError(
                "value_range must contain finite values with minimum < maximum"
            )
        return vmin, vmax

    valid = np.isfinite(depth)
    if valid_mask is not None:
        valid &= valid_mask.astype(bool)
    values = np.asarray(depth)[valid]
    if values.size == 0:
        return None, None

    vmin, vmax = np.percentile(values, robust_percentiles)
    if vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_depth_visualization(
    depth: np.ndarray,
    output_path: Path,
    rgb_image: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    colormap: str = "inferno",
    invert: bool = False,
    robust_percentiles: tuple[float, float] = (2.0, 98.0),
    value_range: tuple[float, float] | None = None,
    depth_unit: str | None = None,
    invalid_color: str = "#808080",
) -> None:
    """Save a colorized depth map as a PNG, optionally side-by-side with the RGB image.

    With no explicit range, the colormap limits are computed from valid pixels
    only, so an invalid fisheye border cannot skew the scale. Supply the same
    ``value_range`` to every plot used in a comparison. Invalid pixels are gray.

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
        value_range: optional fixed (minimum, maximum) display range. Use this
                for comparable plots across cameras, frames, or models.
        depth_unit: optional unit shown on the colorbar, e.g. ``"m"`` for
                metric depth or ``"a.u."`` for relative depth.
        invalid_color: color for masked/no-prediction pixels. The default gray
                cannot be confused with valid far depth, which renders black.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a masked array so matplotlib computes colorbar range from valid pixels only
    combined_invalid = ~np.isfinite(depth)
    if valid_mask is not None:
        combined_invalid |= ~valid_mask.astype(bool)
    depth_display = ma.array(depth, mask=combined_invalid)

    vmin, vmax = depth_visualization_limits(
        depth,
        valid_mask=valid_mask,
        robust_percentiles=robust_percentiles,
        value_range=value_range,
    )

    cmap = plt.get_cmap(colormap)
    if invert:
        cmap = cmap.reversed()
    cmap = cmap.copy()
    cmap.set_bad(color=invalid_color)

    depth_title = (
        "Predicted Depth (gray = invalid)"
        if valid_mask is not None
        else "Predicted Depth"
    )

    if rgb_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("RGB Image", fontsize=13)
        axes[0].axis("off")
        im = axes[1].imshow(depth_display, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(depth_title, fontsize=13)
        axes[1].axis("off")
        colorbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(depth_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(depth_title, fontsize=13)
        ax.axis("off")
        colorbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if depth_unit:
        colorbar.set_label(f"Depth ({depth_unit})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mask_visualization(mask: np.ndarray, output_path: Path) -> None:
    """Save a binary valid-region mask as a grayscale PNG (white=valid, black=invalid)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask * 255)
