#!/usr/bin/env python3
"""Create LiDAR residual and person-localization diagnostic plots."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def finite_float(value: str | None) -> float | None:
    """Return a finite float, or ``None`` for empty/non-finite CSV values."""
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_lidar_residual_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load projected pixels and signed ``prediction - LiDAR`` error."""
    pixels: list[tuple[float, float]] = []
    residuals: list[float] = []
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            u = finite_float(row.get("u_px"))
            v = finite_float(row.get("v_px"))
            prediction = finite_float(row.get("aligned_prediction_m"))
            lidar = finite_float(row.get("lidar_depth_m"))
            if None not in (u, v, prediction, lidar) and prediction > 0 and lidar > 0:
                pixels.append((u, v))
                residuals.append(prediction - lidar)
    if not pixels:
        raise ValueError(f"No valid LiDAR/prediction samples found in {path}.")
    return np.asarray(pixels, dtype=np.float64), np.asarray(residuals, dtype=np.float64)


def save_lidar_residual_overlay(
    image_path: Path,
    samples_path: Path,
    output_path: Path,
    error_limit_m: float = 2.0,
    title: str | None = None,
) -> None:
    """Overlay signed metric depth error at every evaluated LiDAR pixel."""
    if error_limit_m <= 0:
        raise ValueError("error_limit_m must be positive.")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image {image_path}.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pixels, residuals = load_lidar_residual_samples(samples_path)

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(image_rgb)
    points = axis.scatter(
        pixels[:, 0],
        pixels[:, 1],
        c=residuals,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-error_limit_m, vcenter=0.0, vmax=error_limit_m),
        s=12,
        alpha=0.9,
        linewidths=0,
    )
    axis.set_title(
        title
        or "Signed depth residual at projected LiDAR points\n"
        "blue: prediction closer | red: prediction farther"
    )
    axis.set_xlim(0, image_rgb.shape[1])
    axis.set_ylim(image_rgb.shape[0], 0)
    axis.axis("off")
    colorbar = figure.colorbar(points, ax=axis, fraction=0.035, pad=0.02, extend="both")
    colorbar.set_label("Prediction minus LiDAR depth (m)")
    figure.text(
        0.5,
        0.015,
        "Only pixels covered by projected LiDAR are evaluated; colors are clipped "
        f"to +/- {error_limit_m:g} m.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def reference_distance(row: dict[str, str], reference: str) -> float | None:
    """Return the ground-truth 3D range, preferring saved XYZ coordinates."""
    coordinates = [finite_float(row.get(f"{reference}_{axis}_m")) for axis in "xyz"]
    if all(value is not None for value in coordinates):
        return float(np.linalg.norm(np.asarray(coordinates, dtype=np.float64)))
    fallback = "lidar_median_m" if reference == "lidar" else "zed_reference_median_m"
    return finite_float(row.get(fallback))


def load_person_localization_samples(
    path: Path,
    reference: str = "lidar",
) -> tuple[np.ndarray, np.ndarray]:
    """Load ground-truth person range and corresponding 3D localization error."""
    if reference not in {"lidar", "zed"}:
        raise ValueError("reference must be 'lidar' or 'zed'.")
    distances: list[float] = []
    errors: list[float] = []
    error_key = f"{reference}_3d_error_m"
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            distance = reference_distance(row, reference)
            error = finite_float(row.get(error_key))
            if distance is not None and error is not None and distance > 0 and error >= 0:
                distances.append(distance)
                errors.append(error)
    if not distances:
        raise ValueError(f"No valid {reference} localization samples found in {path}.")
    return np.asarray(distances), np.asarray(errors)


def binned_medians(
    distances: np.ndarray,
    errors: np.ndarray,
    bin_width_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return occupied-bin centers, median errors, and sample counts."""
    if bin_width_m <= 0:
        raise ValueError("bin_width_m must be positive.")
    upper = max(
        bin_width_m,
        (math.floor(float(np.max(distances)) / bin_width_m) + 1) * bin_width_m,
    )
    edges = np.arange(0.0, upper + bin_width_m * 0.5, bin_width_m)
    indices = np.digitize(distances, edges, right=False) - 1
    centers: list[float] = []
    medians: list[float] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        selected = errors[indices == index]
        if len(selected):
            centers.append(float((edges[index] + edges[index + 1]) / 2))
            medians.append(float(np.median(selected)))
            counts.append(int(len(selected)))
    return np.asarray(centers), np.asarray(medians), np.asarray(counts)


def parse_run(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=person_measurements.csv`` command-line values."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=/path/person_measurements.csv.")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both LABEL and path are required.")
    return label.strip(), Path(path).expanduser()


def save_localization_error_plot(
    runs: list[tuple[str, Path]],
    output_path: Path,
    reference: str = "lidar",
    bin_width_m: float = 1.0,
    title: str | None = None,
) -> None:
    """Plot 3D person localization error against physical-reference distance."""
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for label, path in runs:
        grouped[label].append(load_person_localization_samples(path, reference))
    if not grouped:
        raise ValueError("At least one --run is required.")

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    palette = plt.get_cmap("tab10")
    for index, (label, parts) in enumerate(grouped.items()):
        distances = np.concatenate([part[0] for part in parts])
        errors = np.concatenate([part[1] for part in parts])
        color = palette(index % 10)
        axis.scatter(
            distances,
            errors,
            s=18,
            alpha=0.25,
            color=color,
            linewidths=0,
        )
        centers, medians, _ = binned_medians(distances, errors, bin_width_m)
        axis.plot(
            centers,
            medians,
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label}: {bin_width_m:g} m-bin median (n={len(errors)})",
        )

    reference_name = "LiDAR" if reference == "lidar" else "stereo"
    axis.set_xlabel(f"Ground-truth person distance from {reference_name} (m)")
    axis.set_ylabel("3D person localization error (m)")
    axis.set_title(title or f"Person localization error vs {reference_name} distance")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create signed LiDAR residual and person-localization plots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    overlay = subparsers.add_parser("residual-overlay")
    overlay.add_argument("--image", type=Path, required=True)
    overlay.add_argument("--samples", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    overlay.add_argument("--error-limit-m", type=float, default=2.0)
    overlay.add_argument("--title")

    localization = subparsers.add_parser("localization-error")
    localization.add_argument("--run", type=parse_run, action="append", required=True)
    localization.add_argument("--reference", choices=("lidar", "zed"), default="lidar")
    localization.add_argument("--bin-width-m", type=float, default=1.0)
    localization.add_argument("--output", type=Path, required=True)
    localization.add_argument("--title")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "residual-overlay":
        save_lidar_residual_overlay(
            args.image,
            args.samples,
            args.output,
            error_limit_m=args.error_limit_m,
            title=args.title,
        )
    else:
        save_localization_error_plot(
            args.run,
            args.output,
            reference=args.reference,
            bin_width_m=args.bin_width_m,
            title=args.title,
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
