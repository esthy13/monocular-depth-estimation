#!/usr/bin/env python3
"""Project synchronized LiDAR returns into calibrated G1_A fisheye images."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.evaluation import nearest_timestamp_file, project_lidar_to_fisheye
from src.lidar_evaluation import keep_nearest_per_pixel, load_lidar_points
from src.utils import find_rgb_images, load_extrinsics, load_intrinsics, parse_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Timestamp-match E1 LiDAR clouds and project them into the "
            "calibrated G1_A fisheye image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--recording", default="recording1")
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum selected frames; 0 means no limit.",
    )
    parser.add_argument(
        "--lidar_sensors",
        nargs="+",
        default=["E1_A", "E1_B"],
        choices=["E1_A", "E1_B"],
    )
    parser.add_argument("--lidar_max_dt", type=float, default=0.05)
    parser.add_argument(
        "--lidar_time_offset_s",
        type=float,
        default=0.0,
        help="Offset added to each LiDAR timestamp before camera matching.",
    )
    parser.add_argument("--max_range_m", type=float, default=50.0)
    parser.add_argument(
        "--visualization_range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=(0.5, 10.0),
    )
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument(
        "--extrinsics_convention",
        choices=["sensor_to_reference", "reference_to_sensor"],
        default="sensor_to_reference",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_commit(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def selected_indices(
    image_count: int,
    image_index: int,
    all_frames: bool,
    frame_step: int,
    max_frames: int,
) -> list[int]:
    if image_count < 1:
        return []
    if frame_step < 1:
        raise ValueError("--frame_step must be at least 1")
    if max_frames < 0:
        raise ValueError("--max_frames must not be negative")
    if all_frames:
        indices = list(range(0, image_count, frame_step))
        return indices[:max_frames] if max_frames else indices
    if image_index < 0 or image_index >= image_count:
        raise IndexError(
            f"--image_index {image_index} is outside the available range "
            f"0..{image_count - 1}"
        )
    return [image_index]


def sensor_to_reference_transform(
    extrinsics: dict,
    sensor: str,
    convention: str,
) -> np.ndarray:
    transform = np.asarray(extrinsics[sensor], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform for {sensor}; got {transform.shape}")
    return transform if convention == "sensor_to_reference" else np.linalg.inv(transform)


def validate_existing_metadata(payload: dict, expected: dict, path: Path) -> None:
    """Prevent a resumable run from silently mixing projection protocols."""
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise ValueError(
                f"{path}: existing {field}={payload.get(field)!r}, expected "
                f"{expected_value!r}; rerun with --overwrite or use another output directory"
            )


def save_point_table(
    path: Path,
    pixels: np.ndarray,
    ranges: np.ndarray,
    sensors: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("sensor", "u_px", "v_px", "lidar_range_m"))
        for sensor, pixel, distance in zip(sensors, pixels, ranges):
            writer.writerow((sensor, float(pixel[0]), float(pixel[1]), float(distance)))


def save_projection_figure(
    image: np.ndarray,
    pixels: np.ndarray,
    ranges: np.ndarray,
    output_path: Path,
    visualization_range: tuple[float, float],
    point_size: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    minimum, maximum = visualization_range
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("G1_A RGB")
    axes[1].imshow(rgb)
    axes[1].set_title(title)
    if len(ranges):
        axes[1].scatter(
            pixels[:, 0],
            pixels[:, 1],
            c=ranges,
            cmap="turbo",
            vmin=minimum,
            vmax=maximum,
            s=point_size,
            linewidths=0,
            alpha=0.9,
        )
    for axis in axes:
        axis.set_axis_off()
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(minimum, maximum), cmap="turbo"),
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("LiDAR Euclidean range (m)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def process_frame(
    image_index: int,
    image_path: Path,
    lidar_files: dict[str, list[Path]],
    output_dir: Path,
    recording: str,
    target_camera_matrix: np.ndarray,
    target_distortion: np.ndarray,
    extrinsics: dict,
    lidar_sensors: list[str],
    lidar_max_dt: float,
    lidar_time_offset_s: float,
    max_range_m: float,
    visualization_range: tuple[float, float],
    point_size: float,
    extrinsics_convention: str,
    commit: str | None,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV could not read {image_path}")
    camera_timestamp = parse_timestamp(image_path)
    frame_stem = f"{recording}_G1_A_{image_index:06d}_lidar_fisheye"

    all_pixels: list[np.ndarray] = []
    all_ranges: list[np.ndarray] = []
    all_sensors: list[np.ndarray] = []
    match_metadata: dict[str, dict | None] = {}
    for sensor in lidar_sensors:
        match = nearest_timestamp_file(
            lidar_files[sensor],
            camera_timestamp - lidar_time_offset_s,
            lidar_max_dt,
        )
        if match is None:
            match_metadata[sensor] = None
            continue
        points = load_lidar_points(match.path)
        pixels, ranges = project_lidar_to_fisheye(
            points,
            sensor_to_reference_transform(
                extrinsics, sensor, extrinsics_convention
            ),
            target_camera_matrix,
            target_distortion,
            image.shape[:2],
            max_range_metres=max_range_m,
        )
        all_pixels.append(pixels)
        all_ranges.append(ranges)
        all_sensors.append(np.full(len(ranges), sensor, dtype=object))
        match_metadata[sensor] = {
            "file": match.path.name,
            "timestamp_seconds": parse_timestamp(match.path),
            "timestamp_delta_s": match.delta_seconds + lidar_time_offset_s,
            "finite_points": int(len(points)),
            "projected_in_frame_points_before_z_buffer": int(len(ranges)),
        }

    pixels = (
        np.concatenate(all_pixels, axis=0)
        if all_pixels
        else np.empty((0, 2), dtype=np.float32)
    )
    ranges = (
        np.concatenate(all_ranges)
        if all_ranges
        else np.empty(0, dtype=np.float32)
    )
    sensor_labels = (
        np.concatenate(all_sensors)
        if all_sensors
        else np.empty(0, dtype=object)
    )
    nearest = keep_nearest_per_pixel(pixels, ranges)
    pixels, ranges, sensor_labels = (
        pixels[nearest],
        ranges[nearest],
        sensor_labels[nearest],
    )
    for sensor in lidar_sensors:
        if match_metadata[sensor] is not None:
            match_metadata[sensor]["visible_points_after_z_buffer"] = int(
                np.count_nonzero(sensor_labels == sensor)
            )

    save_projection_figure(
        image,
        pixels,
        ranges,
        output_dir / f"{frame_stem}_projection.png",
        visualization_range,
        point_size,
        (
            f"LiDAR projected into G1_A fisheye "
            f"({', '.join(lidar_sensors)}; "
            f"{visualization_range[0]:g}–{visualization_range[1]:g} m)"
        ),
    )
    save_point_table(
        output_dir / f"{frame_stem}_points.csv",
        pixels,
        ranges,
        sensor_labels,
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": commit,
        "recording": recording,
        "image_index": image_index,
        "camera_sensor": "G1_A",
        "image": str(image_path.resolve()),
        "camera_timestamp_seconds": camera_timestamp,
        "lidar_sensors": lidar_sensors,
        "lidar_matches": match_metadata,
        "lidar_time_offset_s": lidar_time_offset_s,
        "lidar_max_dt_s": lidar_max_dt,
        "max_range_m": max_range_m,
        "visualization_range_m": list(visualization_range),
        "extrinsics_convention": extrinsics_convention,
        "G1_A_camera_matrix": target_camera_matrix.tolist(),
        "G1_A_distortion": target_distortion.tolist(),
        "sensor_to_G1_A_transforms": {
            sensor: sensor_to_reference_transform(
                extrinsics, sensor, extrinsics_convention
            ).tolist()
            for sensor in lidar_sensors
        },
        "projection_model": "OpenCV fisheye with calibrated G1_A K and distortion",
        "range_definition": "Euclidean distance from the G1_A camera origin",
        "visible_points_after_z_buffer": int(len(ranges)),
    }
    (output_dir / f"{frame_stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def summarize_frames(frames: list[dict], commit: str | None) -> dict:
    sensors = sorted(
        {
            sensor
            for frame in frames
            for sensor in frame["lidar_sensors"]
        }
    )
    sensor_summaries = {}
    for sensor in sensors:
        matches = [
            frame["lidar_matches"][sensor]
            for frame in frames
            if frame["lidar_matches"].get(sensor) is not None
        ]
        offsets = np.asarray(
            [match["timestamp_delta_s"] for match in matches], dtype=float
        )
        sensor_summaries[sensor] = {
            "matched_frames": len(matches),
            "timestamp_dt_min_s": float(np.min(offsets)) if len(offsets) else None,
            "timestamp_dt_max_s": float(np.max(offsets)) if len(offsets) else None,
            "timestamp_dt_mean_s": float(np.mean(offsets)) if len(offsets) else None,
            "timestamp_dt_median_s": float(np.median(offsets)) if len(offsets) else None,
            "timestamp_dt_std_s": float(np.std(offsets)) if len(offsets) else None,
            "projected_points_before_z_buffer": sum(
                match["projected_in_frame_points_before_z_buffer"]
                for match in matches
            ),
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": commit,
        "recording": frames[0]["recording"] if frames else None,
        "camera_sensor": "G1_A",
        "lidar_sensors": sensors,
        "processed_frames": len(frames),
        "visible_points_after_z_buffer": sum(
            frame["visible_points_after_z_buffer"] for frame in frames
        ),
        "sensors": sensor_summaries,
    }


def main() -> None:
    args = parse_args()
    minimum, maximum = map(float, args.visualization_range)
    if not (0 <= minimum < maximum):
        raise ValueError("--visualization_range must satisfy 0 <= MIN_M < MAX_M")
    if args.max_range_m <= 0:
        raise ValueError("--max_range_m must be positive")
    if args.lidar_max_dt < 0:
        raise ValueError("--lidar_max_dt must not be negative")
    if args.point_size <= 0:
        raise ValueError("--point_size must be positive")

    intrinsics = load_intrinsics(args.data_dir / "intrinsic.json")
    extrinsics = load_extrinsics(args.data_dir / "extrinsics.json")
    if extrinsics.get("G1_A") is not None:
        raise ValueError("Expected G1_A to be the reference sensor")
    g1 = intrinsics["G1_A"]
    target_camera_matrix = np.asarray(g1["K"], dtype=np.float64)
    target_distortion = np.asarray(g1["dist"], dtype=np.float64)
    images = find_rgb_images(args.data_dir, "G1_A", args.recording)
    indices = selected_indices(
        len(images),
        args.image_index,
        args.all_frames,
        args.frame_step,
        args.max_frames,
    )
    lidar_sensors = list(dict.fromkeys(args.lidar_sensors))
    lidar_files = {
        sensor: sorted(
            (args.data_dir / args.recording / "data" / sensor / sensor).glob("*.npy")
        )
        for sensor in lidar_sensors
    }
    for sensor, files in lidar_files.items():
        if not files:
            raise FileNotFoundError(f"No LiDAR arrays found for {sensor}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = project_commit(Path(__file__).resolve().parent)
    frame_summaries = []
    for position, index in enumerate(indices, start=1):
        metadata_path = (
            args.output_dir
            / f"{args.recording}_G1_A_{index:06d}_lidar_fisheye_metadata.json"
        )
        if metadata_path.is_file() and not args.overwrite:
            existing = json.loads(metadata_path.read_text())
            validate_existing_metadata(
                existing,
                {
                    "recording": args.recording,
                    "image_index": index,
                    "lidar_sensors": lidar_sensors,
                    "lidar_time_offset_s": args.lidar_time_offset_s,
                    "lidar_max_dt_s": args.lidar_max_dt,
                    "max_range_m": args.max_range_m,
                    "visualization_range_m": [minimum, maximum],
                    "extrinsics_convention": args.extrinsics_convention,
                },
                metadata_path,
            )
            frame_summaries.append(existing)
            print(f"[{position}/{len(indices)}] frame {index}: existing output")
            continue
        frame_summaries.append(
            process_frame(
                index,
                images[index],
                lidar_files,
                args.output_dir,
                args.recording,
                target_camera_matrix,
                target_distortion,
                extrinsics,
                lidar_sensors,
                args.lidar_max_dt,
                args.lidar_time_offset_s,
                args.max_range_m,
                (minimum, maximum),
                args.point_size,
                args.extrinsics_convention,
                commit,
            )
        )
        print(
            f"[{position}/{len(indices)}] frame {index}: "
            f"{frame_summaries[-1]['visible_points_after_z_buffer']} visible points"
        )

    summary_path = args.output_dir / f"{args.recording}_G1_A_lidar_fisheye_summary.json"
    summary_path.write_text(
        json.dumps(summarize_frames(frame_summaries, commit), indent=2) + "\n"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
