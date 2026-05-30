#!/usr/bin/env python3
"""Monocular depth estimation pipeline — loads one RGB image and saves a depth visualization.

Usage:
    python run_depth.py --data_dir /path/to/cv_project_data --output_dir outputs

Examples:
    # ZED_B perspective camera, first frame
    python run_depth.py --data_dir ../  --output_dir outputs

    # G1_A fisheye camera, frame index 5, recording2
    python run_depth.py --data_dir ../ --sensor G1_A --recording recording2 --image_index 5

    # Larger encoder (slower but sharper)
    python run_depth.py --data_dir ../ --encoder base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run monocular depth estimation on a single RGB image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help=(
            "Root data directory containing intrinsic.json, extrinsics.json, "
            "and recording1..4 folders."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where depth visualizations and raw arrays are saved.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default="recording1",
        help="Recording folder to use (e.g. recording1, recording2, ...).",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="ZED_B",
        choices=["ZED_B", "G1_A"],
        help="RGB camera sensor: 'ZED_B' (perspective) or 'G1_A' (fisheye).",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=0,
        help="Zero-based index into the sorted list of images for this sensor.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth_anything_v2",
        help="Depth model identifier (see src/depth_models.py).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Encoder size for Depth Anything V2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.utils import (
        load_intrinsics,
        load_extrinsics,
        find_rgb_images,
        save_depth_visualization,
    )
    from src.depth_models import build_depth_model

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    intrinsics = load_intrinsics(args.data_dir / "intrinsic.json")
    extrinsics = load_extrinsics(args.data_dir / "extrinsics.json")
    print(f"Intrinsics loaded for: {list(intrinsics.keys())}")
    print(f"Extrinsics loaded for: {list(extrinsics.keys())}")

    if args.sensor in intrinsics:
        cam = intrinsics[args.sensor]
        K = cam["K"]
        print(
            f"Camera model: {cam['model']}  |  "
            f"fx={K[0][0]:.1f}  fy={K[1][1]:.1f}  "
            f"cx={K[0][2]:.1f}  cy={K[1][2]:.1f}"
        )

    # ------------------------------------------------------------------
    # Find and load the target image
    # ------------------------------------------------------------------
    images = find_rgb_images(args.data_dir, sensor_name=args.sensor, recording=args.recording)
    print(f"Found {len(images)} images for '{args.sensor}' in '{args.recording}'")

    if args.image_index >= len(images):
        print(
            f"Error: --image_index {args.image_index} is out of range "
            f"(0 – {len(images) - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    image_path = images[args.image_index]
    print(f"Image: {image_path.name}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: cv2.imread failed for: {image_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded image: {image.shape[1]}×{image.shape[0]} px")

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------
    model = build_depth_model(args.model, encoder=args.encoder)
    model.load()

    depth = model.predict(image)
    print(
        f"Depth map: {depth.shape[1]}×{depth.shape[0]} px  "
        f"range [{depth.min():.3f}, {depth.max():.3f}]"
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    stem = f"{args.recording}_{args.sensor}_{args.image_index:04d}"
    output_dir = Path(args.output_dir)

    # Colorized side-by-side visualization
    vis_path = output_dir / f"{stem}_depth.png"
    save_depth_visualization(depth, vis_path, rgb_image=image)
    print(f"Saved visualization: {vis_path}")

    # Raw depth array for later comparison with stereo / LiDAR
    raw_path = output_dir / f"{stem}_depth_raw.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, depth)
    print(f"Saved raw depth: {raw_path}")


if __name__ == "__main__":
    main()
