#!/usr/bin/env python3
"""Monocular depth estimation pipeline — loads one RGB image and saves a depth visualization.

Usage:
    python run_depth.py --data_dir /path/to/cv_project_data --output_dir outputs

Examples:
    # ZED_B perspective camera, first frame
    python run_depth.py --data_dir ../

    # G1_A fisheye camera — mask is applied automatically (camera model = fisheye)
    python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 5

    # Disable fisheye masking explicitly
    python run_depth.py --data_dir ../ --sensor G1_A --fisheye_mask none

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
    parser.add_argument(
        "--fisheye_mask",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help=(
            "'auto': apply fisheye valid-region mask when camera model is fisheye "
            "(determined from intrinsics). 'none': skip masking."
        ),
    )
    parser.add_argument(
        "--invalid_value",
        type=str,
        default="nan",
        choices=["nan", "zero"],
        help="Value written to invalid (masked) depth pixels (default: nan).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.utils import (
        load_intrinsics,
        load_extrinsics,
        find_rgb_images,
        create_fisheye_valid_mask,
        save_depth_visualization,
        save_mask_visualization,
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
        f"raw range [{depth.min():.3f}, {depth.max():.3f}]"
    )

    # ------------------------------------------------------------------
    # Fisheye mask (suppresses invalid lens-border pixels)
    # ------------------------------------------------------------------
    valid_mask = None
    cam_model_type = intrinsics.get(args.sensor, {}).get("model", "perspective")
    apply_mask = (args.fisheye_mask == "auto" and cam_model_type == "fisheye")

    if apply_mask:
        print("Generating fisheye valid-region mask ...")
        valid_mask = create_fisheye_valid_mask(image)
        n_valid = int(valid_mask.sum())
        print(
            f"Valid pixels: {n_valid} / {valid_mask.size} "
            f"({n_valid / valid_mask.size * 100:.1f}%)"
        )
        # Write invalid pixels with the chosen sentinel so they don't affect
        # any downstream statistics or comparisons with stereo/LiDAR
        invalid = ~valid_mask.astype(bool)
        if args.invalid_value == "nan":
            depth[invalid] = np.nan
        else:
            depth[invalid] = 0.0
        valid_range = depth[valid_mask.astype(bool)]
        print(
            f"Depth range (valid only): "
            f"[{np.nanmin(valid_range):.3f}, {np.nanmax(valid_range):.3f}]"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    stem = f"{args.recording}_{args.sensor}_{args.image_index:04d}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original RGB image
    rgb_path = output_dir / f"{stem}_rgb.jpg"
    cv2.imwrite(str(rgb_path), image)
    print(f"Saved RGB image:      {rgb_path}")

    # 2. Side-by-side depth visualization (colormap range = valid pixels only)
    vis_path = output_dir / f"{stem}_depth.png"
    save_depth_visualization(depth, vis_path, rgb_image=image, valid_mask=valid_mask)
    print(f"Saved visualization:  {vis_path}")

    # 3. Binary valid-region mask (only written when masking was applied)
    if valid_mask is not None:
        mask_path = output_dir / f"{stem}_mask.png"
        save_mask_visualization(valid_mask, mask_path)
        print(f"Saved valid mask:     {mask_path}")

    # 4. Raw depth array for later comparison with stereo / LiDAR
    raw_path = output_dir / f"{stem}_depth_raw.npy"
    np.save(raw_path, depth)
    print(f"Saved raw depth:      {raw_path}")


if __name__ == "__main__":
    main()
