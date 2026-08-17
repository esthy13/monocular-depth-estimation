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
        help=(
            "Depth model identifier. Options: 'depth_anything_v2', 'dac', "
            "or a full DAC variant like 'dac-outdoor-resnet101'."
        ),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Encoder size for Depth Anything V2 (ignored for DAC).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "DAC model variant, e.g. 'dac-outdoor-resnet101'. "
            "Only used when --model dac. If omitted, defaults to 'dac-outdoor-resnet101'."
        ),
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
    parser.add_argument(
        "--evaluate_lidar",
        action="store_true",
        help="Project the timestamp-matched LiDAR cloud and evaluate the predicted depth.",
    )
    parser.add_argument(
        "--lidar_sensor",
        type=str,
        default=None,
        help="LiDAR sensor key in the recording and extrinsics.json (required with --evaluate_lidar).",
    )
    parser.add_argument(
        "--max_lidar_dt",
        type=float,
        default=0.05,
        help="Maximum RGB-to-LiDAR timestamp difference in seconds.",
    )
    parser.add_argument(
        "--extrinsics_convention",
        choices=["sensor_to_reference", "reference_to_sensor"],
        default="sensor_to_reference",
        help="Direction used by matrices in extrinsics.json.",
    )
    parser.add_argument(
        "--alignment",
        choices=["auto", "none", "median", "least_squares"],
        default="auto",
        help="Metric alignment for sparse LiDAR evaluation. 'auto' uses least-squares scaling for relative models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import math

    from src.utils import (
        load_intrinsics,
        load_extrinsics,
        find_rgb_images,
        create_fisheye_valid_mask,
        save_depth_visualization,
        save_mask_visualization,
        intrinsics_to_dac_cam_params,
        find_sensor_npy_files,
        match_by_timestamp,
        camera_from_lidar_transform,
    )
    from src.depth_models import build_depth_model
    from src.lidar_evaluation import (
        evaluate_depth,
        keep_nearest_per_pixel,
        load_lidar_points,
        project_perspective_points,
        sample_bilinear,
        save_metrics,
        save_point_samples,
        save_projection_overlay,
        transform_points,
        align_depth,
    )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    intrinsics = load_intrinsics(args.data_dir / "intrinsic.json")
    extrinsics = load_extrinsics(args.data_dir / "extrinsics.json")
    print(f"Intrinsics loaded for: {list(intrinsics.keys())}")
    print(f"Extrinsics loaded for: {list(extrinsics.keys())}")

    # Determine camera model type early — used by both DAC config and fisheye masking
    cam_model_type = intrinsics.get(args.sensor, {}).get("model", "perspective")

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
    is_dac = args.model in ("dac",) or args.model.startswith("dac-")
    model_kwargs: dict = {}

    if is_dac:
        # Build DAC camera params from the loaded intrinsics
        cam_params = intrinsics_to_dac_cam_params(args.sensor, intrinsics)
        # Compute crop FoV: 180° for fisheye, horizontal FoV for perspective
        if cam_model_type == "fisheye":
            crop_wfov = 180.0
        else:
            fx = intrinsics[args.sensor]["K"][0][0]
            img_w = image.shape[1]
            crop_wfov = math.degrees(2.0 * math.atan(img_w / (2.0 * fx)))
        model_kwargs = {
            "cam_params": cam_params,
            "crop_wfov": crop_wfov,
        }
        if args.variant:
            model_kwargs["variant"] = args.variant
        print(f"DAC cam_params: {cam_params['camera_model']}  crop_wFov={crop_wfov:.1f}°")
    else:
        model_kwargs = {"encoder": args.encoder}

    model = build_depth_model(args.model, **model_kwargs)
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
    apply_mask = (args.fisheye_mask == "auto" and cam_model_type == "fisheye")

    if apply_mask:
        print("Generating fisheye valid-region mask ...")
        # Use the camera principal point as the lens-circle centre when available
        K = intrinsics.get(args.sensor, {}).get("K")
        center = (K[0][2], K[1][2]) if K is not None else None
        valid_mask = create_fisheye_valid_mask(image, center=center)
        # Metric models mark unfilled pixels as 0; fold those into the invalid
        # region so they aren't colored as "nearest" inside the lens circle.
        if getattr(model, "is_metric", False):
            valid_mask = (valid_mask.astype(bool) & (depth > 0)).astype(np.uint8)
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

    # 2. Side-by-side depth visualization (colormap range = valid pixels only).
    # Invert the colormap for metric models so near pixels render bright, matching
    # the inverse-depth convention of Depth Anything (near = bright).
    vis_path = output_dir / f"{stem}_depth.png"
    save_depth_visualization(
        depth, vis_path, rgb_image=image, valid_mask=valid_mask,
        invert=getattr(model, "is_metric", False),
    )
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

    # ------------------------------------------------------------------
    # Sparse LiDAR evaluation (perspective cameras only)
    # ------------------------------------------------------------------
    if args.evaluate_lidar:
        if args.lidar_sensor is None:
            raise ValueError("--evaluate_lidar requires --lidar_sensor (the LiDAR key in extrinsics.json).")
        if cam_model_type != "perspective":
            raise ValueError("LiDAR evaluation currently supports perspective cameras only.")

        lidar_files = find_sensor_npy_files(args.data_dir, args.lidar_sensor, args.recording)
        matched = match_by_timestamp([image_path], lidar_files, max_dt=args.max_lidar_dt)
        if not matched:
            raise FileNotFoundError(
                f"No {args.lidar_sensor} point cloud within {args.max_lidar_dt:.3f}s of {image_path.name}."
            )
        _, lidar_path = matched[0]
        print(f"LiDAR cloud: {lidar_path.name}")

        lidar_points = load_lidar_points(lidar_path)
        T_camera_lidar = camera_from_lidar_transform(
            extrinsics, args.sensor, args.lidar_sensor, args.extrinsics_convention
        )
        points_camera = transform_points(lidar_points, T_camera_lidar)
        pixels, lidar_depth, _ = project_perspective_points(
            points_camera, intrinsics[args.sensor], image.shape[:2]
        )
        nearest = keep_nearest_per_pixel(pixels, lidar_depth)
        pixels, lidar_depth = pixels[nearest], lidar_depth[nearest]
        if not len(pixels):
            raise ValueError("No forward-facing LiDAR points project inside this image; check the extrinsics convention.")

        projection_path = output_dir / f"{stem}_lidar_projection.png"
        save_projection_overlay(image, pixels, lidar_depth, projection_path)
        print(f"Saved LiDAR projection: {projection_path} ({len(pixels)} visible points)")

        sampled_prediction = sample_bilinear(depth, pixels)
        valid = np.isfinite(sampled_prediction) & (sampled_prediction > 0) & np.isfinite(lidar_depth) & (lidar_depth > 0)
        pixels, lidar_depth, sampled_prediction = pixels[valid], lidar_depth[valid], sampled_prediction[valid]
        if not len(sampled_prediction):
            raise ValueError("No projected LiDAR points have valid predicted depth samples.")

        # Depth Anything V2 emits a relative inverse-depth map. Its reciprocal is
        # a distance-like proxy; a LiDAR scale is essential before metre errors.
        prediction_proxy = sampled_prediction if getattr(model, "is_metric", False) else 1.0 / sampled_prediction
        alignment = args.alignment
        if alignment == "auto":
            alignment = "none" if getattr(model, "is_metric", False) else "least_squares"
        result = evaluate_depth(prediction_proxy, lidar_depth, alignment)
        aligned_prediction, _, _ = align_depth(prediction_proxy, lidar_depth, alignment)

        metrics_path = output_dir / f"{stem}_lidar_metrics.json"
        samples_path = output_dir / f"{stem}_lidar_samples.csv"
        save_metrics(
            metrics_path, result,
            image=str(image_path), lidar_cloud=str(lidar_path), camera_sensor=args.sensor,
            lidar_sensor=args.lidar_sensor, timestamp_tolerance_s=args.max_lidar_dt,
            extrinsics_convention=args.extrinsics_convention,
            prediction_representation=("metric_depth_m" if getattr(model, "is_metric", False) else "reciprocal_relative_depth"),
        )
        save_point_samples(samples_path, pixels, lidar_depth, prediction_proxy, aligned_prediction)
        print(f"LiDAR metrics ({result.count} points): MAE={result.mae_m:.3f} m, RMSE={result.rmse_m:.3f} m, AbsRel={result.abs_rel:.3f}")
        print(f"Saved metrics:        {metrics_path}")
        print(f"Saved point samples:  {samples_path}")


if __name__ == "__main__":
    main()
