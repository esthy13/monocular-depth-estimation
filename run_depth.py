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
import csv
import json
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
        "--evaluate_lidar_all",
        action="store_true",
        help="Evaluate every RGB image that has a timestamp-matched LiDAR cloud and save aggregate metrics.",
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
        choices=["auto", "none", "median", "least_squares", "inverse_least_squares"],
        default="auto",
        help="Metric alignment for sparse LiDAR evaluation. 'auto' uses least-squares scaling for relative models.",
    )
    parser.add_argument("--debug_evaluation", action="store_true", help="Write per-frame LiDAR/depth diagnostics to CSV.")
    parser.add_argument("--plot_evaluation", action="store_true", help="Save evaluation plots under outputs/evaluation_plots/.")
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
        parse_timestamp,
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
    dataset_path = args.data_dir / args.recording / "dataset.json"
    if dataset_path.exists() and not json.loads(dataset_path.read_text()).get("calibrated_extrinsics", True):
        print("WARNING: dataset.json reports calibrated_extrinsics=false. Projection overlays are diagnostic only; absolute geometric validity requires calibrated LiDAR-to-camera extrinsics.")

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
    # Find images and pair them with LiDAR before inference.  In batch mode,
    # unmatched camera frames are expected (the sensors run at different rates).
    # ------------------------------------------------------------------
    images = find_rgb_images(args.data_dir, sensor_name=args.sensor, recording=args.recording)
    print(f"Found {len(images)} images for '{args.sensor}' in '{args.recording}'")
    if args.image_index < 0 or args.image_index >= len(images):
        print(f"Error: --image_index {args.image_index} is out of range (0 – {len(images) - 1}).", file=sys.stderr)
        sys.exit(1)

    batch_lidar = args.evaluate_lidar_all
    if batch_lidar and not args.evaluate_lidar:
        args.evaluate_lidar = True
    if args.evaluate_lidar and args.lidar_sensor is None:
        raise ValueError("--evaluate_lidar requires --lidar_sensor (the LiDAR key in extrinsics.json).")
    if args.evaluate_lidar and cam_model_type != "perspective":
        raise ValueError("LiDAR evaluation currently supports perspective cameras only.")

    requested_images = images if batch_lidar else [images[args.image_index]]
    lidar_pairs: dict[Path, Path] = {}
    if args.evaluate_lidar:
        lidar_files = find_sensor_npy_files(args.data_dir, args.lidar_sensor, args.recording)
        lidar_pairs = dict(match_by_timestamp(requested_images, lidar_files, max_dt=args.max_lidar_dt))
        if batch_lidar:
            print(f"LiDAR matches: {len(lidar_pairs)} / {len(requested_images)} images within {args.max_lidar_dt:.3f}s")
            requested_images = [path for path in requested_images if path in lidar_pairs]
            if not requested_images:
                raise FileNotFoundError(f"No {args.lidar_sensor} point clouds match any RGB images within {args.max_lidar_dt:.3f}s.")
        elif requested_images[0] not in lidar_pairs:
            raise FileNotFoundError(
                f"No {args.lidar_sensor} point cloud within {args.max_lidar_dt:.3f}s of {requested_images[0].name}. "
                "Use --evaluate_lidar_all to evaluate all timestamp-matched frames."
            )

    first_image = cv2.imread(str(requested_images[0]))
    if first_image is None:
        raise RuntimeError(f"cv2.imread failed for: {requested_images[0]}")

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
            img_w = first_image.shape[1]
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = args.alignment
    if alignment == "auto":
        alignment = "none" if getattr(model, "is_metric", False) else "inverse_least_squares"
    T_camera_lidar = camera_from_lidar_transform(extrinsics, args.sensor, args.lidar_sensor, args.extrinsics_convention) if args.evaluate_lidar else None
    global_predictions, global_ground_truth, debug_rows, frame_results = [], [], [], []

    for image_path in requested_images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
            continue
        image_index = images.index(image_path)
        print(f"\nImage {image_index:04d}: {image_path.name} ({image.shape[1]}×{image.shape[0]} px)")
        depth = model.predict(image)
        stem = f"{args.recording}_{args.sensor}_{image_index:04d}"
        valid_mask = None
        if args.fisheye_mask == "auto" and cam_model_type == "fisheye":
            K = intrinsics.get(args.sensor, {}).get("K")
            center = (K[0][2], K[1][2]) if K is not None else None
            valid_mask = create_fisheye_valid_mask(image, center=center)
            if getattr(model, "is_metric", False):
                valid_mask = (valid_mask.astype(bool) & (depth > 0)).astype(np.uint8)
            depth[~valid_mask.astype(bool)] = np.nan if args.invalid_value == "nan" else 0.0

        cv2.imwrite(str(output_dir / f"{stem}_rgb.jpg"), image)
        save_depth_visualization(depth, output_dir / f"{stem}_depth.png", rgb_image=image, valid_mask=valid_mask, invert=getattr(model, "is_metric", False))
        if valid_mask is not None:
            save_mask_visualization(valid_mask, output_dir / f"{stem}_mask.png")
        np.save(output_dir / f"{stem}_depth_raw.npy", depth)

        if not args.evaluate_lidar:
            continue
        lidar_path = lidar_pairs[image_path]
        raw_lidar_points = load_lidar_points(lidar_path)
        points_camera = transform_points(raw_lidar_points, T_camera_lidar)
        front_count = int(np.count_nonzero(np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0)))
        pixels, lidar_depth, _ = project_perspective_points(points_camera, intrinsics[args.sensor], image.shape[:2])
        nearest = keep_nearest_per_pixel(pixels, lidar_depth)
        pixels, lidar_depth = pixels[nearest], lidar_depth[nearest]
        sampled_prediction = sample_bilinear(depth, pixels) if len(pixels) else np.empty(0)
        valid = np.isfinite(sampled_prediction) & (sampled_prediction > 0) & np.isfinite(lidar_depth) & (lidar_depth > 0)
        pixels, lidar_depth, sampled_prediction = pixels[valid], lidar_depth[valid], sampled_prediction[valid]
        if len(sampled_prediction) < 2:
            print("Skipping frame: fewer than two valid projected LiDAR samples.")
            continue
        # Depth Anything V2 uses a ReLU relative-depth head. Its output is
        # inverse-depth-like (near surfaces receive larger values), not metres.
        # Keep it in that space; ``inverse_least_squares`` fits 1/Z = a*r + b.
        prediction_proxy = sampled_prediction
        result = evaluate_depth(prediction_proxy, lidar_depth, alignment)
        aligned_prediction, _, _ = align_depth(prediction_proxy, lidar_depth, alignment)
        save_projection_overlay(image, pixels, lidar_depth, output_dir / f"{stem}_lidar_projection.png")
        save_metrics(output_dir / f"{stem}_lidar_metrics.json", result, image=str(image_path), lidar_cloud=str(lidar_path), camera_sensor=args.sensor, lidar_sensor=args.lidar_sensor, timestamp_tolerance_s=args.max_lidar_dt, extrinsics_convention=args.extrinsics_convention, prediction_representation=("metric_depth_m" if getattr(model, "is_metric", False) else "reciprocal_relative_depth"))
        save_point_samples(output_dir / f"{stem}_lidar_samples.csv", pixels, lidar_depth, prediction_proxy, aligned_prediction)
        global_predictions.append(prediction_proxy)
        global_ground_truth.append(lidar_depth)
        frame_results.append(result)
        if args.debug_evaluation or args.plot_evaluation:
            raw_finite = depth[np.isfinite(depth)]
            def summary(values, name):
                return {f"{name}_{stat}": float(value) for stat, value in zip(("min", "max", "mean", "median"), (np.min(values), np.max(values), np.mean(values), np.median(values)))}
            row = {
                "frame": image_index, "image": image_path.name, "lidar_cloud": lidar_path.name,
                "camera_timestamp": parse_timestamp(image_path), "lidar_timestamp": parse_timestamp(lidar_path),
                "timestamp_difference_s": parse_timestamp(lidar_path) - parse_timestamp(image_path),
                "raw_lidar_points": len(raw_lidar_points), "front_camera_points": front_count,
                "projected_points": len(nearest), "valid_projected_points": len(prediction_proxy),
                "zero_predictions": int(np.count_nonzero(depth == 0)), "negative_predictions": int(np.count_nonzero(depth < 0)),
                "invalid_predictions": int(np.count_nonzero(~np.isfinite(depth))), "invalid_lidar_points": int(len(raw_lidar_points) - len(load_lidar_points(lidar_path))),
                "camera_x_min": float(np.min(points_camera[:, 0])), "camera_x_max": float(np.max(points_camera[:, 0])),
                "camera_y_min": float(np.min(points_camera[:, 1])), "camera_y_max": float(np.max(points_camera[:, 1])),
                "camera_z_min": float(np.min(points_camera[:, 2])), "camera_z_max": float(np.max(points_camera[:, 2])),
                "alignment_scale": result.scale, "alignment_shift": result.shift, "mae_m": result.mae_m, "rmse_m": result.rmse_m, "abs_rel": result.abs_rel,
                **summary(lidar_depth, "lidar_depth"), **summary(prediction_proxy, "raw_prediction"), **summary(aligned_prediction, "aligned_prediction"),
            }
            debug_rows.append(row)
        print(f"LiDAR metrics ({result.count} points): MAE={result.mae_m:.3f} m, RMSE={result.rmse_m:.3f} m, AbsRel={result.abs_rel:.3f}")

    if batch_lidar and global_predictions:
        global_result = evaluate_depth(np.concatenate(global_predictions), np.concatenate(global_ground_truth), alignment)
        global_path = output_dir / f"{args.recording}_{args.sensor}_{args.lidar_sensor}_lidar_global_metrics.json"
        def frame_stat(attribute: str) -> dict[str, float]:
            values = np.array([getattr(result, attribute) for result in frame_results])
            return {f"mean_frame_{attribute}": float(values.mean()), f"median_frame_{attribute}": float(np.median(values)), f"std_frame_{attribute}": float(values.std())}
        abs_rel = np.array([result.abs_rel for result in frame_results])
        save_metrics(global_path, global_result, camera_sensor=args.sensor, lidar_sensor=args.lidar_sensor, recording=args.recording, evaluated_frames=len(global_predictions), matched_images=len(lidar_pairs), timestamp_tolerance_s=args.max_lidar_dt, extrinsics_convention=args.extrinsics_convention, alignment_scope="single inverse-depth affine fit across all valid LiDAR samples", **frame_stat("mae_m"), **frame_stat("rmse_m"), **frame_stat("abs_rel"), frames_abs_rel_lt_025=float(np.mean(abs_rel < .25)), frames_abs_rel_lt_05=float(np.mean(abs_rel < .5)), frames_abs_rel_ge_09=float(np.mean(abs_rel >= .9)))
        print(f"\nGlobal LiDAR metrics ({global_result.count} points across {len(global_predictions)} frames): MAE={global_result.mae_m:.3f} m, RMSE={global_result.rmse_m:.3f} m, AbsRel={global_result.abs_rel:.3f}")
        print(f"Saved global metrics: {global_path}")

    if args.debug_evaluation and debug_rows:
        debug_path = output_dir / f"{args.recording}_{args.sensor}_{args.lidar_sensor}_evaluation_debug.csv"
        with debug_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=debug_rows[0].keys())
            writer.writeheader()
            writer.writerows(debug_rows)
        print(f"Saved evaluation debug: {debug_path}")

    if args.plot_evaluation and debug_rows and global_predictions:
        import matplotlib.pyplot as plt

        plot_dir = output_dir / "evaluation_plots"
        plot_dir.mkdir(exist_ok=True)
        frame = np.array([row["frame"] for row in debug_rows])
        def line_plot(column: str, title: str, ylabel: str) -> None:
            plt.figure(figsize=(9, 4)); plt.plot(frame, [row[column] for row in debug_rows], marker=".")
            plt.xlabel("frame number"); plt.ylabel(ylabel); plt.title(title); plt.grid(); plt.tight_layout()
            plt.savefig(plot_dir / f"{column}.png", dpi=150); plt.close()
        line_plot("mae_m", "MAE vs frame", "MAE (m)")
        line_plot("rmse_m", "RMSE vs frame", "RMSE (m)")
        line_plot("abs_rel", "AbsRel vs frame", "AbsRel")
        line_plot("valid_projected_points", "Valid LiDAR samples vs frame", "points")
        line_plot("timestamp_difference_s", "LiDAR − camera timestamp vs frame", "seconds")
        line_plot("alignment_scale", "Alignment scale vs frame", "scale")
        line_plot("alignment_shift", "Alignment shift vs frame", "shift")
        raw, gt = np.concatenate(global_predictions), np.concatenate(global_ground_truth)
        aligned, _, _ = align_depth(raw, gt, alignment)
        valid = np.isfinite(aligned) & (aligned > 0)
        upper = np.percentile(np.r_[gt[valid], aligned[valid]], 99)
        plt.figure(figsize=(5, 5)); plt.scatter(gt[valid], aligned[valid], s=1, alpha=.08)
        plt.plot([0, upper], [0, upper], "r--", label="y = x"); plt.xlim(0, upper); plt.ylim(0, upper)
        plt.xlabel("LiDAR depth (m)"); plt.ylabel("aligned prediction (m)"); plt.legend(); plt.tight_layout()
        plt.savefig(plot_dir / "depth_scatter.png", dpi=150); plt.close()
        plt.figure(figsize=(7, 4)); plt.hist(np.abs(aligned[valid] - gt[valid]), bins=100)
        plt.xlabel("absolute error (m)"); plt.ylabel("count"); plt.title("Absolute error histogram"); plt.tight_layout()
        plt.savefig(plot_dir / "absolute_error_histogram.png", dpi=150); plt.close()
        print(f"Saved evaluation plots: {plot_dir}")


if __name__ == "__main__":
    main()
