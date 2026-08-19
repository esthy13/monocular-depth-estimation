#!/usr/bin/env python3
"""Generate resumable metric-depth batches with DAC or UniDAC."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.utils import (
    create_fisheye_valid_mask,
    find_rgb_images,
    intrinsics_to_dac_cam_params,
    load_intrinsics,
    parse_timestamp,
)


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def accelerator_name(torch, device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"{platform.machine()} Apple Metal (MPS)"
    return platform.processor() or "CPU"


def git_commit(directory: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def camera_geometry(intrinsics: dict, sensor: str, image_width: int) -> tuple[dict, float]:
    camera = intrinsics[sensor]
    parameters = intrinsics_to_dac_cam_params(sensor, intrinsics)
    if camera.get("model") == "fisheye":
        crop_wfov = 180.0
    else:
        focal_x = float(camera["K"][0][0])
        crop_wfov = math.degrees(2.0 * math.atan(image_width / (2.0 * focal_x)))
    return parameters, crop_wfov


def load_model(args: argparse.Namespace):
    if args.model == "DAC":
        from src.depth_models import DepthAnyCamera

        model = DepthAnyCamera(
            variant=args.dac_variant,
            fwd_sz=tuple(args.dac_forward_size),
            device=args.device,
        )
    else:
        from src.depth_models import UniDACDepth

        model = UniDACDepth(
            device=args.device,
            repo_dir=str(args.unidac_repo_dir),
        )
    model.load()
    return model


def model_provenance(args: argparse.Namespace, project_dir: Path) -> dict:
    if args.model == "DAC":
        return {
            "model": "DAC",
            "variant": args.dac_variant,
            "checkpoint_repo": "yuliangguo/depth-any-camera",
            "dac_commit": git_commit(project_dir / "third_party" / "depth_any_camera"),
        }
    return {
        "model": "UniDAC",
        "checkpoint_repo": "girish1511/UniDAC",
        "unidac_commit": git_commit(args.unidac_repo_dir),
    }


def write_summary(output_dir: Path, model: str) -> Path:
    fields = [
        "recording", "sensor", "image_index", "image_file",
        "timestamp_seconds", "device", "gpu", "median_time_ms",
        "valid_fraction", "valid_depth_min_m", "valid_depth_median_m",
        "valid_depth_max_m", "project_commit", "upstream_commit", "variant",
    ]
    rows = []
    for metadata_path in sorted(output_dir.rglob("*_metadata.json")):
        record = json.loads(metadata_path.read_text())
        upstream = record.get("dac_commit") or record.get("unidac_commit")
        rows.append({
            **{field: record.get(field) for field in fields},
            "upstream_commit": upstream,
        })
    summary_path = output_dir / f"{model.lower()}_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable DAC or UniDAC metric-depth batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", choices=("DAC", "UniDAC"), required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--sensor", default="G1_A")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dac_variant", default="dac-indoor-resnet101")
    parser.add_argument("--dac_forward_size", type=int, nargs=2, default=[500, 750])
    parser.add_argument(
        "--unidac_repo_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "third_party" / "UniDAC",
    )
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame_step must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup_runs cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    import torch

    args.device = resolve_device(args.device, torch)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal was requested but MPS is unavailable")

    project_dir = Path(__file__).resolve().parent
    project_revision = git_commit(project_dir)
    intrinsics = load_intrinsics(args.data_dir / "intrinsic.json")
    model = load_model(args)
    provenance = model_provenance(args, project_dir)
    accelerator = accelerator_name(torch, args.device)
    print(f"Device: {accelerator}")

    warmup_complete = False
    completed = 0
    skipped = 0
    started_batch = time.perf_counter()
    for recording in args.recordings:
        images = find_rgb_images(
            args.data_dir, sensor_name=args.sensor, recording=recording
        )
        indices = list(range(args.start_index, len(images), args.frame_step))
        if args.max_frames > 0:
            indices = indices[:args.max_frames]
        for image_index in indices:
            frame_dir = args.output_dir / recording / args.sensor
            frame_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{recording}_{args.sensor}_{image_index:06d}"
            raw_path = frame_dir / f"{stem}_depth_raw.npy"
            metadata_path = frame_dir / f"{stem}_metadata.json"
            if (
                not args.overwrite
                and raw_path.is_file()
                and metadata_path.is_file()
            ):
                skipped += 1
                continue

            image_path = images[image_index]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"OpenCV could not read {image_path}")
            parameters, crop_wfov = camera_geometry(
                intrinsics, args.sensor, image.shape[1]
            )
            model.set_camera(parameters, crop_wfov)
            if not warmup_complete:
                for _ in range(args.warmup_runs):
                    model.predict(image)
                synchronize(torch, args.device)
                warmup_complete = True

            synchronize(torch, args.device)
            started = time.perf_counter()
            depth = model.predict(image)
            synchronize(torch, args.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            valid_mask = np.isfinite(depth) & (depth > 0)
            if intrinsics[args.sensor].get("model") == "fisheye":
                intrinsic_matrix = intrinsics[args.sensor]["K"]
                lens_mask = create_fisheye_valid_mask(
                    image,
                    center=(intrinsic_matrix[0][2], intrinsic_matrix[1][2]),
                )
                valid_mask &= lens_mask.astype(bool)
            if not np.any(valid_mask):
                raise RuntimeError(f"{args.model} returned no valid depth for {image_path}")
            depth = depth.astype(np.float32, copy=True)
            depth[~valid_mask] = np.nan
            np.save(raw_path, depth)

            valid_depth = depth[valid_mask]
            metadata = {
                **provenance,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "project_commit": project_revision,
                "depth_definition": "metric Euclidean ray distance",
                "depth_unit": "metre",
                "recording": recording,
                "sensor": args.sensor,
                "image_index": image_index,
                "image_file": image_path.name,
                "timestamp_seconds": parse_timestamp(image_path),
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "camera_model": intrinsics[args.sensor].get("model"),
                "requested_crop_wfov_degrees": crop_wfov,
                "projection": model.last_projection_metadata,
                "device": args.device,
                "gpu": accelerator,
                "torch_version": torch.__version__,
                "warmup_runs_before_batch": args.warmup_runs,
                "timed_runs": 1,
                "timing_scope": "preprocess + model + camera back-projection",
                "timings_ms": [elapsed_ms],
                "median_time_ms": elapsed_ms,
                "valid_fraction": float(valid_mask.mean()),
                "valid_depth_min_m": float(valid_depth.min()),
                "valid_depth_median_m": float(np.median(valid_depth)),
                "valid_depth_max_m": float(valid_depth.max()),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            completed += 1
            elapsed_batch = time.perf_counter() - started_batch
            rate = elapsed_batch / completed
            print(
                f"[{completed}] {recording}/{args.sensor}/{image_index}: "
                f"{elapsed_ms:.1f} ms; average {rate:.1f} s/frame",
                flush=True,
            )

    summary_path = write_summary(args.output_dir, args.model)
    print(
        f"Batch complete: {completed} processed, {skipped} skipped. "
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
