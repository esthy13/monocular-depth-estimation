#!/usr/bin/env python3
"""Benchmark DAC and UniDAC with one controlled, repeated GPU protocol."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev

import cv2
import numpy as np

from src.utils import find_rgb_images, intrinsics_to_dac_cam_params, load_intrinsics


MODEL_NAMES = ("DAC", "UniDAC")
TIMING_SCOPE = "model.predict: preprocessing + neural inference + camera back-projection"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(requested: str, torch) -> str:
    """Resolve ``auto`` to CUDA, Apple Metal, or CPU in that order."""
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


def clear_device_cache(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def accelerator_name(torch, device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"{platform.machine()} Apple Metal (MPS)"
    return platform.processor() or "CPU"


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile without an extra dependency."""
    if not values:
        raise ValueError("At least one timing value is required")
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_model_timings(model: str, rows: list[dict]) -> dict:
    """Aggregate long-form timing rows for one model."""
    values = [
        float(row["time_ms"])
        for row in rows
        if row.get("model") == model
    ]
    if not values:
        raise ValueError(f"No timing rows found for {model}")
    median_ms = percentile(values, 50.0)
    return {
        "model": model,
        "measurements": len(values),
        "mean_ms": fmean(values),
        "median_ms": median_ms,
        "stddev_ms": pstdev(values),
        "p10_ms": percentile(values, 10.0),
        "p90_ms": percentile(values, 90.0),
        "min_ms": min(values),
        "max_ms": max(values),
        "throughput_fps_from_median": 1000.0 / median_ms,
    }


def build_comparison(model_summaries: list[dict]) -> dict:
    """Describe the median-latency winner for exactly DAC and UniDAC."""
    by_model = {summary["model"]: summary for summary in model_summaries}
    missing = set(MODEL_NAMES) - set(by_model)
    if missing:
        raise ValueError(f"Missing model summaries: {sorted(missing)}")
    dac_ms = float(by_model["DAC"]["median_ms"])
    unidac_ms = float(by_model["UniDAC"]["median_ms"])
    faster_model = "DAC" if dac_ms < unidac_ms else "UniDAC"
    slower_model = "UniDAC" if faster_model == "DAC" else "DAC"
    faster_ms = min(dac_ms, unidac_ms)
    slower_ms = max(dac_ms, unidac_ms)
    return {
        "faster_model": faster_model,
        "slower_model": slower_model,
        "median_speedup_ratio": slower_ms / faster_ms,
        "median_latency_reduction_percent": 100.0 * (slower_ms - faster_ms) / slower_ms,
    }


def markdown_report(summary: dict) -> str:
    """Create a compact report from the benchmark summary."""
    protocol = summary["protocol"]
    environment = summary["environment"]
    lines = [
        "# Controlled DAC versus UniDAC speed benchmark",
        "",
        "Both models were measured sequentially in the same Python process and on",
        "the same GPU. Image loading, model loading, and result-file writing are",
        "outside the timed region.",
        "",
        "## Protocol",
        "",
        f"- Recording: `{protocol['recording']}`",
        f"- Sensor: `{protocol['sensor']}`",
        f"- Frame indices: `{protocol['frame_indices']}`",
        f"- Warm-up runs per model: {protocol['warmup_runs']}",
        f"- Timed runs per frame and model: {protocol['timed_runs_per_frame']}",
        f"- Timing scope: {protocol['timing_scope']}",
        f"- Device: {environment.get('device', 'GPU')}",
        f"- Accelerator: {environment['gpu']}",
        f"- PyTorch: {environment['torch_version']}",
        "",
        "## Results",
        "",
        "| Model | Measurements | Mean ms | Median ms | P10 ms | P90 ms | FPS from median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if environment.get("cuda_version"):
        lines.insert(16, f"- CUDA runtime: {environment['cuda_version']}")
    for model_summary in summary["models"]:
        lines.append(
            "| {model} | {measurements} | {mean_ms:.2f} | {median_ms:.2f} | "
            "{p10_ms:.2f} | {p90_ms:.2f} | {throughput_fps_from_median:.3f} |".format(
                **model_summary
            )
        )
    comparison = summary["comparison"]
    lines.extend([
        "",
        "## Result",
        "",
        f"{comparison['faster_model']} is "
        f"{comparison['median_speedup_ratio']:.3f}× faster by median latency "
        f"({comparison['median_latency_reduction_percent']:.1f}% lower latency) "
        f"than {comparison['slower_model']} under this protocol.",
        "",
        "This benchmark measures depth-pipeline latency only. Person detection,",
        "tracking, disk I/O, and model initialization are intentionally excluded.",
        "",
    ])
    return "\n".join(lines)


def validate_frame_indices(indices: list[int], image_count: int) -> list[int]:
    """Return unique frame indices in input order after range validation."""
    if not indices:
        raise ValueError("At least one frame index is required")
    unique = list(dict.fromkeys(indices))
    invalid = [index for index in unique if not 0 <= index < image_count]
    if invalid:
        raise ValueError(
            f"Frame indices {invalid} are outside 0..{image_count - 1}"
        )
    return unique


def project_commit(project_dir: Path) -> str | None:
    """Return the checked-out project commit when available."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pinned_commit(repository_dir: Path) -> str | None:
    marker = repository_dir / ".pinned_commit"
    return marker.read_text().strip() if marker.is_file() else None


def load_benchmark_images(
    data_dir: Path,
    recording: str,
    sensor: str,
    frame_indices: list[int],
) -> list[tuple[int, Path, np.ndarray]]:
    paths = find_rgb_images(data_dir, sensor_name=sensor, recording=recording)
    selected = validate_frame_indices(frame_indices, len(paths))
    loaded = []
    for index in selected:
        image = cv2.imread(str(paths[index]))
        if image is None:
            raise RuntimeError(f"OpenCV could not read {paths[index]}")
        loaded.append((index, paths[index], image))
    return loaded


def camera_geometry(data_dir: Path, sensor: str, image_width: int) -> tuple[dict, float]:
    intrinsics = load_intrinsics(data_dir / "intrinsic.json")
    camera = intrinsics[sensor]
    parameters = intrinsics_to_dac_cam_params(sensor, intrinsics)
    if camera.get("model") == "fisheye":
        crop_wfov = 180.0
    else:
        focal_x = float(camera["K"][0][0])
        crop_wfov = math.degrees(2.0 * math.atan(image_width / (2.0 * focal_x)))
    return parameters, crop_wfov


def time_model(
    model_name: str,
    model,
    images: list[tuple[int, Path, np.ndarray]],
    camera_parameters: dict,
    crop_wfov: float,
    warmup_runs: int,
    timed_runs: int,
    torch,
    device: str,
) -> list[dict]:
    """Warm up and time one loaded model on already-decoded images."""
    model.set_camera(camera_parameters, crop_wfov)
    warmup_image = images[0][2]
    for _ in range(warmup_runs):
        model.predict(warmup_image)
    synchronize(torch, device)

    rows = []
    for image_index, image_path, image in images:
        for repeat in range(timed_runs):
            synchronize(torch, device)
            start = time.perf_counter()
            depth = model.predict(image)
            synchronize(torch, device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not np.any(np.isfinite(depth) & (depth > 0)):
                raise RuntimeError(
                    f"{model_name} returned no valid positive depth for {image_path}"
                )
            rows.append({
                "model": model_name,
                "image_index": image_index,
                "image_file": image_path.name,
                "repeat": repeat + 1,
                "time_ms": elapsed_ms,
            })
        print(
            f"{model_name}: frame {image_index} complete "
            f"({timed_runs} timed runs)"
        )
    return rows


def build_model(model_name: str, args: argparse.Namespace):
    """Load one requested metric-depth model."""
    if model_name == "DAC":
        from src.depth_models import DepthAnyCamera

        model = DepthAnyCamera(
            variant=args.dac_variant,
            fwd_sz=tuple(args.dac_forward_size),
            device=args.device,
            config_path=str(args.dac_config_path),
            checkpoint_path=str(args.dac_checkpoint_path),
        )
    elif model_name == "UniDAC":
        from src.depth_models import UniDACDepth

        model = UniDACDepth(
            device=args.device,
            repo_dir=str(args.unidac_repo_dir),
            checkpoint_path=str(args.unidac_checkpoint_path),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.load()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DAC and UniDAC sequentially on one CUDA GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--recording", default="recording1")
    parser.add_argument("--sensor", default="G1_A")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--frame_indices", type=int, nargs="+", default=[0, 15, 30, 45, 60, 75, 90, 105, 120, 132]
    )
    parser.add_argument("--warmup_runs", type=int, default=5)
    parser.add_argument("--timed_runs", type=int, default=10)
    parser.add_argument("--model_order", nargs=2, choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--dac_repo_dir", type=Path, required=True)
    parser.add_argument("--dac_config_path", type=Path, required=True)
    parser.add_argument("--dac_checkpoint_path", type=Path, required=True)
    parser.add_argument("--dac_variant", default="dac-indoor-resnet101")
    parser.add_argument("--dac_forward_size", type=int, nargs=2, default=[500, 750])
    parser.add_argument("--unidac_repo_dir", type=Path, required=True)
    parser.add_argument("--unidac_checkpoint_path", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_runs < 1:
        parser.error("--warmup_runs must be at least 1")
    if args.timed_runs < 2:
        parser.error("--timed_runs must be at least 2")
    if len(set(args.model_order)) != 2:
        parser.error("--model_order must contain DAC and UniDAC exactly once")
    return args


def main() -> None:
    args = parse_args()
    import torch

    args.device = resolve_device(args.device, torch)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal was requested but MPS is unavailable")

    images = load_benchmark_images(
        args.data_dir, args.recording, args.sensor, args.frame_indices
    )
    parameters, crop_wfov = camera_geometry(
        args.data_dir, args.sensor, images[0][2].shape[1]
    )

    all_rows = []
    for model_name in args.model_order:
        model = build_model(model_name, args)
        try:
            all_rows.extend(time_model(
                model_name,
                model,
                images,
                parameters,
                crop_wfov,
                args.warmup_runs,
                args.timed_runs,
                torch,
                args.device,
            ))
        finally:
            del model
            gc.collect()
            clear_device_cache(torch, args.device)

    model_summaries = [
        summarize_model_timings(model_name, all_rows)
        for model_name in MODEL_NAMES
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "recording": args.recording,
            "sensor": args.sensor,
            "frame_indices": [index for index, _, _ in images],
            "warmup_runs": args.warmup_runs,
            "timed_runs_per_frame": args.timed_runs,
            "model_order": args.model_order,
            "timing_scope": TIMING_SCOPE,
            "image_loading_timed": False,
            "model_loading_timed": False,
        },
        "environment": {
            "device": args.device,
            "gpu": accelerator_name(torch, args.device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if args.device == "cuda" else None,
            "project_commit": project_commit(Path(__file__).resolve().parent),
        },
        "model_provenance": {
            "DAC": {
                "variant": args.dac_variant,
                "upstream_commit": pinned_commit(args.dac_repo_dir),
            },
            "UniDAC": {
                "upstream_commit": pinned_commit(args.unidac_repo_dir),
            },
        },
        "models": model_summaries,
        "comparison": build_comparison(model_summaries),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings_path = args.output_dir / "benchmark_timings.csv"
    with timings_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "image_index", "image_file", "repeat", "time_ms"],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = args.output_dir / "benchmark_report.md"
    report_path.write_text(markdown_report(summary))

    print(markdown_report(summary))
    print(f"Timing rows: {timings_path}")
    print(f"Summary:     {summary_path}")
    print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()
