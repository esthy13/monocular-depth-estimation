#!/usr/bin/env python3
"""Compare metric-depth models across a matched suite of recordings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from compare_evaluation_runs import comparability_warnings, load_comparison_row


SUITE_FIELDS = [
    "scope",
    "label",
    "prediction_models",
    "processed_frames",
    "person_detections",
    "detections_with_zed_reference",
    "detections_with_lidar_reference",
    "mean_depth_inference_ms",
    "mean_zed_mae_m",
    "mean_zed_rmse_m",
    "mean_zed_abs_rel",
    "mean_zed_3d_error_m",
    "mean_lidar_3d_error_m",
]

WEIGHTS = {
    "mean_depth_inference_ms": "processed_frames",
    "mean_zed_mae_m": "detections_with_zed_reference",
    "mean_zed_rmse_m": "detections_with_zed_reference",
    "mean_zed_abs_rel": "detections_with_zed_reference",
    "mean_zed_3d_error_m": "detections_with_zed_reference",
    "mean_lidar_3d_error_m": "detections_with_lidar_reference",
}


def parse_model_root(specification: str) -> tuple[str, Path]:
    """Parse ``LABEL=/path/to/model/evaluation``."""
    label, separator, path = specification.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("Each --model_root must use LABEL=/path/to/evaluation")
    return label.strip(), Path(path).expanduser()


def load_suite_rows(
    model_roots: list[tuple[str, Path]], recordings: list[str]
) -> list[dict]:
    """Load one evaluator summary for every model/recording pair."""
    rows = []
    for recording in recordings:
        for label, root in model_roots:
            row = load_comparison_row(
                label, root / recording / "evaluation_summary.json"
            )
            if row.get("recording") != recording:
                raise ValueError(
                    f"Expected {recording!r} in {row['summary_path']}, got "
                    f"{row.get('recording')!r}"
                )
            row["scope"] = recording
            # Reference counts are not part of the compact single-run table but
            # are required to combine per-detection means without giving short
            # recordings the same weight as long recordings.
            source = json.loads(Path(row["summary_path"]).read_text())
            row["detections_with_zed_reference"] = source.get(
                "detections_with_zed_reference"
            )
            row["detections_with_lidar_reference"] = source.get(
                "detections_with_lidar_reference"
            )
            rows.append(row)
    return rows


def suite_warnings(rows: list[dict], recordings: list[str]) -> list[str]:
    warnings = []
    for recording in recordings:
        recording_rows = [row for row in rows if row["scope"] == recording]
        warnings.extend(
            f"{recording}: {warning}"
            for warning in comparability_warnings(recording_rows)
        )
        for field, label in (
            ("detections_with_zed_reference", "valid ZED references"),
            ("detections_with_lidar_reference", "valid LiDAR references"),
        ):
            if len({row.get(field) for row in recording_rows}) > 1:
                warnings.append(
                    f"{recording}: Runs contain different numbers of {label}."
                )
    return warnings


def weighted_mean(rows: list[dict], value_field: str, weight_field: str) -> float | None:
    pairs = [
        (float(row[value_field]), int(row[weight_field]))
        for row in rows
        if row.get(value_field) is not None
        and row.get(weight_field) is not None
        and int(row[weight_field]) > 0
    ]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def aggregate_rows(rows: list[dict], labels: list[str]) -> list[dict]:
    """Combine recordings using the denominator of each reported mean."""
    aggregates = []
    for label in labels:
        model_rows = [row for row in rows if row["label"] == label]
        if not model_rows:
            raise ValueError(f"No suite rows found for {label}")
        aggregate = {
            "scope": "all recordings",
            "label": label,
            "prediction_models": model_rows[0].get("prediction_models"),
            "processed_frames": sum(int(row["processed_frames"]) for row in model_rows),
            "person_detections": sum(int(row["person_detections"]) for row in model_rows),
            "detections_with_zed_reference": sum(
                int(row["detections_with_zed_reference"]) for row in model_rows
            ),
            "detections_with_lidar_reference": sum(
                int(row["detections_with_lidar_reference"]) for row in model_rows
            ),
        }
        for value_field, weight_field in WEIGHTS.items():
            aggregate[value_field] = weighted_mean(
                model_rows, value_field, weight_field
            )
        aggregates.append(aggregate)
    return aggregates


def markdown_table(rows: list[dict]) -> str:
    headers = [
        ("scope", "Recording"),
        ("label", "Model"),
        ("processed_frames", "Frames"),
        ("person_detections", "Detections"),
        ("detections_with_zed_reference", "ZED refs"),
        ("detections_with_lidar_reference", "LiDAR refs"),
        ("mean_depth_inference_ms", "Depth ms"),
        ("mean_zed_mae_m", "ZED MAE m"),
        ("mean_zed_rmse_m", "ZED RMSE m"),
        ("mean_zed_abs_rel", "ZED AbsRel"),
        ("mean_zed_3d_error_m", "ZED 3D m"),
        ("mean_lidar_3d_error_m", "LiDAR 3D m"),
    ]

    def display(value: object) -> str:
        if value is None:
            return "—"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(title for _, title in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(display(row.get(field)) for field, _ in headers) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def markdown_report(
    rows: list[dict], aggregates: list[dict], warnings: list[str]
) -> str:
    lines = [
        "# DAC versus UniDAC recording-suite comparison",
        "",
        "## Per-recording results",
        "",
        markdown_table(rows).rstrip(),
        "",
        "## Weighted result across all recordings",
        "",
        markdown_table(aggregates).rstrip(),
        "",
        "Accuracy means are weighted by the number of valid physical-reference",
        "detections in each recording. Depth time is weighted by processed frames.",
    ]
    if warnings:
        lines.extend(["", "## Protocol warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.extend(["", "All model pairs use matching per-recording protocols."])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models across multiple matched evaluator summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_root",
        action="append",
        required=True,
        metavar="LABEL=EVALUATION_DIR",
    )
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_roots = [parse_model_root(specification) for specification in args.model_root]
    labels = [label for label, _ in model_roots]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("Supply at least two uniquely labelled --model_root values")
    recordings = list(dict.fromkeys(args.recordings))
    rows = load_suite_rows(model_roots, recordings)
    warnings = suite_warnings(rows, recordings)
    aggregates = aggregate_rows(rows, labels)
    report = markdown_report(rows, aggregates, warnings)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUITE_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in SUITE_FIELDS}
                for row in [*rows, *aggregates]
            )
        print(f"Suite CSV:      {args.output_csv}")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(report)
        print(f"Suite Markdown: {args.output_markdown}")
    else:
        print(report, end="")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
