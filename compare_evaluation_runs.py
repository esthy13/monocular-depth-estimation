#!/usr/bin/env python3
"""Create a compact, reproducible comparison of model-evaluation summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COMPARISON_FIELDS = [
    "label",
    "prediction_models",
    "recording",
    "processed_frames",
    "person_detections",
    "mean_depth_inference_ms",
    "mean_zed_mae_m",
    "mean_zed_rmse_m",
    "mean_zed_abs_rel",
    "mean_zed_3d_error_m",
    "mean_lidar_3d_error_m",
]


def parse_run_spec(specification: str) -> tuple[str, Path]:
    """Parse a ``LABEL=/path/to/evaluation_summary.json`` argument."""
    label, separator, path = specification.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError(
            "Each --run must use LABEL=/path/to/evaluation_summary.json"
        )
    return label.strip(), Path(path).expanduser()


def load_comparison_row(label: str, summary_path: Path) -> dict:
    """Load the comparable fields from one evaluator summary."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    models = summary.get("prediction_models") or []
    row = {
        "label": label,
        "prediction_models": ", ".join(str(model) for model in models),
        "summary_path": str(summary_path),
    }
    for field in COMPARISON_FIELDS[2:]:
        row[field] = summary.get(field)
    row["configuration"] = summary.get("configuration") or {}
    return row


def comparability_warnings(rows: list[dict]) -> list[str]:
    """Return reasons the supplied experiment summaries may not be comparable."""
    if len(rows) < 2:
        return ["At least two runs are required for a model comparison."]

    warnings: list[str] = []
    if len({row.get("recording") for row in rows}) > 1:
        warnings.append("Runs use different recordings.")
    if len({row.get("processed_frames") for row in rows}) > 1:
        warnings.append("Runs contain different numbers of processed frames.")
    if len({row.get("person_detections") for row in rows}) > 1:
        warnings.append("Runs contain different numbers of person detections.")

    protocol_fields = (
        "start_index",
        "max_frames",
        "frame_step",
        "detector_confidence",
        "detector_iou",
        "detector_image_size",
        "zed_max_dt_seconds",
        "lidar_max_dt_seconds",
        "lidar_consistency_metres",
        "minimum_lidar_points",
    )
    for field in protocol_fields:
        values = {row["configuration"].get(field) for row in rows}
        if len(values) > 1:
            warnings.append(f"Runs use different {field} settings.")
    return warnings


def markdown_table(rows: list[dict]) -> str:
    """Format comparison rows for a report or pull request."""
    headers = {
        "label": "Run",
        "prediction_models": "Model",
        "processed_frames": "Frames",
        "person_detections": "Detections",
        "mean_depth_inference_ms": "Depth ms",
        "mean_zed_mae_m": "ZED MAE m",
        "mean_zed_rmse_m": "ZED RMSE m",
        "mean_zed_abs_rel": "ZED AbsRel",
        "mean_zed_3d_error_m": "ZED 3D m",
        "mean_lidar_3d_error_m": "LiDAR 3D m",
    }
    fields = list(headers)

    def display(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers[field] for field in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(display(row.get(field)) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare evaluator JSON summaries for two or more depth models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=SUMMARY_JSON",
        help="Named evaluation run; repeat for every model.",
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        load_comparison_row(*parse_run_spec(specification))
        for specification in args.run
    ]
    warnings = comparability_warnings(rows)
    table = markdown_table(rows)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in COMPARISON_FIELDS}
                for row in rows
            )
        print(f"CSV comparison:      {args.output_csv}")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(table)
        print(f"Markdown comparison: {args.output_markdown}")
    else:
        print(table, end="")

    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
