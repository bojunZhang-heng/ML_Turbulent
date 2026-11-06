#!/usr/bin/env python3
"""Compare DrivAerStar models with CANO/LinearNO/Transolver-style evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_split import load_config
from inference import evaluate_run
from metrics.coefficient_metrics import read_coefficient_csv
from metrics.statistics import paired_bootstrap_difference
from plots.ablation_plots import plot_ablation_csv
from plots.attention_maps import plot_attention_npz
from plots.coefficient_plots import plot_coefficients
from plots.efficiency_bubble import plot_efficiency_bubble
from plots.field_maps import plot_field_comparison
from plots.raincloud import plot_error_raincloud
from plots.scaling_curves import plot_scaling_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run", nargs=3, action="append", metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        help="Model label, YAML and checkpoint. Repeat this option to compare models.",
    )
    parser.add_argument("--config", type=Path, help="Backward-compatible single-run YAML")
    parser.add_argument("--checkpoint", type=Path, help="Backward-compatible single-run checkpoint")
    parser.add_argument("--label", default=None, help="Label used with --config/--checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("postprocess_results"))
    parser.add_argument("--data-path", type=Path, default=None, help="Override YAML data path for every run")
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-case-id", default=None, help="Case ID for GT/prediction/error field maps")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--print-per-case",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print every case metric so terminal output can be redirected to a log.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--coefficient-csv", type=Path, default=None)
    parser.add_argument("--attention-npz", type=Path, default=None)
    parser.add_argument("--attention-index", type=int, default=0)
    parser.add_argument("--scaling-csv", type=Path, default=None)
    parser.add_argument("--ablation-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_test_samples < 0 or args.bootstrap_samples < 0:
        parser.error("sample counts must be non-negative")
    if bool(args.config) != bool(args.checkpoint):
        parser.error("--config and --checkpoint must be supplied together")
    runs = list(args.run or [])
    if args.config:
        runs.append((args.label or args.config.stem.replace("config_", ""), str(args.config), str(args.checkpoint)))
    if not runs:
        parser.error("provide at least one --run, or --config together with --checkpoint")
    labels = [run[0] for run in runs]
    if len(labels) != len(set(labels)):
        parser.error("every run LABEL must be unique")
    args.runs = [(label, Path(config), Path(checkpoint)) for label, config, checkpoint in runs]
    return args


def _print_json(title, value):
    print(f"\n{'=' * 24} {title} {'=' * 24}", flush=True)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str), flush=True)


def _print_launch_configuration(args):
    configuration = {
        "runs": [
            {
                "label": label,
                "config": str(config.resolve()),
                "checkpoint": str(checkpoint.resolve()),
            }
            for label, config, checkpoint in args.runs
        ],
        "data_path_override": str(args.data_path.resolve()) if args.data_path else None,
        "output_dir": str(args.output_dir.resolve()),
        "max_test_samples": args.max_test_samples or "all",
        "device": args.device,
        "amp_requested": args.amp,
        "bootstrap_samples": args.bootstrap_samples,
        "print_per_case": args.print_per_case,
        "save_predictions": args.save_predictions,
        "plot_case_id": args.plot_case_id,
        "coefficient_csv": str(args.coefficient_csv) if args.coefficient_csv else None,
        "attention_npz": str(args.attention_npz) if args.attention_npz else None,
        "scaling_csv": str(args.scaling_csv) if args.scaling_csv else None,
        "ablation_csv": str(args.ablation_csv) if args.ablation_csv else None,
        "seed": args.seed,
    }
    _print_json("POSTPROCESS CONFIGURATION", configuration)


def _print_output_inventory(output_dir):
    files = sorted(path for path in Path(output_dir).rglob("*") if path.is_file())
    print(f"\n{'=' * 24} OUTPUT FILES {'=' * 24}", flush=True)
    if not files:
        print("No output files were generated.", flush=True)
        return
    for path in files:
        size_mb = path.stat().st_size / 1024**2
        print(f"{path.resolve()}  ({size_mb:.6f} MiB)", flush=True)


def _write_case_csv(path, all_rows):
    fieldnames = ["model", "case_id", "points", "relative_l2", "r2", "mae", "rmse", "bias", "pearson_r", "max_absolute_error"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in all_rows.items():
            for row in rows:
                writer.writerow({"model": label, **row})


def _validate_comparison(summaries):
    first_label = next(iter(summaries))
    first = summaries[first_label]
    for label, summary in summaries.items():
        if summary["task"] != first["task"]:
            raise ValueError(f"Cannot compare tasks {first['task']!r} and {summary['task']!r} ({label})")
        if summary["test_case_ids"] != first["test_case_ids"]:
            raise ValueError(
                f"Test cases for {label!r} differ from {first_label!r}. "
                "Use identical data path, split ratios, selection mode and seed."
            )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _print_launch_configuration(args)
    figure_dir = args.output_dir / "figures"
    summaries, all_rows, cases = {}, {}, {}

    for label, config_path, checkpoint in args.runs:
        raw_cfg, cfg = load_config(config_path)
        _print_json(f"FLATTENED YAML: {label}", cfg)
        summary, rows, selected_case = evaluate_run(
            label, config_path, checkpoint, raw_cfg, cfg, args
        )
        summaries[label] = summary
        all_rows[label] = rows
        cases[label] = selected_case
        printable_summary = dict(summary)
        printable_summary["test_case_ids"] = {
            "count": len(summary["test_case_ids"]),
            "values": summary["test_case_ids"],
        }
        _print_json(f"MODEL RESULT: {label}", printable_summary)

    _validate_comparison(summaries)
    comparison = {
        "models": summaries,
        "paired_relative_l2": {},
        "notes": {
            "physical_units": True,
            "global_aggregation": "node weighted",
            "case_aggregation": "each vehicle has equal weight",
        },
    }
    labels = list(summaries)
    for first_index, first in enumerate(labels):
        for second in labels[first_index + 1:]:
            first_values = [row["relative_l2"] for row in all_rows[first]]
            second_values = [row["relative_l2"] for row in all_rows[second]]
            comparison["paired_relative_l2"][f"{first}_minus_{second}"] = paired_bootstrap_difference(
                first_values, second_values, samples=max(args.bootstrap_samples, 1), seed=args.seed
            )

    coefficient_data = None
    if args.coefficient_csv:
        coefficient_data, coefficient_metrics = read_coefficient_csv(args.coefficient_csv, labels)
        comparison["aerodynamic_coefficients"] = coefficient_metrics
        _print_json("AERODYNAMIC COEFFICIENT METRICS", coefficient_metrics)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2)
    _write_case_csv(args.output_dir / "per_case_metrics.csv", all_rows)

    plot_error_raincloud(
        {label: [row["relative_l2"] for row in rows] for label, rows in all_rows.items()},
        figure_dir / "error_raincloud.png", seed=args.seed,
    )
    if all(cases[label] is not None for label in labels):
        reference = cases[labels[0]]
        plot_field_comparison(
            reference["case_id"], reference["coords"], reference["truth"],
            {label: cases[label]["prediction"] for label in labels},
            figure_dir / f"field_comparison_{reference['case_id'].replace('/', '_')}.png",
        )
    elif args.plot_case_id:
        raise ValueError(f"Requested plot case {args.plot_case_id!r} was not found in every run")
    plot_efficiency_bubble(summaries, figure_dir / "efficiency_bubble.png")
    if coefficient_data is not None:
        plot_coefficients(coefficient_data, comparison["aerodynamic_coefficients"], figure_dir)
    if args.attention_npz:
        plot_attention_npz(args.attention_npz, figure_dir / "attention", args.attention_index)
    if args.scaling_csv:
        plot_scaling_csv(args.scaling_csv, figure_dir)
    if args.ablation_csv:
        plot_ablation_csv(args.ablation_csv, figure_dir / "ablation.png")

    _print_json("PAIRED MODEL COMPARISON", comparison["paired_relative_l2"])

    print("\nModel comparison (physical-unit, equal-case relative L2)")
    for label, summary in summaries.items():
        metric = summary["physical_metrics"]["case_equal_weighted"]["relative_l2"]
        print(f"  {label:20s} mean={metric['mean']:.6g}  median={metric['median']:.6g}  p95={metric['p95']:.6g}")
    print(f"\nSummary: {summary_path.resolve()}")
    print(f"Per-case CSV: {(args.output_dir / 'per_case_metrics.csv').resolve()}")
    print(f"Figures: {figure_dir.resolve()}")
    _print_output_inventory(args.output_dir)
    print("Postprocess completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
