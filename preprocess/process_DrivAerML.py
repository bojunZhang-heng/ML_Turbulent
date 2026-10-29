#!/usr/bin/env python3
"""Convert DrivAerML run directories into normalized per-sample PKL files.

All non-empty ``run_<id>`` directories are converted.  Complete and partial
runs are both preserved; each PKL records whether all expected arrays were
available.  The source run directories are read-only by default and may be
removed only with the explicit ``--remove-source-runs`` option after every
output has been verified.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_DIR = Path(
    "/Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000"
)
SCALARS_FILENAME = "normalization_scalars.pkl"


@dataclass(frozen=True)
class FieldSpec:
    source_name: str
    output_name: str
    kind: str  # "coordinate" or "physical"
    components: int
    required: bool = True


FIELD_SPECS = (
    FieldSpec("surface_position_vtp.npy", "x_surface", "coordinate", 3),
    FieldSpec("surface_pressure.npy", "y_spressure", "physical", 1),
    FieldSpec("volume_cell_position.npy", "x_volume", "coordinate", 3),
    FieldSpec("volume_cell_velocity.npy", "y_velocity", "physical", 3),
    FieldSpec("surface_wallshearstress.npy", "y_wss", "physical", 3),
    # This sixth file exists in 465 runs. It was not assigned a new name in
    # the request, so its original stem is retained rather than silently lost.
    FieldSpec(
        "volume_cell_totalpcoeff.npy",
        "volume_cell_totalpcoeff",
        "physical",
        1,
        required=False,
    ),
)


@dataclass
class RunningMoments:
    """Numerically stable population moments accumulated in float64."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected non-empty (N,C) values, got {values.shape}")
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        if self.mean is None or self.m2 is None or self.mean.shape != batch_mean.shape:
            raise ValueError("Physical field component count changed between runs")
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + np.square(delta) * self.count * batch_count / total
        )
        self.count = total

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("No values were accumulated")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std[std <= np.finfo(np.float32).eps] = 1.0
        return self.mean.astype(np.float32), std.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DrivAerML run_* NPY directories into normalized PKL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing run_<id> subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PKL output directory; defaults to data-dir itself.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PKL outputs.",
    )
    parser.add_argument(
        "--remove-source-runs",
        action="store_true",
        help=(
            "Permanently remove run_* directories only after every non-empty "
            "run has a verified PKL output. Empty run directories are removed too."
        ),
    )
    return parser.parse_args()


def run_id(path: Path) -> int:
    match = re.fullmatch(r"run_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid run directory name: {path.name}")
    return int(match.group(1))


def load_array(path: Path, spec: FieldSpec) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim == 1 and spec.components == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != spec.components:
        raise ValueError(
            f"{path}: expected shape (N,{spec.components}), got {values.shape}"
        )
    if values.shape[0] == 0:
        raise ValueError(f"{path}: array is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: array contains NaN or Inf")
    return np.asarray(values, dtype=np.float32)


def validate_alignment(arrays: dict[str, np.ndarray], run: Path) -> None:
    surface_names = ("x_surface", "y_spressure", "y_wss")
    volume_names = ("x_volume", "y_velocity", "volume_cell_totalpcoeff")
    for group_name, names in (("surface", surface_names), ("volume", volume_names)):
        lengths = {name: len(arrays[name]) for name in names if name in arrays}
        if lengths and len(set(lengths.values())) != 1:
            raise ValueError(
                f"{run.name}: misaligned {group_name} arrays: {lengths}"
            )


def inspect_runs(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coords_min = np.full(3, np.inf, dtype=np.float64)
    coords_max = np.full(3, -np.inf, dtype=np.float64)
    moments = {
        spec.output_name: RunningMoments()
        for spec in FIELD_SPECS
        if spec.kind == "physical"
    }
    records: list[dict[str, Any]] = []

    for position, run in enumerate(run_dirs, start=1):
        started = time.monotonic()
        arrays: dict[str, np.ndarray] = {}
        available_files: list[str] = []
        for spec in FIELD_SPECS:
            source = run / spec.source_name
            if not source.is_file():
                continue
            values = load_array(source, spec)
            arrays[spec.output_name] = values
            available_files.append(spec.source_name)
            if spec.kind == "coordinate":
                coords_min = np.minimum(coords_min, values.min(axis=0))
                coords_max = np.maximum(coords_max, values.max(axis=0))
            else:
                moments[spec.output_name].update(values)

        validate_alignment(arrays, run)
        missing_required = [
            spec.source_name
            for spec in FIELD_SPECS
            if spec.required and spec.source_name not in available_files
        ]
        records.append(
            {
                "run_id": run_id(run),
                "source_run": run.name,
                "available_files": available_files,
                "missing_required_files": missing_required,
                "complete": not missing_required,
                "surface_points": int(len(arrays["x_surface"]))
                if "x_surface" in arrays
                else 0,
                "volume_points": int(len(arrays["x_volume"]))
                if "x_volume" in arrays
                else 0,
            }
        )
        print(
            f"[statistics {position:>3}/{len(run_dirs)}] {run.name}: "
            f"fields={len(arrays)}, complete={not missing_required}, "
            f"{time.monotonic() - started:.2f}s"
        )

    nonempty = [record for record in records if record["available_files"]]
    if not nonempty:
        raise RuntimeError("All run directories are empty")
    if not np.isfinite(coords_min).all() or not np.isfinite(coords_max).all():
        raise RuntimeError("No coordinate arrays were found")

    coords_range = coords_max - coords_min
    coords_range[coords_range <= np.finfo(np.float32).eps] = 1.0
    scalars: dict[str, Any] = {
        "schema_version": "drivaerml-normalized-v1",
        "coords_min": coords_min.astype(np.float32),
        "coords_max": coords_max.astype(np.float32),
        "coords_range": coords_range.astype(np.float32),
        "coordinate_fields": ("x_surface", "x_volume"),
        "physical_fields": tuple(moments),
        "field_mapping": {
            spec.source_name: spec.output_name for spec in FIELD_SPECS
        },
        "normalization": {
            "x_surface": "global_shared_minmax",
            "x_volume": "global_shared_minmax",
            **{name: "global_componentwise_mean_std" for name in moments},
        },
        "samples": nonempty,
        "num_run_directories": len(records),
        "num_converted_samples": len(nonempty),
        "num_complete_samples": sum(record["complete"] for record in nonempty),
        "num_partial_samples": sum(not record["complete"] for record in nonempty),
        "empty_runs": [
            record["source_run"] for record in records if not record["available_files"]
        ],
    }
    for name, accumulator in moments.items():
        mean, std = accumulator.statistics()
        scalars[f"{name}_mean"] = mean
        scalars[f"{name}_std"] = std
        scalars[f"{name}_count"] = accumulator.count
    return nonempty, scalars


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert_run(
    run: Path,
    record: dict[str, Any],
    scalars: dict[str, Any],
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "schema_version": scalars["schema_version"],
        "sample_id": record["run_id"],
        "source_run": record["source_run"],
        "complete": record["complete"],
        "missing_required_files": tuple(record["missing_required_files"]),
    }
    for spec in FIELD_SPECS:
        source = run / spec.source_name
        if not source.is_file():
            continue
        values = load_array(source, spec)
        if spec.kind == "coordinate":
            values = (
                (values - scalars["coords_min"]) / scalars["coords_range"]
            ).astype(np.float32, copy=False)
        else:
            values = (
                (values - scalars[f"{spec.output_name}_mean"])
                / scalars[f"{spec.output_name}_std"]
            ).astype(np.float32, copy=False)
        sample[spec.output_name] = values

    validate_alignment(
        {key: value for key, value in sample.items() if isinstance(value, np.ndarray)},
        run,
    )
    sample["normalization"] = scalars["normalization"].copy()
    return sample


def verify_output(
    destination: Path,
    record: dict[str, Any],
) -> None:
    with destination.open("rb") as stream:
        sample = pickle.load(stream)
    if sample.get("sample_id") != record["run_id"]:
        raise RuntimeError(f"Incorrect sample ID in {destination}")
    for spec in FIELD_SPECS:
        source_exists = spec.source_name in record["available_files"]
        if source_exists != (spec.output_name in sample):
            raise RuntimeError(
                f"Field preservation check failed for {spec.output_name} in {destination}"
            )
        if spec.output_name in sample:
            values = sample[spec.output_name]
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid normalized field in {destination}: {spec.output_name}")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir
    )
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(data_dir.glob("run_*"), key=run_id)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {data_dir}")

    print(f"Dataset       : {data_dir}")
    print(f"PKL output    : {output_dir}")
    print(f"Run directories: {len(run_dirs)}")
    records, scalars = inspect_runs(run_dirs)
    print(
        "Inventory: "
        f"converted={len(records)}, complete={scalars['num_complete_samples']}, "
        f"partial={scalars['num_partial_samples']}, "
        f"empty={len(scalars['empty_runs'])}"
    )

    record_by_id = {record["run_id"]: record for record in records}
    run_by_id = {run_id(run): run for run in run_dirs}
    for position, sample_id in enumerate(sorted(record_by_id), start=1):
        record = record_by_id[sample_id]
        destination = output_dir / f"{sample_id}.pkl"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {destination}; use --overwrite to replace it"
            )
        started = time.monotonic()
        sample = convert_run(run_by_id[sample_id], record, scalars)
        atomic_pickle_dump(sample, destination)
        verify_output(destination, record)
        print(
            f"[convert {position:>3}/{len(records)}] {record['source_run']} "
            f"-> {destination.name}, complete={record['complete']}, "
            f"{time.monotonic() - started:.2f}s"
        )

    scalars["data_dir"] = str(data_dir)
    scalars["output_dir"] = str(output_dir)
    scalars_path = output_dir / SCALARS_FILENAME
    if scalars_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {scalars_path}; use --overwrite to replace it"
        )
    atomic_pickle_dump(scalars, scalars_path)
    with scalars_path.open("rb") as stream:
        verified_scalars = pickle.load(stream)
    if verified_scalars["num_converted_samples"] != len(records):
        raise RuntimeError("normalization_scalars.pkl verification failed")

    expected_outputs = {output_dir / f"{record['run_id']}.pkl" for record in records}
    missing_outputs = [path for path in expected_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f"Missing PKL outputs: {missing_outputs[:5]}")

    if args.remove_source_runs:
        # Every non-empty run has been converted and verified above. Empty run
        # directories contain no NPY data and are included in this exact list.
        for position, run in enumerate(run_dirs, start=1):
            shutil.rmtree(run)
            print(f"[remove {position:>3}/{len(run_dirs)}] {run}")

    print("Conversion complete.")
    print(f"Samples: {len(records)}; scalars: {scalars_path}")
    if not args.remove_source_runs:
        print("Source run_* directories were preserved (read-only conversion).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
