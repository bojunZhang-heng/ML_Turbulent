#!/usr/bin/env python3
"""Convert DrivAerML data into normalized per-sample PKL files.

All non-empty ``run_<id>`` directories are converted.  Complete and partial
runs are both preserved; each PKL records whether all expected arrays were
available.  The source run directories are read-only by default and may be
removed only with the explicit ``--remove-source-runs`` option after every
output has been verified.

Surface and volume coordinates use independent dataset-wide min-max ranges.
Legacy v1 PKLs can be migrated without the deleted NPY sources by reversing
their old shared transform. Physical fields remain unchanged during migration.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
    r"/Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000"
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
        "--source-pkl-dir",
        type=Path,
        default=None,
        help=(
            "Migrate existing normalized v1/v2 PKLs instead of reading run_* "
            "NPY directories. Requires normalization_scalars.pkl."
        ),
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


COORDINATE_FIELDS = ("x_surface", "x_volume")
COORDINATE_PREFIX = {
    "x_surface": "surface_coords",
    "x_volume": "volume_coords",
}


def coordinate_scalar_keys(field: str) -> tuple[str, str, str]:
    try:
        prefix = COORDINATE_PREFIX[field]
    except KeyError as error:
        raise ValueError(f"Unknown coordinate field: {field!r}") from error
    return f"{prefix}_min", f"{prefix}_max", f"{prefix}_range"


def coordinate_scalars(
    scalars: dict[str, Any], field: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve v2 field-specific or legacy v1 shared coordinate statistics."""

    min_key, _, range_key = coordinate_scalar_keys(field)
    if min_key in scalars and range_key in scalars:
        return (
            np.asarray(scalars[min_key], dtype=np.float32),
            np.asarray(scalars[range_key], dtype=np.float32),
        )
    if "coords_min" in scalars and "coords_range" in scalars:
        return (
            np.asarray(scalars["coords_min"], dtype=np.float32),
            np.asarray(scalars["coords_range"], dtype=np.float32),
        )
    raise KeyError(
        f"No coordinate statistics for {field!r}; expected {min_key!r}/"
        f"{range_key!r} or legacy coords_min/coords_range"
    )


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
    coordinate_bounds = {
        field: {
            "min": np.full(3, np.inf, dtype=np.float64),
            "max": np.full(3, -np.inf, dtype=np.float64),
        }
        for field in COORDINATE_FIELDS
    }
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
                bounds = coordinate_bounds[spec.output_name]
                bounds["min"] = np.minimum(bounds["min"], values.min(axis=0))
                bounds["max"] = np.maximum(bounds["max"], values.max(axis=0))
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
    for field, bounds in coordinate_bounds.items():
        if not np.isfinite(bounds["min"]).all() or not np.isfinite(bounds["max"]).all():
            raise RuntimeError(f"No coordinate arrays were found for {field}")
    scalars: dict[str, Any] = {
        "schema_version": "drivaerml-normalized-v2",
        "coordinate_fields": COORDINATE_FIELDS,
        "physical_fields": tuple(moments),
        "field_mapping": {
            spec.source_name: spec.output_name for spec in FIELD_SPECS
        },
        "normalization": {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_volume_minmax",
            **{name: "global_componentwise_mean_std" for name in moments},
        },
        "coordinate_normalization_scopes": {
            "x_surface": {
                "statistics_prefix": "surface_coords",
                "population": "all available DrivAerML surface coordinates",
            },
            "x_volume": {
                "statistics_prefix": "volume_coords",
                "population": "all available DrivAerML volume coordinates",
            },
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
    for field, bounds in coordinate_bounds.items():
        values_range = bounds["max"] - bounds["min"]
        values_range[values_range <= np.finfo(np.float32).eps] = 1.0
        min_key, max_key, range_key = coordinate_scalar_keys(field)
        scalars[min_key] = bounds["min"].astype(np.float32)
        scalars[max_key] = bounds["max"].astype(np.float32)
        scalars[range_key] = values_range.astype(np.float32)
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
            coords_min, coords_range = coordinate_scalars(
                scalars, spec.output_name
            )
            values = (
                (values - coords_min) / coords_range
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
            if spec.kind == "coordinate" and (
                np.any(values < -1.0e-5) or np.any(values > 1.0 + 1.0e-5)
            ):
                raise RuntimeError(
                    f"Normalized coordinate outside [0,1] in {destination}: "
                    f"{spec.output_name}"
                )


def numbered_pkl_paths(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def verify_migrated_sample(path: Path) -> None:
    with path.open("rb") as stream:
        sample = pickle.load(stream)
    arrays = {
        key: value for key, value in sample.items() if isinstance(value, np.ndarray)
    }
    validate_alignment(arrays, path)
    for key, values in arrays.items():
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid array {key!r} in {path}")
        if key in COORDINATE_FIELDS and (
            np.any(values < -1.0e-5) or np.any(values > 1.0 + 1.0e-5)
        ):
            raise RuntimeError(f"Migrated {key} outside [0,1] in {path}")


def migrate_processed_pkls(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> int:
    """Migrate old shared-coordinate PKLs to independent v2 coordinates."""

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source PKL directory not found: {source_dir}")
    if source_dir == output_dir:
        raise ValueError(
            "Migration output must differ from --source-pkl-dir so recoverable "
            "source PKLs cannot be overwritten"
        )
    source_scalar_path = source_dir / SCALARS_FILENAME
    if not source_scalar_path.is_file():
        raise FileNotFoundError(f"Missing {source_scalar_path}")
    with source_scalar_path.open("rb") as stream:
        source_scalars = pickle.load(stream)
    if not isinstance(source_scalars, dict):
        raise TypeError(f"Expected scalar dictionary in {source_scalar_path}")
    paths = numbered_pkl_paths(source_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered PKLs found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = numbered_pkl_paths(output_dir)
    output_scalar_path = output_dir / SCALARS_FILENAME
    if (existing or output_scalar_path.exists()) and not overwrite:
        first = existing[0] if existing else output_scalar_path
        raise FileExistsError(f"Migration output exists: {first}; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
        if output_scalar_path.exists():
            output_scalar_path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()

    bounds = {
        field: {
            "min": np.full(3, np.inf, dtype=np.float64),
            "max": np.full(3, -np.inf, dtype=np.float64),
        }
        for field in COORDINATE_FIELDS
    }
    records: list[dict[str, Any]] = []
    print(f"PKL migration source : {source_dir}")
    print(f"PKL migration output : {output_dir}")
    print(f"Samples              : {len(paths)}")

    # Pass 1: recover physical coordinates and collect independent bounds.
    for position, path in enumerate(paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        available_coordinates = []
        for field in COORDINATE_FIELDS:
            if field not in sample:
                continue
            old_min, old_range = coordinate_scalars(source_scalars, field)
            physical = (
                np.asarray(sample[field], dtype=np.float32) * old_range + old_min
            )
            bounds[field]["min"] = np.minimum(
                bounds[field]["min"], physical.min(axis=0)
            )
            bounds[field]["max"] = np.maximum(
                bounds[field]["max"], physical.max(axis=0)
            )
            available_coordinates.append(field)
        if not available_coordinates:
            raise ValueError(f"{path} contains no coordinate field")
        records.append(
            {
                "run_id": int(path.stem),
                "surface_points": int(len(sample.get("x_surface", ()))),
                "volume_points": int(len(sample.get("x_volume", ()))),
            }
        )
        print(f"[recover statistics {position}/{len(paths)}] {path.name}")
        del sample
        gc.collect()

    new_scalars = copy.deepcopy(source_scalars)
    new_scalars.pop("coords_min", None)
    new_scalars.pop("coords_max", None)
    new_scalars.pop("coords_range", None)
    new_scalars["schema_version"] = "drivaerml-normalized-v2"
    new_scalars["coordinate_fields"] = COORDINATE_FIELDS
    new_scalars["migrated_from_schema"] = source_scalars.get("schema_version")
    new_scalars["migrated_from_directory"] = str(source_dir)
    new_scalars["output_dir"] = str(output_dir)
    new_scalars["num_converted_samples"] = len(paths)
    normalization = dict(new_scalars.get("normalization", {}))
    normalization.update(
        {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_volume_minmax",
        }
    )
    new_scalars["normalization"] = normalization
    new_scalars["coordinate_normalization_scopes"] = {
        "x_surface": {
            "statistics_prefix": "surface_coords",
            "population": "all surface coordinates reconstructed from source PKLs",
        },
        "x_volume": {
            "statistics_prefix": "volume_coords",
            "population": "all volume coordinates reconstructed from source PKLs",
        },
    }
    migrated_ids = {int(path.stem) for path in paths}
    source_records = source_scalars.get("samples")
    if isinstance(source_records, list):
        filtered = [
            record
            for record in source_records
            if isinstance(record, dict)
            and int(record.get("run_id", record.get("sample_id", -1)))
            in migrated_ids
        ]
        new_scalars["samples"] = filtered or records
    else:
        new_scalars["samples"] = records
    for field, field_bounds in bounds.items():
        if not np.isfinite(field_bounds["min"]).all():
            raise RuntimeError(f"No {field} arrays found during migration")
        values_range = field_bounds["max"] - field_bounds["min"]
        values_range[values_range <= np.finfo(np.float32).eps] = 1.0
        min_key, max_key, range_key = coordinate_scalar_keys(field)
        new_scalars[min_key] = field_bounds["min"].astype(np.float32)
        new_scalars[max_key] = field_bounds["max"].astype(np.float32)
        new_scalars[range_key] = values_range.astype(np.float32)

    # Pass 2: only replace coordinates; physical arrays stay unchanged.
    for position, path in enumerate(paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        for field in COORDINATE_FIELDS:
            if field not in sample:
                continue
            old_min, old_range = coordinate_scalars(source_scalars, field)
            physical = (
                np.asarray(sample[field], dtype=np.float32) * old_range + old_min
            )
            new_min, new_range = coordinate_scalars(new_scalars, field)
            sample[field] = ((physical - new_min) / new_range).astype(
                np.float32, copy=False
            )
        sample["schema_version"] = new_scalars["schema_version"]
        sample_normalization = dict(sample.get("normalization", {}))
        for field in COORDINATE_FIELDS:
            if field in sample:
                group = "surface" if field == "x_surface" else "volume"
                sample_normalization[field] = f"global_{group}_minmax"
        sample["normalization"] = sample_normalization
        destination = output_dir / path.name
        atomic_pickle_dump(sample, destination)
        verify_migrated_sample(destination)
        print(f"[migrate {position}/{len(paths)}] {path.name}")
        del sample
        gc.collect()

    atomic_pickle_dump(new_scalars, output_scalar_path)
    print(f"Migration complete: {len(paths)} samples")
    print(f"Normalization scalars: {output_scalar_path}")
    print("Physical fields were copied without re-normalization.")
    return 0


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print("Preprocessing configuration: process_DrivAerML.py")
    for name, value in sorted(vars(args).items()):
        print(f"{name:<24}: {value}")
    print("=" * 72)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir
    )
    if args.source_pkl_dir is not None:
        if args.remove_source_runs:
            raise ValueError(
                "--remove-source-runs cannot be used with --source-pkl-dir"
            )
        if args.output_dir is None:
            raise ValueError(
                "PKL migration requires --output-dir different from source"
            )
        return migrate_processed_pkls(
            args.source_pkl_dir, output_dir, args.overwrite
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
    required_coordinate_keys = {
        "surface_coords_min",
        "surface_coords_max",
        "surface_coords_range",
        "volume_coords_min",
        "volume_coords_max",
        "volume_coords_range",
    }
    missing_coordinate_keys = required_coordinate_keys - set(verified_scalars)
    if missing_coordinate_keys:
        raise RuntimeError(
            "Separate surface/volume normalization statistics are missing: "
            f"{sorted(missing_coordinate_keys)}"
        )

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
