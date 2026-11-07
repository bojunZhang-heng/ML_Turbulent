#!/usr/bin/env python3
"""Convert legacy MLCFD PKLs to split surface/volume normalized PKLs.

The source files produced by ``process_mlcfd.py`` store surface and volume
nodes in one ``input``/``output`` array.  This converter first reverses that
normalization, then computes independent dataset-wide coordinate min-max
statistics for the surface and volume point sets.  Surface pressure and volume
velocity keep component-wise mean/std normalization.

The output contains both DrivAerStar-style split fields and the legacy combined
fields, so ShapeNet training can select one of these tasks without regenerating
the data:

* seven input features -> velocity and pressure;
* surface coordinates -> surface pressure;
* volume coordinates -> volume velocity.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "mlcfd_data" / "pkl_test"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "mlcfd_data" / "pkl_test_v2"
SCALARS_FILENAME = "normalization_scalars.pkl"
SCHEMA_VERSION = "mlcfd-split-fields-v3"


@dataclass
class RunningMoments:
    """Numerically stable population moments accumulated in float64."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError(f"Expected non-empty (N,C) values, got {values.shape}")
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        if self.mean is None or self.m2 is None or self.mean.shape != batch_mean.shape:
            raise ValueError("Physical field component count changed between samples")
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("No values were accumulated")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std[std <= np.finfo(np.float32).eps] = 1.0
        return self.mean.astype(np.float32), std.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-scalars",
        type=Path,
        default=None,
        help="Defaults to INPUT_DIR/normalization_scalars.pkl.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N numbered PKLs for a smoke test.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def numbered_pickles(directory: Path) -> list[Path]:
    paths = [path for path in directory.glob("*.pkl") if path.stem.isdigit()]
    return sorted(paths, key=lambda path: int(path.stem))


def require_vector(scalars: dict[str, Any], key: str, components: int) -> np.ndarray:
    if key not in scalars:
        raise KeyError(f"Source normalization file is missing {key!r}")
    values = np.asarray(scalars[key], dtype=np.float32)
    if values.shape != (components,):
        raise ValueError(f"{key} must have shape ({components},), got {values.shape}")
    return values


def validate_source_scalars(scalars: dict[str, Any]) -> dict[str, np.ndarray]:
    values = {
        "coords_min": require_vector(scalars, "coords_min", 3),
        "coords_range": require_vector(scalars, "coords_range", 3),
        "output_mean": require_vector(scalars, "output_mean", 4),
        "output_std": require_vector(scalars, "output_std", 4),
    }
    if np.any(values["coords_range"] <= 0.0) or np.any(values["output_std"] <= 0.0):
        raise ValueError("Source coordinate ranges and output std values must be positive")
    return values


def decode_sample(
    path: Path,
    source_scalars: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    sample = load_pickle(path)
    if not isinstance(sample, dict):
        raise TypeError(f"{path}: expected a dictionary")
    if sample.get("schema_version") == SCHEMA_VERSION:
        raise ValueError(f"{path}: already uses {SCHEMA_VERSION}; refusing double processing")
    model_input = np.asarray(sample.get("input", sample.get("x")), dtype=np.float32)
    target = np.asarray(sample.get("output", sample.get("y")), dtype=np.float32)
    surface_mask = np.asarray(
        sample.get("surface_mask", sample.get("surf")), dtype=bool
    ).reshape(-1)
    if model_input.ndim != 2 or model_input.shape[1] != 7:
        raise ValueError(f"{path}: expected input shape (N,7), got {model_input.shape}")
    if target.shape != (len(model_input), 4):
        raise ValueError(f"{path}: expected output shape ({len(model_input)},4), got {target.shape}")
    if surface_mask.shape != (len(model_input),):
        raise ValueError(f"{path}: invalid surface_mask shape {surface_mask.shape}")
    if not surface_mask.any() or surface_mask.all():
        raise ValueError(f"{path}: both surface and volume nodes are required")
    if not np.isfinite(model_input).all() or not np.isfinite(target).all():
        raise ValueError(f"{path}: input/output contains NaN or Inf")

    raw_coords = (
        model_input[:, :3] * source_scalars["coords_range"]
        + source_scalars["coords_min"]
    ).astype(np.float32, copy=False)
    raw_target = (
        target * source_scalars["output_std"] + source_scalars["output_mean"]
    ).astype(np.float32, copy=False)
    return sample, raw_coords, raw_target, surface_mask


def collect_statistics(
    sample_paths: list[Path],
    source_scalars: dict[str, np.ndarray],
) -> dict[str, Any]:
    bounds = {
        "surface": {
            "min": np.full(3, np.inf, dtype=np.float64),
            "max": np.full(3, -np.inf, dtype=np.float64),
        },
        "volume": {
            "min": np.full(3, np.inf, dtype=np.float64),
            "max": np.full(3, -np.inf, dtype=np.float64),
        },
    }
    pressure_moments = RunningMoments()
    velocity_moments = RunningMoments()
    records: list[dict[str, Any]] = []

    for position, path in enumerate(sample_paths, start=1):
        started = time.monotonic()
        sample, coords, target, surface_mask = decode_sample(path, source_scalars)
        surface_coords = coords[surface_mask]
        volume_coords = coords[~surface_mask]
        for name, values in (("surface", surface_coords), ("volume", volume_coords)):
            bounds[name]["min"] = np.minimum(bounds[name]["min"], values.min(axis=0))
            bounds[name]["max"] = np.maximum(bounds[name]["max"], values.max(axis=0))
        pressure_moments.update(target[surface_mask, 3:4])
        velocity_moments.update(target[~surface_mask, :3])
        records.append(
            {
                "sample_id": sample.get("sample_id", path.stem),
                "source_pkl": str(path),
                "surface_points": int(surface_mask.sum()),
                "volume_points": int((~surface_mask).sum()),
            }
        )
        print(
            f"[statistics {position}/{len(sample_paths)}] {path.name}: "
            f"surface={surface_mask.sum()}, volume={(~surface_mask).sum()}, "
            f"{time.monotonic() - started:.2f}s"
        )

    pressure_mean, pressure_std = pressure_moments.statistics()
    velocity_mean, velocity_std = velocity_moments.statistics()
    scalars: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_fields": ("x_surface", "x_volume"),
        "physical_fields": ("y_spressure", "y_velocity"),
        "field_mapping": {
            "surface_coordinates": "x_surface",
            "surface_pressure": "y_spressure",
            "volume_coordinates": "x_volume",
            "volume_velocity": "y_velocity",
        },
        "normalization": {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_volume_minmax",
            "y_spressure": "global_componentwise_mean_std",
            "y_velocity": "global_componentwise_mean_std",
            "sdf": "source_global_minmax_unchanged",
            "normals": "unit_vector_unchanged",
        },
        "coordinate_normalization_scopes": {
            "x_surface": {
                "statistics_prefix": "surface_coords",
                "population": "all selected MLCFD surface points",
            },
            "x_volume": {
                "statistics_prefix": "volume_coords",
                "population": "all selected MLCFD exterior volume points",
            },
        },
        "coordinate_semantics": "mesh_points",
        "y_spressure_mean": pressure_mean,
        "y_spressure_std": pressure_std,
        "y_spressure_count": pressure_moments.count,
        "y_velocity_mean": velocity_mean,
        "y_velocity_std": velocity_std,
        "y_velocity_count": velocity_moments.count,
        "output_mean": np.concatenate((velocity_mean, pressure_mean)).astype(np.float32),
        "output_std": np.concatenate((velocity_std, pressure_std)).astype(np.float32),
        "input_features": ("x", "y", "z", "sdf", "normal_x", "normal_y", "normal_z"),
        "output_features": ("velocity_x", "velocity_y", "velocity_z", "pressure"),
        "samples": records,
        "num_samples": len(records),
    }
    for name, prefix in (("surface", "surface_coords"), ("volume", "volume_coords")):
        minimum = bounds[name]["min"]
        maximum = bounds[name]["max"]
        values_range = maximum - minimum
        values_range[values_range <= np.finfo(np.float32).eps] = 1.0
        scalars[f"{prefix}_min"] = minimum.astype(np.float32)
        scalars[f"{prefix}_max"] = maximum.astype(np.float32)
        scalars[f"{prefix}_range"] = values_range.astype(np.float32)
    return scalars


def normalize_coordinates(values: np.ndarray, scalars: dict[str, Any], prefix: str) -> np.ndarray:
    return (
        (values - scalars[f"{prefix}_min"]) / scalars[f"{prefix}_range"]
    ).astype(np.float32, copy=False)


def convert_sample(
    path: Path,
    destination: Path,
    source_scalars: dict[str, np.ndarray],
    scalars: dict[str, Any],
) -> None:
    source, raw_coords, raw_target, surface_mask = decode_sample(path, source_scalars)
    volume_mask = ~surface_mask
    x_surface = normalize_coordinates(raw_coords[surface_mask], scalars, "surface_coords")
    x_volume = normalize_coordinates(raw_coords[volume_mask], scalars, "volume_coords")
    y_spressure = (
        (raw_target[surface_mask, 3:4] - scalars["y_spressure_mean"])
        / scalars["y_spressure_std"]
    ).astype(np.float32, copy=False)
    y_velocity = (
        (raw_target[volume_mask, :3] - scalars["y_velocity_mean"])
        / scalars["y_velocity_std"]
    ).astype(np.float32, copy=False)

    combined_input = np.asarray(source.get("input", source.get("x")), dtype=np.float32).copy()
    combined_input[surface_mask, :3] = x_surface
    combined_input[volume_mask, :3] = x_volume
    combined_output = np.empty_like(raw_target, dtype=np.float32)
    combined_output[:, :3] = (
        (raw_target[:, :3] - scalars["y_velocity_mean"])
        / scalars["y_velocity_std"]
    )
    combined_output[:, 3:4] = (
        (raw_target[:, 3:4] - scalars["y_spressure_mean"])
        / scalars["y_spressure_std"]
    )

    sample = dict(source)
    sample.update(
        {
            "schema_version": SCHEMA_VERSION,
            "x_surface": x_surface,
            "y_spressure": y_spressure,
            "x_volume": x_volume,
            "y_velocity": y_velocity,
            "input": combined_input,
            "output": combined_output,
            "x": combined_input,
            "y": combined_output,
            "pos": combined_input[:, :3].copy(),
            "surface_mask": surface_mask,
            "surf": surface_mask,
            "normalization": dict(scalars["normalization"]),
            "metadata": {
                **dict(source.get("metadata", {})),
                "source_schema_version": source.get("schema_version"),
                "surface_points": int(surface_mask.sum()),
                "volume_points": int(volume_mask.sum()),
                "combined_node_order": "source order preserved",
            },
        }
    )
    atomic_pickle_dump(sample, destination)


def verify_output(path: Path) -> None:
    sample = load_pickle(path)
    required = (
        "x_surface",
        "y_spressure",
        "x_volume",
        "y_velocity",
        "input",
        "output",
        "surface_mask",
    )
    missing = [name for name in required if name not in sample]
    if missing:
        raise RuntimeError(f"{path}: missing output fields {missing}")
    if len(sample["x_surface"]) != len(sample["y_spressure"]):
        raise RuntimeError(f"{path}: surface fields are not aligned")
    if len(sample["x_volume"]) != len(sample["y_velocity"]):
        raise RuntimeError(f"{path}: volume fields are not aligned")
    for field in required[:-1]:
        if not np.isfinite(np.asarray(sample[field])).all():
            raise RuntimeError(f"{path}: {field} contains NaN or Inf")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_scalars_path = (
        args.source_scalars.expanduser().resolve()
        if args.source_scalars is not None
        else input_dir / SCALARS_FILENAME
    )
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ to prevent double normalization")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not source_scalars_path.is_file():
        raise FileNotFoundError(f"Source normalization file not found: {source_scalars_path}")
    paths = numbered_pickles(input_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No numbered PKL files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / path.name for path in paths]
    scalar_destination = output_dir / SCALARS_FILENAME
    existing = [path for path in (*destinations, scalar_destination) if path.exists()]
    if existing and not args.overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {preview}")

    source_scalars_raw = load_pickle(source_scalars_path)
    if source_scalars_raw.get("schema_version") == SCHEMA_VERSION:
        raise ValueError("The source directory is already converted; refusing double normalization")
    source_scalars = validate_source_scalars(source_scalars_raw)
    print("=" * 78)
    print("MLCFD split-field preprocessing")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Source scalars  : {source_scalars_path}")
    print(f"Samples         : {len(paths)}")
    print("=" * 78)

    scalars = collect_statistics(paths, source_scalars)
    scalars["source_directory"] = str(input_dir)
    scalars["source_normalization_scalars"] = str(source_scalars_path)
    for position, (source_path, destination) in enumerate(
        zip(paths, destinations), start=1
    ):
        started = time.monotonic()
        convert_sample(source_path, destination, source_scalars, scalars)
        verify_output(destination)
        print(
            f"[write {position}/{len(paths)}] {destination.name}: "
            f"{time.monotonic() - started:.2f}s"
        )
    atomic_pickle_dump(scalars, scalar_destination)
    print(f"Wrote normalization metadata: {scalar_destination}")
    print("Done. Source PKLs were not modified.")


if __name__ == "__main__":
    main()
