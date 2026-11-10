#!/usr/bin/env python3
"""Shared migration logic for using the surface scale on all coordinates."""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SCALARS_FILENAME = "normalization_scalars.pkl"
COORDINATE_FIELDS = ("x_surface", "x_volume")
FIELD_PREFIXES = {
    "x_surface": "surface_coords",
    "x_volume": "volume_coords",
}


def vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values, got {array}")
    return array


def source_transform(
    scalars: dict[str, Any], field: str
) -> tuple[np.ndarray, np.ndarray, str]:
    prefix = FIELD_PREFIXES[field]
    min_key = f"{prefix}_min"
    range_key = f"{prefix}_range"
    if min_key in scalars and range_key in scalars:
        minimum = vector3(scalars[min_key], min_key)
        values_range = vector3(scalars[range_key], range_key)
        source = prefix
    elif "coords_min" in scalars and "coords_range" in scalars:
        minimum = vector3(scalars["coords_min"], "coords_min")
        values_range = vector3(scalars["coords_range"], "coords_range")
        source = "coords"
    else:
        raise KeyError(
            f"Cannot reconstruct {field}: expected {min_key}/{range_key} "
            "or legacy coords_min/coords_range"
        )
    if np.any(values_range <= 0.0):
        raise ValueError(f"Invalid non-positive coordinate range for {field}")
    return minimum, values_range, source


def numbered_pkls(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


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


def recover_physical(
    normalized: np.ndarray,
    minimum: np.ndarray,
    values_range: np.ndarray,
    field: str,
    source: Path,
) -> np.ndarray:
    values = np.asarray(normalized)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{source}: {field} must have shape (N,3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: {field} contains NaN or Inf")
    return values.astype(np.float64) * values_range + minimum


def resolve_surface_scale(
    paths: list[Path], scalars: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    surface_min, surface_range, source = source_transform(scalars, "x_surface")
    if "surface_coords_min" in scalars and "surface_coords_range" in scalars:
        surface_max = surface_min + surface_range
        return surface_min, surface_max, surface_range, "stored_surface_statistics"

    recovered_min = np.full(3, np.inf, dtype=np.float64)
    recovered_max = np.full(3, -np.inf, dtype=np.float64)
    surface_samples = 0
    for position, path in enumerate(paths, start=1):
        sample = load_pickle(path)
        if not isinstance(sample, dict) or "x_surface" not in sample:
            continue
        physical = recover_physical(
            sample["x_surface"], surface_min, surface_range, "x_surface", path
        )
        recovered_min = np.minimum(recovered_min, physical.min(axis=0))
        recovered_max = np.maximum(recovered_max, physical.max(axis=0))
        surface_samples += 1
        print(f"[surface statistics {position:>4}/{len(paths)}] {path.name}")
    if surface_samples == 0 or not np.isfinite(recovered_min).all():
        raise RuntimeError("No x_surface arrays were found")
    recovered_range = recovered_max - recovered_min
    recovered_range[recovered_range <= np.finfo(np.float32).eps] = 1.0
    return recovered_min, recovered_max, recovered_range, f"recovered_from_{source}"


def preserved_arrays(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: value
        for key, value in sample.items()
        if isinstance(value, np.ndarray) and key not in COORDINATE_FIELDS
    }


def update_sample_metadata(sample: dict[str, Any], schema_version: str) -> None:
    sample["schema_version"] = schema_version
    normalization = dict(sample.get("normalization", {}))
    normalization["x_surface"] = "global_surface_minmax"
    normalization["x_volume"] = "global_surface_minmax"
    sample["normalization"] = normalization
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        metadata = copy.deepcopy(metadata)
        metadata["coordinate_normalization"] = {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_surface_minmax",
            "note": "Volume coordinates may fall outside [0, 1].",
        }
        sample["metadata"] = metadata


def verify_output(
    source_sample: dict[str, Any],
    output_path: Path,
    target_min: np.ndarray,
    target_range: np.ndarray,
    source_transforms: dict[str, tuple[np.ndarray, np.ndarray, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = load_pickle(output_path)
    if not isinstance(output, dict):
        raise TypeError(f"{output_path}: expected a dictionary")
    for key, original in preserved_arrays(source_sample).items():
        if key not in output or not np.array_equal(original, output[key]):
            raise RuntimeError(f"{output_path}: non-coordinate array {key!r} changed")

    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for field in COORDINATE_FIELDS:
        if field not in source_sample:
            continue
        old_min, old_range, _ = source_transforms[field]
        expected_physical = recover_physical(
            source_sample[field], old_min, old_range, field, output_path
        )
        migrated = np.asarray(output[field], dtype=np.float64)
        reconstructed = migrated * target_range + target_min
        if not np.allclose(
            expected_physical, reconstructed, rtol=2.0e-6, atol=2.0e-5
        ):
            max_error = float(np.max(np.abs(expected_physical - reconstructed)))
            raise RuntimeError(
                f"{output_path}: {field} physical-coordinate mismatch; "
                f"max error={max_error:.6g}"
            )
        bounds[field] = (migrated.min(axis=0), migrated.max(axis=0))
    return bounds


def migrate_directory(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    schema_version: str,
    overwrite: bool,
) -> int:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Output directory must differ from input directory")

    scalar_path = input_dir / SCALARS_FILENAME
    if not scalar_path.is_file():
        raise FileNotFoundError(f"Missing {scalar_path}")
    scalars = load_pickle(scalar_path)
    if not isinstance(scalars, dict):
        raise TypeError(f"{scalar_path}: expected a dictionary")
    paths = numbered_pkls(input_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered PKL files found in {input_dir}")

    source_transforms = {
        field: source_transform(scalars, field) for field in COORDINATE_FIELDS
    }
    target_min, target_max, target_range, scale_source = resolve_surface_scale(
        paths, scalars
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / path.name for path in paths]
    destinations.append(output_dir / SCALARS_FILENAME)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()

    print("=" * 72)
    print(f"Dataset                 : {dataset_name}")
    print(f"Input directory         : {input_dir}")
    print(f"Output directory        : {output_dir}")
    print(f"Samples                 : {len(paths)}")
    print(f"Surface-scale source    : {scale_source}")
    print(f"Surface minimum         : {target_min}")
    print(f"Surface maximum         : {target_max}")
    print("Physical fields         : copied unchanged")
    print("=" * 72)

    normalized_bounds = {
        field: {
            "min": np.full(3, np.inf, dtype=np.float64),
            "max": np.full(3, -np.inf, dtype=np.float64),
        }
        for field in COORDINATE_FIELDS
    }
    for position, source in enumerate(paths, start=1):
        sample = load_pickle(source)
        if not isinstance(sample, dict):
            raise TypeError(f"{source}: expected a dictionary")
        # Coordinate arrays are replaced below; all large physical arrays can
        # remain shared because this migration never mutates them in place.
        result = sample.copy()
        for field in COORDINATE_FIELDS:
            if field not in sample:
                continue
            old_min, old_range, _ = source_transforms[field]
            physical = recover_physical(
                sample[field], old_min, old_range, field, source
            )
            result[field] = ((physical - target_min) / target_range).astype(
                np.float32
            )
        update_sample_metadata(result, schema_version)
        destination = output_dir / source.name
        atomic_pickle_dump(result, destination)
        sample_bounds = verify_output(
            sample, destination, target_min, target_range, source_transforms
        )
        for field, (minimum, maximum) in sample_bounds.items():
            normalized_bounds[field]["min"] = np.minimum(
                normalized_bounds[field]["min"], minimum
            )
            normalized_bounds[field]["max"] = np.maximum(
                normalized_bounds[field]["max"], maximum
            )
        print(f"[migrate {position:>4}/{len(paths)}] {source.name}")

    output_scalars = copy.deepcopy(scalars)
    original_coordinate_scalars = {}
    for key in (
        "coords_min", "coords_max", "coords_range",
        "surface_coords_min", "surface_coords_max", "surface_coords_range",
        "volume_coords_min", "volume_coords_max", "volume_coords_range",
    ):
        if key in scalars:
            original_coordinate_scalars[key] = copy.deepcopy(scalars[key])
    output_scalars["source_coordinate_scalars"] = original_coordinate_scalars
    output_scalars["schema_version"] = schema_version
    output_scalars["coordinate_fields"] = COORDINATE_FIELDS
    output_scalars["coordinate_scale_policy"] = "surface_scale_for_surface_and_volume"
    output_scalars["volume_coordinates_may_exceed_unit_interval"] = True
    output_scalars["migrated_from_directory"] = str(input_dir)
    output_scalars["output_dir"] = str(output_dir)
    for prefix in ("coords", "surface_coords", "volume_coords"):
        output_scalars[f"{prefix}_min"] = target_min.astype(np.float32)
        output_scalars[f"{prefix}_max"] = target_max.astype(np.float32)
        output_scalars[f"{prefix}_range"] = target_range.astype(np.float32)
    normalization = dict(output_scalars.get("normalization", {}))
    normalization.update(
        {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_surface_minmax",
        }
    )
    output_scalars["normalization"] = normalization
    output_scalars["coordinate_normalization_scopes"] = {
        "x_surface": {
            "statistics_prefix": "surface_coords",
            "population": "surface coordinates",
        },
        "x_volume": {
            "statistics_prefix": "surface_coords",
            "population": "surface coordinates; volume values may be outside [0,1]",
        },
    }
    output_scalars["normalized_coordinate_bounds"] = {
        field: {
            "min": values["min"].astype(np.float32),
            "max": values["max"].astype(np.float32),
        }
        for field, values in normalized_bounds.items()
        if np.isfinite(values["min"]).all()
    }
    atomic_pickle_dump(output_scalars, output_dir / SCALARS_FILENAME)
    print(f"Updated scalars          : {output_dir / SCALARS_FILENAME}")
    for field, values in output_scalars["normalized_coordinate_bounds"].items():
        print(
            f"{field:24s}: normalized min={values['min']}, "
            f"max={values['max']}"
        )
    print("Migration complete.")
    return 0


def build_parser(dataset_name: str, default_input: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Re-normalize {dataset_name} surface and volume coordinates with "
            "one dataset-wide surface-coordinate min-max scale."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-dir>_surface_scale.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_cli(
    dataset_name: str,
    schema_version: str,
    default_input: Path,
) -> int:
    parser = build_parser(dataset_name, default_input)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        input_path = args.input_dir.expanduser()
        output_dir = input_path.parent / f"{input_path.name}_surface_scale"
    return migrate_directory(
        input_dir=args.input_dir,
        output_dir=output_dir,
        dataset_name=dataset_name,
        schema_version=schema_version,
        overwrite=args.overwrite,
    )
