#!/usr/bin/env python3
"""Downsample split-geometry transient DrivAerML PKL data.

The input directory contains one shared ``geometry.pkl`` and numbered time-step
PKLs.  Surface and volume indices are generated once and reused for every time
step, so node correspondence remains fixed throughout the transient sequence.
Already-normalized values are selected without being normalized again.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "run_1_pklFile"
GEOMETRY_FILENAME = "geometry.pkl"
SCALARS_FILENAME = "normalization_scalars.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; it must differ from input-dir.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Use the same target count for surface and volume points.",
    )
    parser.add_argument(
        "--surface-points",
        type=int,
        default=None,
        help="Surface target count; overrides --num-points.",
    )
    parser.add_argument(
        "--volume-points",
        type=int,
        default=None,
        help="Volume target count; overrides --num-points.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.surface_points = (
        args.surface_points if args.surface_points is not None else args.num_points
    )
    args.volume_points = (
        args.volume_points if args.volume_points is not None else args.num_points
    )
    if args.surface_points is None or args.volume_points is None:
        parser.error(
            "set --num-points, or set both --surface-points and --volume-points"
        )
    if args.surface_points <= 0 or args.volume_points <= 0:
        parser.error("target point counts must be positive")
    return args


def numbered_paths(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def as_point_array(values: Any, field: str, components: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != components:
        raise ValueError(
            f"{field} must have shape (N,{components}), got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be numeric, got {array.dtype}")
    return array


def random_indices(count: int, target: int, seed: int, stream: int) -> np.ndarray:
    if target > count:
        raise ValueError(f"Requested {target:,} points, but only {count:,} exist")
    if target == count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return np.sort(rng.choice(count, size=target, replace=False)).astype(
        np.int64, copy=False
    )


def sampled_patch_records(
    source_records: Any, surface_indices: np.ndarray
) -> list[dict[str, Any]]:
    if not isinstance(source_records, list):
        return []
    result: list[dict[str, Any]] = []
    sampled_start = 0
    for source in source_records:
        if not isinstance(source, dict):
            continue
        source_start = int(source.get("start", 0))
        source_count = int(source.get("count", 0))
        left = int(np.searchsorted(surface_indices, source_start, side="left"))
        right = int(
            np.searchsorted(
                surface_indices, source_start + source_count, side="left"
            )
        )
        count = right - left
        record = copy.deepcopy(source)
        record["start"] = sampled_start
        record["count"] = count
        record["original_start"] = source_start
        record["original_count"] = source_count
        result.append(record)
        sampled_start += count
    if sampled_start != len(surface_indices):
        raise ValueError(
            "surface_patches do not cover every sampled surface coordinate"
        )
    return result


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed = numbered_paths(output_dir)
    managed.extend(
        path
        for path in (
            output_dir / GEOMETRY_FILENAME,
            output_dir / SCALARS_FILENAME,
        )
        if path.exists()
    )
    if managed and not overwrite:
        raise FileExistsError(f"Output exists: {managed[0]}; use --overwrite")
    if overwrite:
        for path in managed:
            path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()


def update_sample_records(
    source_scalars: dict[str, Any],
    paths: list[Path],
    surface_points: int,
    volume_points: int,
) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    records = source_scalars.get("samples")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and "sample_id" in record:
                by_id[int(record["sample_id"])] = record

    output: list[dict[str, Any]] = []
    for path in paths:
        sample_id = int(path.stem)
        record = copy.deepcopy(by_id.get(sample_id, {"sample_id": sample_id}))
        if "surface_points" in record:
            record["original_surface_points"] = int(record["surface_points"])
        if "volume_points" in record:
            record["original_volume_points"] = int(record["volume_points"])
        record["surface_points"] = int(surface_points)
        record["volume_points"] = int(volume_points)
        record["output_file"] = path.name
        output.append(record)
    return output


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Output directory must differ from input directory")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    geometry_path = input_dir / GEOMETRY_FILENAME
    scalar_path = input_dir / SCALARS_FILENAME
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Missing {geometry_path}")
    if not scalar_path.is_file():
        raise FileNotFoundError(f"Missing {scalar_path}")
    paths = numbered_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered time-step PKLs found in {input_dir}")

    prepare_output(output_dir, args.overwrite)
    source_scalars = load_pickle(scalar_path)
    if not isinstance(source_scalars, dict):
        raise TypeError(f"{scalar_path} must contain a dictionary")

    print("Loading shared geometry (this can require several GB of memory)...")
    geometry = load_pickle(geometry_path)
    if not isinstance(geometry, dict):
        raise TypeError(f"{geometry_path} must contain a dictionary")
    x_surface = as_point_array(geometry.get("x_surface"), "x_surface", 3)
    x_volume = as_point_array(geometry.get("x_volume"), "x_volume", 3)
    surface_indices = random_indices(
        len(x_surface), args.surface_points, args.seed, stream=0
    )
    volume_indices = random_indices(
        len(x_volume), args.volume_points, args.seed, stream=1
    )

    print("=" * 76)
    print("Transient split-geometry PKL downsampling")
    print(f"Input            : {input_dir}")
    print(f"Output           : {output_dir}")
    print(f"Time steps       : {len(paths)}")
    print(f"Surface          : {len(x_surface):,} -> {len(surface_indices):,}")
    print(f"Volume           : {len(x_volume):,} -> {len(volume_indices):,}")
    print(f"Seed             : {args.seed}")
    print("Index policy     : one fixed pair of indices for every time step")
    print("Normalization    : unchanged")
    print("=" * 76)

    output_geometry = copy.copy(geometry)
    output_geometry["x_surface"] = x_surface[surface_indices]
    output_geometry["x_volume"] = x_volume[volume_indices]
    output_geometry["surface_downsample_indices"] = surface_indices
    output_geometry["volume_downsample_indices"] = volume_indices
    output_geometry["source_surface_patches"] = copy.deepcopy(
        geometry.get("surface_patches", [])
    )
    output_geometry["surface_patches"] = sampled_patch_records(
        geometry.get("surface_patches"), surface_indices
    )
    output_geometry["downsampling"] = {
        "schema": "drivaerml-transient-fixed-indices-v1",
        "original_surface_points": int(len(x_surface)),
        "sampled_surface_points": int(len(surface_indices)),
        "original_volume_points": int(len(x_volume)),
        "sampled_volume_points": int(len(volume_indices)),
        "method": "random",
        "seed": int(args.seed),
        "physical_fields_renormalized": False,
    }
    atomic_pickle_dump(output_geometry, output_dir / GEOMETRY_FILENAME)
    del output_geometry, geometry, x_surface, x_volume

    for position, path in enumerate(paths, start=1):
        sample = load_pickle(path)
        if not isinstance(sample, dict):
            raise TypeError(f"{path} must contain a dictionary")
        pressure = as_point_array(sample.get("y_spressure"), "y_spressure", 1)
        velocity = as_point_array(sample.get("y_velocity"), "y_velocity", 3)
        if len(pressure) <= int(surface_indices[-1]):
            raise ValueError(
                f"{path}: pressure has {len(pressure):,} rows but geometry indices "
                f"require at least {int(surface_indices[-1]) + 1:,}"
            )
        if len(velocity) <= int(volume_indices[-1]):
            raise ValueError(
                f"{path}: velocity has {len(velocity):,} rows but geometry indices "
                f"require at least {int(volume_indices[-1]) + 1:,}"
            )
        result = copy.copy(sample)
        result["y_spressure"] = pressure[surface_indices]
        result["y_velocity"] = velocity[volume_indices]
        result["geometry_file"] = GEOMETRY_FILENAME
        result["normalization_scalars_file"] = SCALARS_FILENAME
        result["downsampling"] = {
            "indices_file": GEOMETRY_FILENAME,
            "surface_index_field": "surface_downsample_indices",
            "volume_index_field": "volume_downsample_indices",
            "method": "random",
            "seed": int(args.seed),
            "physical_fields_renormalized": False,
        }
        atomic_pickle_dump(result, output_dir / path.name)
        print(
            f"[{position:4d}/{len(paths)}] {path.name}: "
            f"pressure {len(pressure):,}->{len(surface_indices):,}, "
            f"velocity {len(velocity):,}->{len(volume_indices):,}"
        )
        del sample, result, pressure, velocity

    output_scalars = copy.deepcopy(source_scalars)
    output_scalars["geometry_file"] = GEOMETRY_FILENAME
    output_scalars["samples"] = update_sample_records(
        source_scalars, paths, len(surface_indices), len(volume_indices)
    )
    output_scalars["downsampling"] = {
        "schema": "drivaerml-transient-fixed-indices-v1",
        "source_directory": str(input_dir),
        "surface_points": int(len(surface_indices)),
        "volume_points": int(len(volume_indices)),
        "method": "random",
        "seed": int(args.seed),
        "indices_file": GEOMETRY_FILENAME,
        "physical_fields_renormalized": False,
    }
    atomic_pickle_dump(output_scalars, output_dir / SCALARS_FILENAME)

    print("Downsampling complete.")
    print(f"Output: {output_dir}")
    print("Source files were preserved; normalized values were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
