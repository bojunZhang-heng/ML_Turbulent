#!/usr/bin/env python3
"""Downsample normalized DrivAerNet-CFD volume-velocity PKL files.

The input schema is ``x_volume`` and ``y_velocity`` with shape ``(1, N, 3)``.
One shared index array is applied to coordinates, velocity and optional source
point IDs.  Values are never normalized again.
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
DEFAULT_INPUT_DIR = SCRIPT_DIR / "DrivAerNet-CFD" / "volume_velocity_pklFile"
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
        required=True,
        help="Number of volume points retained in every sample.",
    )
    parser.add_argument(
        "--method",
        choices=("random", "voxel"),
        default="random",
        help="Random is recommended for multi-million-point volume fields.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_points <= 0:
        parser.error("--num-points must be positive")
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
        if temporary.exists():
            temporary.unlink()


def as_batched_xyz(values: Any, field: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 3:
        raise ValueError(f"{field} must have shape (1,N,3), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-numeric values or NaN/Inf")
    return array


def random_indices(
    count: int, target: int, rng: np.random.Generator
) -> np.ndarray:
    return np.sort(rng.choice(count, size=target, replace=False)).astype(
        np.int64, copy=False
    )


def voxel_indices(
    coordinates: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select exactly target spatially distributed points."""

    xyz = np.asarray(coordinates, dtype=np.float32)
    xyz_min = xyz.min(axis=0, keepdims=True)
    xyz_range = xyz.max(axis=0, keepdims=True) - xyz_min
    xyz_range[xyz_range <= np.finfo(np.float32).eps] = 1.0
    unit = np.clip((xyz - xyz_min) / xyz_range, 0.0, 1.0)
    permutation = rng.permutation(len(unit))
    resolution = max(2, int(np.ceil(np.cbrt(target))))
    representatives = np.empty(0, dtype=np.int64)

    for _ in range(16):
        bins = np.minimum((unit * resolution).astype(np.int64), resolution - 1)
        keys = (
            bins[:, 0]
            + resolution * bins[:, 1]
            + resolution * resolution * bins[:, 2]
        )
        _, first = np.unique(keys[permutation], return_index=True)
        representatives = permutation[first]
        if len(representatives) >= target:
            return np.sort(
                rng.choice(representatives, size=target, replace=False)
            ).astype(np.int64, copy=False)
        resolution = int(np.ceil(resolution * 1.5))

    selected = np.zeros(len(unit), dtype=bool)
    selected[representatives] = True
    remaining = np.flatnonzero(~selected)
    supplement = rng.choice(
        remaining, size=target - len(representatives), replace=False
    )
    return np.sort(np.concatenate((representatives, supplement))).astype(
        np.int64, copy=False
    )


def select_indices(
    coordinates: np.ndarray,
    target: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    count = coordinates.shape[0]
    if target > count:
        raise ValueError(f"Requested {target} points, but sample contains {count}")
    if target == count:
        return np.arange(count, dtype=np.int64)
    if method == "random":
        return random_indices(count, target, rng)
    return voxel_indices(coordinates, target, rng)


def downsample_sample(
    sample: dict[str, Any],
    target: int,
    method: str,
    seed: int,
    sample_id: int,
) -> dict[str, Any]:
    if "x_volume" not in sample or "y_velocity" not in sample:
        raise KeyError("Sample must contain x_volume and y_velocity")

    coordinates = as_batched_xyz(sample["x_volume"], "x_volume")
    velocity = as_batched_xyz(sample["y_velocity"], "y_velocity")
    if coordinates.shape != velocity.shape:
        raise ValueError(
            f"x_volume {coordinates.shape} and y_velocity {velocity.shape} "
            "are not aligned"
        )

    original_count = coordinates.shape[1]
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_id]))
    indices = select_indices(coordinates[0], target, method, rng)

    result = copy.copy(sample)
    result["x_volume"] = coordinates[:, indices, :]
    result["y_velocity"] = velocity[:, indices, :]

    source_ids = sample.get("source_point_ids")
    if source_ids is not None:
        source_ids = np.asarray(source_ids)
        if source_ids.ndim == 1:
            source_ids = source_ids[None, :]
        if source_ids.shape != (1, original_count):
            raise ValueError(
                f"source_point_ids must have shape (1,N), got {source_ids.shape}"
            )
        result["source_point_ids"] = source_ids[:, indices]

    result["volume_downsample_indices"] = indices[None, :]
    result["downsampling"] = {
        "schema": "drivaernet-cfd-volume-v1",
        "original_num_points": int(original_count),
        "sampled_num_points": int(target),
        "method": method,
        "seed": int(seed),
        "index_field": "volume_downsample_indices",
    }
    return result


def verify_sample(sample: dict[str, Any], target: int, path: Path) -> None:
    for field in ("x_volume", "y_velocity"):
        values = as_batched_xyz(sample[field], field)
        if values.shape != (1, target, 3):
            raise RuntimeError(f"Invalid {field} shape in {path}: {values.shape}")
    indices = np.asarray(sample["volume_downsample_indices"])
    if indices.shape != (1, target):
        raise RuntimeError(f"Invalid downsample index shape in {path}: {indices.shape}")
    if "source_point_ids" in sample:
        source_ids = np.asarray(sample["source_point_ids"])
        if source_ids.shape != (1, target):
            raise RuntimeError(f"Invalid source_point_ids shape in {path}")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = numbered_paths(output_dir)
    scalar_path = output_dir / SCALARS_FILENAME
    if (existing or scalar_path.exists()) and not overwrite:
        first = existing[0] if existing else scalar_path
        raise FileExistsError(f"Output exists: {first}; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
        if scalar_path.exists():
            scalar_path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Output directory must differ from input directory")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    scalar_path = input_dir / SCALARS_FILENAME
    if not scalar_path.is_file():
        raise FileNotFoundError(f"Missing {scalar_path}")
    with scalar_path.open("rb") as stream:
        source_scalars = pickle.load(stream)

    paths = numbered_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered PKLs found in {input_dir}")
    prepare_output(output_dir, args.overwrite)

    print("=" * 72)
    print("DrivAerNet-CFD volume downsampling")
    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Samples     : {len(paths)}")
    print(f"Target points: {args.num_points}")
    print(f"Method      : {args.method}")
    print(f"Seed        : {args.seed}")
    print("Normalization: unchanged")
    print("=" * 72)

    output_scalars = copy.deepcopy(source_scalars)
    for position, path in enumerate(paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        sample_id = int(path.stem)
        result = downsample_sample(
            sample, args.num_points, args.method, args.seed, sample_id
        )
        destination = output_dir / path.name
        atomic_pickle_dump(result, destination)
        verify_sample(result, args.num_points, destination)
        original = result["downsampling"]["original_num_points"]
        print(
            f"[{position:4d}/{len(paths)}] {path.name}: "
            f"{original} -> {args.num_points}"
        )
        del sample, result

    output_scalars["downsampling"] = {
        "source_directory": str(input_dir),
        "point_set": "volume",
        "num_points": int(args.num_points),
        "method": args.method,
        "seed": int(args.seed),
        "physical_fields_renormalized": False,
    }
    records = output_scalars.get("samples")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if "points" in record:
                record["original_points"] = record["points"]
            record["points"] = int(args.num_points)
    output_scalars["output_dir"] = str(output_dir)
    atomic_pickle_dump(output_scalars, output_dir / SCALARS_FILENAME)

    print(f"Completed: {len(paths)} samples")
    print("Coordinates and velocity were not normalized again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
