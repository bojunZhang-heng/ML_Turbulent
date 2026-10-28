#!/usr/bin/env python3
"""Downsample normalized point-cloud PKL samples with aligned point fields.

The script never renormalizes data.  One index array is generated per sample
and applied consistently to coordinates, normals, physical fields and any
other NumPy array whose first dimension is the number of points.  A segmentation
matrix shaped (S, N) is sampled along its node dimension as well.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


SCALARS_FILENAME = "normalization_scalars.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample normalized point-cloud PKL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/Users/zhangbojun/ML_Turbulent/"
            "preprocess/E_S_WW_WM_small/pklFile"
        ),
        help="Directory containing numbered sample PKL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, a sibling directory named "
            "<input>_<method>_<num-points> is created."
        ),
    )
    parser.add_argument(
        "--num-points",
        type=int,
        required=True,
        help="Target number of points retained in every sample.",
    )
    parser.add_argument(
        "--method",
        choices=("voxel", "random"),
        default="voxel",
        help="Geometry-aware voxel sampling or uniform random sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; each sample receives a deterministic child seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output sample files to be replaced.",
    )
    return parser.parse_args()


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def random_indices(
    num_input_points: int,
    num_output_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.sort(
        rng.choice(num_input_points, size=num_output_points, replace=False)
    ).astype(np.int64, copy=False)


def voxel_indices(
    coordinates: np.ndarray,
    num_output_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return exactly ``num_output_points`` spatially distributed indices.

    One representative is selected from each occupied voxel.  The resolution
    is increased until enough occupied voxels are available.  If the surface
    geometry occupies fewer voxels than requested, remaining points are filled
    uniformly without replacement.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        raise ValueError(
            f"'coor' must have shape (N, >=3), got {coordinates.shape}"
        )
    if not np.isfinite(coordinates[:, :3]).all():
        raise ValueError("'coor' contains NaN or infinite values")

    xyz = coordinates[:, :3]
    xyz_min = xyz.min(axis=0, keepdims=True)
    xyz_range = xyz.max(axis=0, keepdims=True) - xyz_min
    xyz_range[xyz_range == 0.0] = 1.0
    xyz_unit = np.clip((xyz - xyz_min) / xyz_range, 0.0, 1.0)

    # Surface point clouds usually need roughly sqrt(M) bins per axis.
    resolution = max(2, int(np.ceil(np.sqrt(num_output_points))))
    representatives = np.empty(0, dtype=np.int64)
    permutation = rng.permutation(len(xyz_unit))

    for _ in range(10):
        bins = np.minimum(
            (xyz_unit * resolution).astype(np.int64), resolution - 1
        )
        keys = (
            bins[:, 0]
            + resolution * bins[:, 1]
            + resolution * resolution * bins[:, 2]
        )
        _, first_positions = np.unique(
            keys[permutation], return_index=True
        )
        representatives = permutation[first_positions]
        if len(representatives) >= num_output_points:
            representatives = rng.choice(
                representatives, size=num_output_points, replace=False
            )
            return np.sort(representatives).astype(np.int64, copy=False)
        resolution = int(np.ceil(resolution * 1.5))

    # Degenerate or unusually thin geometries may still have too few occupied
    # voxels. Preserve all voxel representatives and fill from unused points.
    selected_mask = np.zeros(len(xyz_unit), dtype=bool)
    selected_mask[representatives] = True
    remaining = np.flatnonzero(~selected_mask)
    needed = num_output_points - len(representatives)
    supplement = rng.choice(remaining, size=needed, replace=False)
    return np.sort(np.concatenate((representatives, supplement))).astype(
        np.int64, copy=False
    )


def sample_value(
    key: str,
    value: Any,
    indices: np.ndarray,
    num_input_points: int,
) -> Any:
    """Sample arrays aligned with nodes while preserving all other metadata."""
    if not isinstance(value, np.ndarray):
        return value

    if value.ndim >= 1 and value.shape[0] == num_input_points:
        return value[indices]

    # seg_matrix convention: (number_of_segments, number_of_nodes)
    if key == "seg_matrix" and value.ndim == 2 and value.shape[1] == num_input_points:
        sampled = value[:, indices]
        # Preserve the original semantics when rows store normalized averaging
        # weights. Empty segments are retained but remain all-zero.
        row_sums = sampled.sum(axis=1, keepdims=True)
        nonzero = np.abs(row_sums[:, 0]) > 0
        sampled = sampled.copy()
        sampled[nonzero] = sampled[nonzero] / row_sums[nonzero]
        return sampled

    return value


def downsample_sample(
    sample: dict[str, Any],
    target_points: int,
    method: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if "coor" not in sample:
        raise KeyError("Sample does not contain the required 'coor' field")

    coordinates = np.asarray(sample["coor"])
    if coordinates.ndim != 2:
        raise ValueError(f"'coor' must be two-dimensional, got {coordinates.shape}")
    num_input_points = coordinates.shape[0]
    if target_points > num_input_points:
        raise ValueError(
            f"Requested {target_points} points, but sample has only "
            f"{num_input_points}. Sampling with replacement is intentionally disabled."
        )

    if target_points == num_input_points:
        indices = np.arange(num_input_points, dtype=np.int64)
    elif method == "random":
        indices = random_indices(num_input_points, target_points, rng)
    else:
        indices = voxel_indices(coordinates, target_points, rng)

    result = {
        key: sample_value(key, value, indices, num_input_points)
        for key, value in sample.items()
    }
    result["downsample_indices"] = indices
    result["downsampling"] = {
        "method": method,
        "original_num_points": int(num_input_points),
        "sampled_num_points": int(len(indices)),
    }

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata = metadata.copy()
        metadata["downsampling"] = result["downsampling"].copy()
        result["metadata"] = metadata

    return result


def atomic_pickle_dump(data: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    if args.num_points <= 0:
        raise ValueError("--num-points must be a positive integer")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir.parent / (
            f"{input_dir.name}_{args.method}_{args.num_points}"
        )
    output_dir = output_dir.expanduser().resolve()
    if output_dir == input_dir:
        raise ValueError("Output directory must differ from input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        (
            path
            for path in input_dir.glob("*.pkl")
            if path.name != SCALARS_FILENAME
        ),
        key=numeric_sort_key,
    )
    if not sample_files:
        raise FileNotFoundError(f"No sample PKL files found in {input_dir}")

    print("=" * 72)
    print("Point-cloud PKL downsampling")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target points   : {args.num_points}")
    print(f"Method          : {args.method}")
    print(f"Samples         : {len(sample_files)}")
    print("=" * 72)

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(sample_files))

    for position, (source, child_seed) in enumerate(
        zip(sample_files, child_seeds), start=1
    ):
        destination = output_dir / source.name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Use --overwrite to replace it."
            )

        with source.open("rb") as stream:
            sample = pickle.load(stream)
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a dictionary in {source}, got {type(sample)}")

        result = downsample_sample(
            sample,
            target_points=args.num_points,
            method=args.method,
            rng=np.random.default_rng(child_seed),
        )
        atomic_pickle_dump(result, destination)
        original_points = result["downsampling"]["original_num_points"]
        print(
            f"[{position:>4}/{len(sample_files)}] {source.name}: "
            f"{original_points} -> {len(result['coor'])} points"
        )

    scalars_source = input_dir / SCALARS_FILENAME
    if scalars_source.exists():
        scalars_destination = output_dir / SCALARS_FILENAME
        if scalars_destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {scalars_destination}. "
                "Use --overwrite to replace it."
            )
        shutil.copy2(scalars_source, scalars_destination)
        print(f"Copied unchanged: {SCALARS_FILENAME}")
    else:
        print(f"Warning: {SCALARS_FILENAME} was not found in the input directory")

    print("Downsampling complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
