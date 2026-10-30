#!/usr/bin/env python3
"""Offline downsampling for legacy point-cloud and DrivAerStar PKL datasets.

No normalization is recomputed. DrivAerStar surface and volume fields use
independent indices because they live on different point sets.
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
SURFACE_FIELDS = ("x_surface", "y_spressure", "y_wss")
VOLUME_FIELDS = ("x_volume", "y_velocity", "volume_cell_totalpcoeff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample normalized PKL point fields without renormalizing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar"),
        help="Directory containing sample PKL files and normalization_scalars.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; generated beside input when omitted.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        required=True,
        help="Points retained in each selected point set.",
    )
    parser.add_argument(
        "--point-set",
        choices=("auto", "surface", "volume", "both"),
        default="auto",
        help=(
            "DrivAerStar point set to retain. 'surface' drops volume arrays; "
            "'volume' drops surface arrays; 'both' samples both independently; "
            "'auto' selects both for DrivAerStar and the only set for legacy PKL."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("voxel", "random"),
        default="random",
        help=(
            "Sampling method. Random is recommended for multi-million-cell "
            "volume fields because voxel sampling needs more temporary memory."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; every sample receives a deterministic child seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output PKL files.",
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
    """Select exactly the requested number of spatially distributed points."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        raise ValueError(f"Coordinates must have shape (N, >=3), got {coordinates.shape}")
    xyz = np.asarray(coordinates[:, :3], dtype=np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinates contain NaN or infinite values")

    xyz_min = xyz.min(axis=0, keepdims=True)
    xyz_range = xyz.max(axis=0, keepdims=True) - xyz_min
    xyz_range[xyz_range == 0.0] = 1.0
    xyz_unit = np.clip((xyz - xyz_min) / xyz_range, 0.0, 1.0)

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
        _, first_positions = np.unique(keys[permutation], return_index=True)
        representatives = permutation[first_positions]
        if len(representatives) >= num_output_points:
            chosen = rng.choice(
                representatives, size=num_output_points, replace=False
            )
            return np.sort(chosen).astype(np.int64, copy=False)
        resolution = int(np.ceil(resolution * 1.5))

    selected = np.zeros(len(xyz_unit), dtype=bool)
    selected[representatives] = True
    remaining = np.flatnonzero(~selected)
    needed = num_output_points - len(representatives)
    supplement = rng.choice(remaining, size=needed, replace=False)
    return np.sort(np.concatenate((representatives, supplement))).astype(
        np.int64, copy=False
    )


def make_indices(
    coordinates: np.ndarray,
    target_points: int,
    method: str,
    rng: np.random.Generator,
    field_name: str,
) -> np.ndarray:
    num_input_points = len(coordinates)
    if target_points > num_input_points:
        raise ValueError(
            f"{field_name} has {num_input_points} points, fewer than requested "
            f"{target_points}; sampling with replacement is disabled"
        )
    if target_points == num_input_points:
        return np.arange(num_input_points, dtype=np.int64)
    if method == "random":
        return random_indices(num_input_points, target_points, rng)
    return voxel_indices(coordinates, target_points, rng)


def detect_format(sample: dict[str, Any]) -> str:
    has_surface = "x_surface" in sample
    has_volume = "x_volume" in sample
    if has_surface or has_volume:
        return "drivaerstar"
    if "coor" in sample:
        return "legacy"
    raise KeyError(
        "Cannot identify PKL schema: expected 'coor', 'x_surface', or 'x_volume'"
    )


def sample_known_group(
    result: dict[str, Any],
    coordinate_key: str,
    field_keys: tuple[str, ...],
    target_points: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if coordinate_key not in result:
        raise KeyError(f"Selected point set requires {coordinate_key!r}")
    coordinates = np.asarray(result[coordinate_key])
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"{coordinate_key} must have shape (N, 3), got {coordinates.shape}")
    original_points = coordinates.shape[0]
    indices = make_indices(
        coordinates, target_points, method, rng, coordinate_key
    )

    for key in field_keys:
        if key not in result:
            continue
        value = result[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key} must be a NumPy array, got {type(value)}")
        if value.ndim < 1 or value.shape[0] != original_points:
            raise ValueError(
                f"{key} is not aligned with {coordinate_key}: "
                f"{value.shape} versus {original_points} points"
            )
        result[key] = value[indices]

    info = {
        "method": method,
        "coordinate_field": coordinate_key,
        "original_num_points": int(original_points),
        "sampled_num_points": int(len(indices)),
    }
    return indices, info


def downsample_drivaerstar(
    sample: dict[str, Any],
    target_points: int,
    method: str,
    point_set: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    available = {
        "surface": "x_surface" in sample,
        "volume": "x_volume" in sample,
    }
    selected = (
        [name for name, exists in available.items() if exists]
        if point_set == "auto"
        else ["surface", "volume"] if point_set == "both"
        else [point_set]
    )
    missing = [name for name in selected if not available[name]]
    if missing:
        raise KeyError(f"Selected DrivAerStar point set(s) missing: {missing}")

    result = sample.copy()
    # Removing the unused group is essential: otherwise a surface-only output
    # would still contain the original multi-million-cell volume arrays.
    for group, fields in (("surface", SURFACE_FIELDS), ("volume", VOLUME_FIELDS)):
        if group not in selected:
            for key in fields:
                result.pop(key, None)

    group_info: dict[str, Any] = {}
    for group in selected:
        coordinate_key = "x_surface" if group == "surface" else "x_volume"
        fields = SURFACE_FIELDS if group == "surface" else VOLUME_FIELDS
        indices, info = sample_known_group(
            result, coordinate_key, fields, target_points, method, rng
        )
        result[f"{group}_downsample_indices"] = indices
        group_info[group] = info

    result["downsampling"] = {
        "schema": "drivaerstar",
        "point_set": selected,
        "method": method,
        "groups": group_info,
    }
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata = metadata.copy()
        metadata["downsampling"] = result["downsampling"].copy()
        metadata["topology_note"] = (
            "Part start/end ranges describe the original arrays; use saved "
            "downsample indices to map predictions back to original cells."
        )
        result["metadata"] = metadata
    return result


def sample_legacy_value(
    key: str,
    value: Any,
    indices: np.ndarray,
    num_input_points: int,
) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == num_input_points:
            return value[indices]
        if key == "seg_matrix" and value.ndim == 2 and value.shape[1] == num_input_points:
            sampled = value[:, indices].copy()
            row_sums = sampled.sum(axis=1, keepdims=True)
            nonzero = np.abs(row_sums[:, 0]) > 0
            sampled[nonzero] /= row_sums[nonzero]
            return sampled
    return value


def downsample_legacy(
    sample: dict[str, Any],
    target_points: int,
    method: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    coordinates = np.asarray(sample["coor"])
    indices = make_indices(coordinates, target_points, method, rng, "coor")
    result = {
        key: sample_legacy_value(key, value, indices, len(coordinates))
        for key, value in sample.items()
    }
    result["downsample_indices"] = indices
    result["downsampling"] = {
        "schema": "legacy",
        "method": method,
        "original_num_points": int(len(coordinates)),
        "sampled_num_points": int(len(indices)),
    }
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata = metadata.copy()
        metadata["downsampling"] = result["downsampling"].copy()
        result["metadata"] = metadata
    return result


def downsample_sample(
    sample: dict[str, Any],
    target_points: int,
    method: str,
    point_set: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    schema = detect_format(sample)
    if schema == "legacy":
        if point_set not in ("auto", "both"):
            raise ValueError("--point-set surface/volume is only valid for DrivAerStar PKL")
        return downsample_legacy(sample, target_points, method, rng)
    return downsample_drivaerstar(
        sample, target_points, method, point_set, rng
    )


def atomic_pickle_dump(data: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def result_summary(result: dict[str, Any]) -> str:
    info = result["downsampling"]
    if info["schema"] == "legacy":
        return (
            f"{info['original_num_points']} -> "
            f"{info['sampled_num_points']} points"
        )
    return ", ".join(
        f"{group}: {values['original_num_points']} -> "
        f"{values['sampled_num_points']}"
        for group, values in info["groups"].items()
    )


def main() -> int:
    args = parse_args()
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        suffix = args.point_set if args.point_set != "auto" else "auto"
        output_dir = input_dir.parent / (
            f"{input_dir.name}_{suffix}_{args.method}_{args.num_points}"
        )
    output_dir = output_dir.expanduser().resolve()
    if output_dir == input_dir:
        raise ValueError("Output directory must differ from input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        (
            path for path in input_dir.glob("*.pkl")
            if path.name != SCALARS_FILENAME
        ),
        key=numeric_sort_key,
    )
    if not sample_files:
        raise FileNotFoundError(f"No sample PKL files found in {input_dir}")

    print("=" * 72)
    print("PKL point-cloud downsampling")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Point set       : {args.point_set}")
    print(f"Target points   : {args.num_points}")
    print(f"Method          : {args.method}")
    print(f"Samples         : {len(sample_files)}")
    print("=" * 72)

    child_seeds = np.random.SeedSequence(args.seed).spawn(len(sample_files))
    for position, (source, child_seed) in enumerate(
        zip(sample_files, child_seeds), start=1
    ):
        destination = output_dir / source.name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {destination}; pass --overwrite to replace it"
            )
        with source.open("rb") as stream:
            sample = pickle.load(stream)
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dictionary in {source}, got {type(sample)}")
        result = downsample_sample(
            sample,
            target_points=args.num_points,
            method=args.method,
            point_set=args.point_set,
            rng=np.random.default_rng(child_seed),
        )
        atomic_pickle_dump(result, destination)
        print(
            f"[{position:>4}/{len(sample_files)}] {source.name}: "
            f"{result_summary(result)}"
        )

    scalars_source = input_dir / SCALARS_FILENAME
    if scalars_source.is_file():
        destination = output_dir / SCALARS_FILENAME
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {destination}; pass --overwrite to replace it"
            )
        shutil.copy2(scalars_source, destination)
        print(f"Copied unchanged: {SCALARS_FILENAME}")
    else:
        print(f"Warning: {SCALARS_FILENAME} not found")

    print("Downsampling complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
