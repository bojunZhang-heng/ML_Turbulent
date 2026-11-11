#!/usr/bin/env python3
"""Offline downsampling dedicated to DrivAerStar PKLs.

Both legacy v1 PKLs and corrected v2 PKLs are supported.  For v1, the script
reverses the old shared wind-tunnel min-max transform, computes independent
statistics for the requested point set, and writes corrected v2 coordinates
while downsampling. Physical fields are never normalized again. One index
array is used for all aligned fields and saved for topology-based post-processing.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


SCALARS_FILENAME = "normalization_scalars.pkl"
GROUPS = {
    "surface": {
        "coordinate": "x_surface",
        "fields": ("x_surface", "y_spressure", "y_wss"),
        "required_scalars": ("surface_coords_min", "surface_coords_range"),
    },
    "volume": {
        "coordinate": "x_volume",
        "fields": ("x_volume", "y_velocity"),
        "required_scalars": ("volume_coords_min", "volume_coords_range"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Legacy v1 or corrected v2 DrivAerStar PKL directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help=(
            "Backward-compatible point count applied to every selected point "
            "set. For --point-set both, prefer --surface-num-points and "
            "--volume-num-points."
        ),
    )
    parser.add_argument(
        "--surface-num-points",
        type=int,
        default=None,
        help="Number of surface points retained in each output PKL.",
    )
    parser.add_argument(
        "--volume-num-points",
        type=int,
        default=None,
        help="Number of volume points retained in each output PKL.",
    )
    parser.add_argument(
        "--point-set",
        choices=("surface", "volume", "both"),
        default="surface",
    )
    parser.add_argument(
        "--method",
        choices=("voxel", "random"),
        default="voxel",
        help="Voxel is recommended for surface fields; random uses less memory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("num_points", "surface_num_points", "volume_num_points"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def resolve_target_counts(args: argparse.Namespace) -> dict[str, int]:
    """Resolve one target count per selected point set.

    ``--num-points`` remains supported for old commands. Group-specific values
    take precedence, allowing surface and volume arrays to remain together in
    one PKL without forcing the same sample count.
    """

    groups = ("surface", "volume") if args.point_set == "both" else (args.point_set,)
    targets: dict[str, int] = {}
    for group in groups:
        specific = getattr(args, f"{group}_num_points")
        target = specific if specific is not None else args.num_points
        if target is None:
            option = f"--{group}-num-points"
            raise ValueError(
                f"No point count supplied for {group}; use {option} or --num-points"
            )
        targets[group] = int(target)
    return targets


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
    """Return exactly ``target`` spatially distributed point indices."""

    xyz = np.asarray(coordinates, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape (N,3), got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("Coordinates contain NaN/Inf")
    xyz_min = xyz.min(axis=0, keepdims=True)
    xyz_range = xyz.max(axis=0, keepdims=True) - xyz_min
    xyz_range[xyz_range <= np.finfo(np.float32).eps] = 1.0
    unit = np.clip((xyz - xyz_min) / xyz_range, 0.0, 1.0)
    permutation = rng.permutation(len(unit))
    resolution = max(2, int(np.ceil(np.sqrt(target))))
    representatives = np.empty(0, dtype=np.int64)
    for _ in range(12):
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
    needed = target - len(representatives)
    supplement = rng.choice(remaining, size=needed, replace=False)
    return np.sort(np.concatenate((representatives, supplement))).astype(
        np.int64, copy=False
    )


def select_indices(
    coordinates: np.ndarray,
    target: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(coordinates)
    if count < target:
        raise ValueError(
            f"Point set contains {count} points, fewer than requested {target}"
        )
    if count == target:
        return np.arange(count, dtype=np.int64)
    if method == "random":
        return random_indices(count, target, rng)
    return voxel_indices(coordinates, target, rng)


def coordinate_stat_keys(group: str) -> tuple[str, str, str]:
    prefix = "surface_coords" if group == "surface" else "volume_coords"
    return f"{prefix}_min", f"{prefix}_max", f"{prefix}_range"


def source_coordinate_scalars(
    scalars: dict[str, Any], group: str
) -> tuple[np.ndarray, np.ndarray]:
    min_key, _, range_key = coordinate_stat_keys(group)
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
    raise KeyError(f"No source coordinate statistics for {group}")


def prepare_output_scalars(
    source_scalars: dict[str, Any],
    paths: list[Path],
    groups: tuple[str, ...],
) -> tuple[dict[str, Any], bool]:
    """Return v2 scalars, recovering requested physical bounds when needed."""

    already_v2 = all(
        all(key in source_scalars for key in GROUPS[group]["required_scalars"])
        for group in groups
    )
    output_scalars = copy.deepcopy(source_scalars)
    if already_v2:
        return output_scalars, False

    if "coords_min" not in source_scalars or "coords_range" not in source_scalars:
        raise KeyError(
            "Legacy migration requires coords_min and coords_range in "
            "normalization_scalars.pkl"
        )
    bounds = {
        group: {
            "min": np.full(3, np.inf, dtype=np.float64),
            "max": np.full(3, -np.inf, dtype=np.float64),
        }
        for group in groups
    }
    print("Legacy v1 coordinates detected; recovering point-set statistics...")
    for position, path in enumerate(paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        for group in groups:
            coordinate_key = GROUPS[group]["coordinate"]
            if coordinate_key not in sample:
                raise KeyError(f"{path}: missing {coordinate_key!r}")
            old_min, old_range = source_coordinate_scalars(source_scalars, group)
            physical = (
                np.asarray(sample[coordinate_key], dtype=np.float32) * old_range
                + old_min
            )
            bounds[group]["min"] = np.minimum(
                bounds[group]["min"], physical.min(axis=0)
            )
            bounds[group]["max"] = np.maximum(
                bounds[group]["max"], physical.max(axis=0)
            )
        print(f"[coordinate statistics {position}/{len(paths)}] {path.name}")
        del sample, physical

    output_scalars.pop("coords_min", None)
    output_scalars.pop("coords_max", None)
    output_scalars.pop("coords_range", None)
    output_scalars["migrated_from_schema"] = source_scalars.get("schema_version")
    output_scalars["schema_version"] = "drivaerstar-normalized-v2"
    normalization = dict(output_scalars.get("normalization", {}))
    for group in groups:
        coordinate_key = GROUPS[group]["coordinate"]
        values_range = bounds[group]["max"] - bounds[group]["min"]
        values_range[values_range <= np.finfo(np.float32).eps] = 1.0
        min_key, max_key, range_key = coordinate_stat_keys(group)
        output_scalars[min_key] = bounds[group]["min"].astype(np.float32)
        output_scalars[max_key] = bounds[group]["max"].astype(np.float32)
        output_scalars[range_key] = values_range.astype(np.float32)
        normalization[coordinate_key] = f"global_{group}_minmax"
    output_scalars["normalization"] = normalization
    scopes = dict(output_scalars.get("coordinate_normalization_scopes", {}))
    for group in groups:
        coordinate_key = GROUPS[group]["coordinate"]
        scopes[coordinate_key] = {
            "statistics_prefix": f"{group}_coords",
            "population": f"all {group} coordinates reconstructed from source PKLs",
        }
    output_scalars["coordinate_normalization_scopes"] = scopes
    return output_scalars, True


def downsample_sample(
    sample: dict[str, Any],
    source_scalars: dict[str, Any],
    output_scalars: dict[str, Any],
    migrate_coordinates: bool,
    groups: tuple[str, ...],
    targets: dict[str, int],
    method: str,
    seed: int,
    sample_id: int,
) -> dict[str, Any]:
    result = copy.copy(sample)
    # Discard unused multi-million-point arrays before writing the result.
    for group, spec in GROUPS.items():
        if group not in groups:
            for key in spec["fields"]:
                result.pop(key, None)

    group_records: dict[str, Any] = {}
    for group_position, group in enumerate(groups):
        spec = GROUPS[group]
        coordinate_key = spec["coordinate"]
        if coordinate_key not in result:
            raise KeyError(f"Sample {sample_id} is missing {coordinate_key!r}")
        coordinates = np.asarray(result[coordinate_key], dtype=np.float32)
        if migrate_coordinates:
            old_min, old_range = source_coordinate_scalars(source_scalars, group)
            physical = coordinates * old_range + old_min
            min_key, _, range_key = coordinate_stat_keys(group)
            coordinates = (
                (physical - output_scalars[min_key])
                / output_scalars[range_key]
            ).astype(np.float32, copy=False)
            result[coordinate_key] = coordinates
        original_count = len(coordinates)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, sample_id, group_position])
        )
        target = targets[group]
        indices = select_indices(coordinates, target, method, rng)
        for key in spec["fields"]:
            if key not in result:
                continue
            values = result[key]
            if not isinstance(values, np.ndarray) or len(values) != original_count:
                raise ValueError(
                    f"Sample {sample_id}: {key} is not aligned with {coordinate_key}"
                )
            result[key] = values[indices]
        result[f"{group}_downsample_indices"] = indices
        group_records[group] = {
            "coordinate_field": coordinate_key,
            "original_num_points": int(original_count),
            "sampled_num_points": int(len(indices)),
            "method": method,
        }

    result["downsampling"] = {
        "schema": "drivaerstar-v2",
        "point_sets": list(groups),
        "method": method,
        "seed": seed,
        "groups": group_records,
    }
    if migrate_coordinates:
        result["schema_version"] = "drivaerstar-normalized-v2"
        sample_normalization = dict(result.get("normalization", {}))
        for group in groups:
            sample_normalization[GROUPS[group]["coordinate"]] = (
                f"global_{group}_minmax"
            )
        result["normalization"] = sample_normalization
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata = metadata.copy()
        metadata["downsampling"] = copy.deepcopy(result["downsampling"])
        metadata["topology_note"] = (
            "Part ranges refer to original arrays. Map predictions back with "
            "the saved <group>_downsample_indices array."
        )
        result["metadata"] = metadata
    return result


def verify_output(
    sample: dict[str, Any],
    groups: tuple[str, ...],
    targets: dict[str, int],
    path: Path,
    scalars: dict[str, Any],
) -> None:
    surface_scale = (
        scalars.get("coordinate_scale_policy")
        == "surface_scale_for_surface_and_volume"
        or bool(scalars.get("volume_coordinates_may_exceed_unit_interval"))
        or scalars.get("normalization", {}).get("x_volume")
        == "global_surface_minmax"
    )
    for group in groups:
        spec = GROUPS[group]
        target = targets[group]
        for key in spec["fields"]:
            if key not in sample:
                continue
            values = sample[key]
            if not isinstance(values, np.ndarray) or len(values) != target:
                raise RuntimeError(f"Invalid {key} in {path}: {getattr(values, 'shape', None)}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"NaN/Inf in {key} of {path}")
        coords = sample[spec["coordinate"]]
        # With a shared surface scale, the larger wind-tunnel volume is
        # expected to extend outside [0, 1]. Surface coordinates still use the
        # defining min-max range and must remain inside the unit interval.
        require_unit_interval = group == "surface" or not surface_scale
        if require_unit_interval and (
            np.any(coords < -1.0e-5) or np.any(coords > 1.0 + 1.0e-5)
        ):
            raise RuntimeError(f"Coordinates outside [0,1] in {path}")


def verify_separate_coordinate_scalars(
    scalars: dict[str, Any], groups: tuple[str, ...]
) -> None:
    """Require a dedicated min/max/range triplet for every retained point set."""

    for group in groups:
        missing = [
            key for key in coordinate_stat_keys(group) if key not in scalars
        ]
        if missing:
            raise RuntimeError(
                f"Missing independent {group} coordinate statistics: {missing}"
            )
        min_key, max_key, range_key = coordinate_stat_keys(group)
        minimum = np.asarray(scalars[min_key], dtype=np.float32)
        maximum = np.asarray(scalars[max_key], dtype=np.float32)
        values_range = np.asarray(scalars[range_key], dtype=np.float32)
        if minimum.shape != (3,) or maximum.shape != (3,) or values_range.shape != (3,):
            raise RuntimeError(f"Invalid {group} coordinate statistic shapes")
        expected_range = maximum - minimum
        expected_range[expected_range <= np.finfo(np.float32).eps] = 1.0
        if np.any(values_range <= 0) or not np.allclose(
            expected_range, values_range, rtol=1.0e-5, atol=1.0e-6
        ):
            raise RuntimeError(f"Invalid {group} min-max coordinate statistics")


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
        scalars = pickle.load(stream)
    groups = ("surface", "volume") if args.point_set == "both" else (args.point_set,)
    targets = resolve_target_counts(args)
    paths = numbered_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No numbered PKLs found in {input_dir}")
    output_scalars, migrate_coordinates = prepare_output_scalars(
        scalars, paths, groups
    )
    verify_separate_coordinate_scalars(output_scalars, groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [output_dir / path.name for path in paths]
    planned.append(output_dir / SCALARS_FILENAME)
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}; use --overwrite")
    if args.overwrite:
        for path in numbered_paths(output_dir):
            path.unlink()
        output_scalar_path = output_dir / SCALARS_FILENAME
        if output_scalar_path.exists():
            output_scalar_path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()

    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Samples     : {len(paths)}")
    print(f"Point sets  : {groups}")
    print(f"Point counts: {targets}")
    print(f"Method      : {args.method}")
    print(
        "Coord policy: "
        f"{output_scalars.get('coordinate_scale_policy', 'separate_scales')}"
    )
    for position, path in enumerate(paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        sample_id = int(path.stem)
        result = downsample_sample(
            sample,
            scalars,
            output_scalars,
            migrate_coordinates,
            groups,
            targets,
            args.method,
            args.seed,
            sample_id,
        )
        destination = output_dir / path.name
        atomic_pickle_dump(result, destination)
        verify_output(result, groups, targets, destination, output_scalars)
        print(f"[{position}/{len(paths)}] {path.name}")
        del sample, result

    output_scalars["downsampling"] = {
        "source_directory": str(input_dir),
        "point_sets": list(groups),
        "num_points": dict(targets),
        "method": args.method,
        "seed": args.seed,
    }
    sample_manifest = output_scalars.get("samples")
    if isinstance(sample_manifest, list):
        for record in sample_manifest:
            if not isinstance(record, dict):
                continue
            for group in groups:
                count_key = f"{group}_cells"
                if count_key in record:
                    record[f"original_{count_key}"] = record[count_key]
                record[count_key] = targets[group]
    output_scalars["output_dir"] = str(output_dir)
    atomic_pickle_dump(output_scalars, output_dir / SCALARS_FILENAME)
    print(f"Completed: {len(paths)} samples")
    if migrate_coordinates:
        print("Legacy coordinates were recovered and separately normalized as v2.")
    print("Physical fields were not normalized again during downsampling.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}")
        raise
