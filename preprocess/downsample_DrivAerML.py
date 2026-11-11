#!/usr/bin/env python3
"""Downsample normalized PKLs produced by process_DrivAerML.py.

Surface and volume arrays remain in one PKL but use independent indices and
may use different target counts. Existing normalization is preserved exactly.
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
SURFACE_FIELDS = ("x_surface", "y_spressure", "y_wss")
VOLUME_FIELDS = ("x_volume", "y_velocity", "volume_cell_totalpcoeff")
GROUPS = {
    "surface": {
        "coordinate": "x_surface",
        "fields": SURFACE_FIELDS,
        "scalar_prefix": "surface_coords",
    },
    "volume": {
        "coordinate": "x_volume",
        "fields": VOLUME_FIELDS,
        "scalar_prefix": "volume_coords",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample normalized PKL point fields without renormalizing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"/Users/zhangbojun/ML_Turbulent/preprocess/drivaerML_N100_000_v2"),
        help="Directory containing sample PKL files and normalization_scalars.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; generated beside input when omitted.",
    )
    parser.add_argument(
        "--num-points", type=int, default=None,
        help="Fallback point count applied to every selected point set.",
    )
    parser.add_argument(
        "--surface-num-points", type=int, default=None,
        help="Number of surface points retained per sample.",
    )
    parser.add_argument(
        "--volume-num-points", type=int, default=None,
        help="Number of volume points retained per sample.",
    )
    parser.add_argument(
        "--point-set",
        choices=("auto", "surface", "volume", "both"),
        default="auto",
        help=(
            "DrivAerML point set to retain. 'surface' drops volume arrays; "
            "'volume' drops surface arrays; 'both' requires both; 'auto' "
            "retains every point set available in each complete or partial sample."
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
    args = parser.parse_args()
    for name in ("num_points", "surface_num_points", "volume_num_points"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def resolve_targets(args: argparse.Namespace) -> dict[str, int]:
    selected = ("surface", "volume") if args.point_set in {"auto", "both"} else (args.point_set,)
    targets: dict[str, int] = {}
    for group in selected:
        specific = getattr(args, f"{group}_num_points")
        target = specific if specific is not None else args.num_points
        if target is None:
            raise ValueError(
                f"No count supplied for {group}; use --{group}-num-points or --num-points"
            )
        targets[group] = int(target)
    return targets


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
        return "drivaerml"
    raise KeyError(
        "Cannot identify DrivAerML PKL schema: expected x_surface or x_volume"
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


def downsample_drivaerml(
    sample: dict[str, Any],
    targets: dict[str, int],
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
        raise KeyError(f"Selected DrivAerML point set(s) missing: {missing}")

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
            result, coordinate_key, fields, targets[group], method, rng
        )
        result[f"{group}_downsample_indices"] = indices
        group_info[group] = info

    result["downsampling"] = {
        "schema": "drivaerml",
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


def downsample_sample(
    sample: dict[str, Any],
    targets: dict[str, int],
    method: str,
    point_set: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    detect_format(sample)
    return downsample_drivaerml(sample, targets, method, point_set, rng)


def uses_surface_scale(scalars: dict[str, Any]) -> bool:
    return (
        scalars.get("coordinate_scale_policy")
        == "surface_scale_for_surface_and_volume"
        or bool(scalars.get("volume_coordinates_may_exceed_unit_interval"))
        or scalars.get("normalization", {}).get("x_volume")
        == "global_surface_minmax"
    )


def verify_coordinate_scalars(
    scalars: dict[str, Any], groups: tuple[str, ...]
) -> None:
    """Validate either separate scales or one shared surface scale."""

    for group in groups:
        prefix = GROUPS[group]["scalar_prefix"]
        keys = (f"{prefix}_min", f"{prefix}_max", f"{prefix}_range")
        missing = [key for key in keys if key not in scalars]
        if missing:
            raise RuntimeError(
                f"Missing {group} coordinate statistics {missing}. "
                "Migrate the input with process_DrivAerML.py --source-pkl-dir first."
            )
        minimum, maximum, values_range = (
            np.asarray(scalars[key], dtype=np.float32) for key in keys
        )
        if any(values.shape != (3,) for values in (minimum, maximum, values_range)):
            raise RuntimeError(f"Invalid {group} coordinate statistic shapes")
        expected_range = maximum - minimum
        expected_range[expected_range <= np.finfo(np.float32).eps] = 1.0
        if np.any(values_range <= 0) or not np.allclose(
            expected_range, values_range, rtol=1.0e-5, atol=1.0e-6
        ):
            raise RuntimeError(f"Invalid {group} min-max coordinate statistics")

    if uses_surface_scale(scalars) and {"surface", "volume"}.issubset(groups):
        for suffix in ("min", "max", "range"):
            surface = np.asarray(scalars[f"surface_coords_{suffix}"])
            volume = np.asarray(scalars[f"volume_coords_{suffix}"])
            if not np.array_equal(surface, volume):
                raise RuntimeError(
                    "Surface-scale policy requires identical surface and "
                    f"volume coordinate statistics; mismatch in {suffix}"
                )


def verify_downsampled_sample(
    sample: dict[str, Any], source: Path, scalars: dict[str, Any]
) -> None:
    """Verify alignment and index counts after writing one sample."""

    info = sample.get("downsampling", {}).get("groups", {})
    if not info:
        raise RuntimeError(f"Missing downsampling metadata in {source}")
    for group, group_info in info.items():
        target = int(group_info["sampled_num_points"])
        spec = GROUPS[group]
        indices = sample.get(f"{group}_downsample_indices")
        if not isinstance(indices, np.ndarray) or indices.shape != (target,):
            raise RuntimeError(f"Invalid {group} indices in {source}")
        for field in spec["fields"]:
            if field not in sample:
                continue
            values = sample[field]
            if len(values) != target:
                raise RuntimeError(f"Misaligned {field} in {source}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"NaN/Inf in {field} of {source}")
        coordinates = np.asarray(sample[spec["coordinate"]])
        require_unit_interval = group == "surface" or not uses_surface_scale(scalars)
        if require_unit_interval and (
            np.any(coordinates < -1.0e-5)
            or np.any(coordinates > 1.0 + 1.0e-5)
        ):
            raise RuntimeError(
                f"Coordinates outside [0,1] in {source}: {spec['coordinate']}"
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
    return ", ".join(
        f"{group}: {values['original_num_points']} -> "
        f"{values['sampled_num_points']}"
        for group, values in info["groups"].items()
    )


def main() -> int:
    args = parse_args()
    targets = resolve_targets(args)

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    scalars_source = input_dir / SCALARS_FILENAME
    if not scalars_source.is_file():
        raise FileNotFoundError(f"Missing {scalars_source}")
    with scalars_source.open("rb") as stream:
        output_scalars = copy.deepcopy(pickle.load(stream))
    verify_coordinate_scalars(output_scalars, tuple(targets))

    output_dir = args.output_dir
    if output_dir is None:
        suffix = args.point_set if args.point_set != "auto" else "auto"
        count_suffix = "_".join(f"{group}{count}" for group, count in targets.items())
        output_dir = input_dir.parent / (
            f"{input_dir.name}_{suffix}_{args.method}_{count_suffix}"
        )
    output_dir = output_dir.expanduser().resolve()
    if output_dir == input_dir:
        raise ValueError("Output directory must differ from input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        (
            path for path in input_dir.glob("*.pkl")
            if path.name != SCALARS_FILENAME and path.stem.isdigit()
        ),
        key=numeric_sort_key,
    )
    if not sample_files:
        raise FileNotFoundError(f"No sample PKL files found in {input_dir}")

    planned_outputs = [output_dir / path.name for path in sample_files]
    planned_outputs.append(output_dir / SCALARS_FILENAME)
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {existing_outputs[0]}; pass --overwrite to replace it"
        )
    if args.overwrite:
        for path in planned_outputs:
            if path.exists():
                path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()

    print("=" * 72)
    print("PKL point-cloud downsampling")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Point set       : {args.point_set}")
    print(f"Target points   : {targets}")
    print(f"Method          : {args.method}")
    print(f"Samples         : {len(sample_files)}")
    print(
        "Coordinate policy: "
        f"{output_scalars.get('coordinate_scale_policy', 'separate_scales')}"
    )
    print("=" * 72)

    child_seeds = np.random.SeedSequence(args.seed).spawn(len(sample_files))
    for position, (source, child_seed) in enumerate(
        zip(sample_files, child_seeds), start=1
    ):
        destination = output_dir / source.name
        with source.open("rb") as stream:
            sample = pickle.load(stream)
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dictionary in {source}, got {type(sample)}")
        result = downsample_sample(
            sample,
            targets=targets,
            method=args.method,
            point_set=args.point_set,
            rng=np.random.default_rng(child_seed),
        )
        if uses_surface_scale(output_scalars):
            result["schema_version"] = "drivaerml-normalized-surface-scale-v3"
        atomic_pickle_dump(result, destination)
        verify_downsampled_sample(result, destination, output_scalars)
        print(
            f"[{position:>4}/{len(sample_files)}] {source.name}: "
            f"{result_summary(result)}"
        )

    output_scalars["downsampling"] = {
        "source_directory": str(input_dir),
        "point_set": args.point_set,
        "num_points": dict(targets),
        "method": args.method,
        "seed": args.seed,
    }
    if uses_surface_scale(output_scalars):
        output_scalars["schema_version"] = "drivaerml-normalized-surface-scale-v3"
    manifest = output_scalars.get("samples")
    if isinstance(manifest, list):
        for record in manifest:
            if not isinstance(record, dict):
                continue
            for group in GROUPS:
                count_key = f"{group}_points"
                if count_key in record and record[count_key] > 0:
                    record[f"original_{count_key}"] = record[count_key]
                    record[count_key] = targets.get(group, 0)
    destination = output_dir / SCALARS_FILENAME
    atomic_pickle_dump(output_scalars, destination)
    print(f"Updated: {SCALARS_FILENAME}")

    print("Downsampling complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
