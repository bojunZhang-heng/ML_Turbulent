#!/usr/bin/env python3
"""Convert DrivAerStar data into normalized PKL samples.

The EnSight variables are cell fields (``per element``), so coordinates are
cell centers rather than mesh vertices. Automotive surface parts are selected
from two-dimensional parts carrying both Pressure and WallShearStress while
excluding wind-tunnel ``Block.*`` boundaries. Three-dimensional parts carrying
Velocity form the volume field.

Surface and volume coordinates share the dataset-wide automotive-surface
min-max statistics.  Volume coordinates may therefore extend beyond [0, 1].
The script can also migrate old ``drivaerstar-normalized-v1`` PKLs without the
deleted EnSight sources: it reverses the old shared min-max transform, computes
separate surface/volume statistics, and writes ``v2`` PKLs. Physical fields are
copied unchanged because they are already mean-std normalized.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


DEFAULT_DATA_DIR = Path(
    r"/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar"
)
SCALARS_FILENAME = "normalization_scalars.pkl"
REQUIRED_CASE_SUFFIXES = (".geo", ".Pressure", ".Velocity", ".WallShearStress")


@dataclass
class RunningMoments:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError(f"Expected non-empty (N,C), got {values.shape}")
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
            raise ValueError("Physical field component count changed between cases")
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
        description="Convert DrivAerStar EnSight cases to normalized PKL samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-pkl-dir",
        type=Path,
        default=None,
        help=(
            "Migrate already normalized DrivAerStar PKLs instead of reading "
            "EnSight files. The source normalization_scalars.pkl is required."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data-dir so numbered PKLs are placed at its root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


COORDINATE_FIELDS = ("x_surface", "x_volume")
COORDINATE_PREFIX = {field: "surface_coords" for field in COORDINATE_FIELDS}


def coordinate_scalar_keys(field: str) -> tuple[str, str, str]:
    try:
        prefix = COORDINATE_PREFIX[field]
    except KeyError as error:
        raise ValueError(f"Unknown coordinate field: {field!r}") from error
    return f"{prefix}_min", f"{prefix}_max", f"{prefix}_range"


def coordinate_scalars(
    scalars: dict[str, Any], field: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve new shared-surface or legacy coordinate statistics."""

    if (
        field == "x_volume"
        and scalars.get("coordinate_scale_policy")
        != "surface_scale_for_surface_and_volume"
        and "volume_coords_min" in scalars
        and "volume_coords_range" in scalars
    ):
        return (
            np.asarray(scalars["volume_coords_min"], dtype=np.float32),
            np.asarray(scalars["volume_coords_range"], dtype=np.float32),
        )
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
        f"{range_key!r} or legacy 'coords_min'/'coords_range'"
    )


def case_id(case_path: Path) -> int:
    match = re.fullmatch(r"[A-Za-z]+_(\d+)\.case", case_path.name)
    if match is None:
        raise ValueError(f"Cannot extract numeric case ID from {case_path.name}")
    return int(match.group(1))


def case_companions(case_path: Path) -> dict[str, Path]:
    stem = case_path.with_suffix("")
    return {suffix: Path(f"{stem}{suffix}") for suffix in REQUIRED_CASE_SUFFIXES}


def incomplete_case_record(case_path: Path) -> dict[str, Any] | None:
    companions = case_companions(case_path)
    missing = [suffix for suffix, path in companions.items() if not path.is_file()]
    if not missing:
        return None
    return {
        "sample_id": case_id(case_path),
        "source_case": str(case_path),
        "missing_files": tuple(missing),
    }


def block_name(dataset: vtk.vtkMultiBlockDataSet, index: int) -> str:
    metadata = dataset.GetMetaData(index)
    key = vtk.vtkCompositeDataSet.NAME()
    if metadata is not None and metadata.Has(key):
        return (metadata.Get(key) or f"block_{index}").strip()
    return f"block_{index}"


def read_case(case_path: Path) -> vtk.vtkMultiBlockDataSet:
    reader = vtk.vtkEnSightGoldBinaryReader()
    reader.SetCaseFileName(str(case_path.resolve()))
    reader.ReadAllVariablesOn()
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfBlocks() == 0:
        raise ValueError(f"Unable to read EnSight case: {case_path}")
    return output


def array_from_cells(
    block: vtk.vtkDataSet,
    name: str,
    components: int,
    source: Path,
) -> np.ndarray:
    vtk_array = block.GetCellData().GetArray(name)
    if vtk_array is None:
        raise KeyError(f"{source}: cell array {name!r} not found")
    values = vtk_to_numpy(vtk_array)
    if values.ndim == 1 and components == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape != (block.GetNumberOfCells(), components):
        raise ValueError(
            f"{source}: {name} shape {values.shape}, expected "
            f"({block.GetNumberOfCells()},{components})"
        )
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: {name} contains NaN or Inf")
    return values


def cell_centers(block: vtk.vtkDataSet, source: Path) -> np.ndarray:
    center_filter = vtk.vtkCellCenters()
    center_filter.SetInputData(block)
    center_filter.VertexCellsOff()
    center_filter.Update()
    points = center_filter.GetOutput().GetPoints()
    if points is None:
        raise ValueError(f"{source}: failed to compute cell centers")
    centers = vtk_to_numpy(points.GetData())
    centers = np.asarray(centers, dtype=np.float32)
    expected = (block.GetNumberOfCells(), 3)
    if centers.shape != expected:
        raise ValueError(f"{source}: cell-center shape {centers.shape}, expected {expected}")
    if not np.isfinite(centers).all():
        raise ValueError(f"{source}: cell centers contain NaN or Inf")
    return centers


def cell_dimension(block: vtk.vtkDataSet) -> int:
    if block.GetNumberOfCells() == 0:
        return -1
    dimensions = {
        vtk.vtkCellTypes.GetDimension(block.GetCellType(index))
        for index in range(block.GetNumberOfCells())
    }
    if len(dimensions) != 1:
        raise ValueError(f"Mixed cell dimensions in one EnSight part: {dimensions}")
    return dimensions.pop()


def concatenate(parts: list[np.ndarray], field: str, case_path: Path) -> np.ndarray:
    if not parts:
        raise ValueError(f"{case_path}: no arrays selected for {field}")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def extract_case(case_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = read_case(case_path)
    surface_coords: list[np.ndarray] = []
    surface_pressure: list[np.ndarray] = []
    surface_wss: list[np.ndarray] = []
    volume_coords: list[np.ndarray] = []
    volume_velocity: list[np.ndarray] = []
    surface_parts: list[dict[str, Any]] = []
    volume_parts: list[dict[str, Any]] = []
    surface_offset = 0
    volume_offset = 0

    for index in range(dataset.GetNumberOfBlocks()):
        block = dataset.GetBlock(index)
        if block is None or block.GetNumberOfCells() == 0:
            continue
        name = block_name(dataset, index)
        cell_data = block.GetCellData()
        arrays = {
            cell_data.GetArrayName(array_index)
            for array_index in range(cell_data.GetNumberOfArrays())
        }
        dimension = cell_dimension(block)

        is_surface = (
            dimension == 2
            and "Pressure" in arrays
            and "WallShearStress" in arrays
            and not name.startswith("Block.")
            and "[交界面" not in name
        )
        is_volume = dimension == 3 and "Velocity" in arrays

        if is_surface:
            centers = cell_centers(block, case_path)
            pressure = array_from_cells(block, "Pressure", 1, case_path)
            wss = array_from_cells(block, "WallShearStress", 3, case_path)
            if not (len(centers) == len(pressure) == len(wss)):
                raise ValueError(f"{case_path}: surface fields misaligned in {name}")
            surface_coords.append(centers)
            surface_pressure.append(pressure)
            surface_wss.append(wss)
            surface_parts.append(
                {
                    "block_index": index,
                    "name": name,
                    "start": surface_offset,
                    "end": surface_offset + len(centers),
                    "cells": len(centers),
                }
            )
            surface_offset += len(centers)

        if is_volume:
            centers = cell_centers(block, case_path)
            velocity = array_from_cells(block, "Velocity", 3, case_path)
            if len(centers) != len(velocity):
                raise ValueError(f"{case_path}: volume fields misaligned in {name}")
            volume_coords.append(centers)
            volume_velocity.append(velocity)
            volume_parts.append(
                {
                    "block_index": index,
                    "name": name,
                    "start": volume_offset,
                    "end": volume_offset + len(centers),
                    "cells": len(centers),
                }
            )
            volume_offset += len(centers)

    arrays = {
        "x_surface": concatenate(surface_coords, "x_surface", case_path),
        "y_spressure": concatenate(surface_pressure, "y_spressure", case_path),
        "y_wss": concatenate(surface_wss, "y_wss", case_path),
        "x_volume": concatenate(volume_coords, "x_volume", case_path),
        "y_velocity": concatenate(volume_velocity, "y_velocity", case_path),
    }
    if not (
        len(arrays["x_surface"])
        == len(arrays["y_spressure"])
        == len(arrays["y_wss"])
    ):
        raise RuntimeError(f"{case_path}: combined surface arrays are misaligned")
    if len(arrays["x_volume"]) != len(arrays["y_velocity"]):
        raise RuntimeError(f"{case_path}: combined volume arrays are misaligned")
    metadata = {
        "source_case": str(case_path),
        "surface_parts": surface_parts,
        "volume_parts": volume_parts,
        "surface_cells": len(arrays["x_surface"]),
        "volume_cells": len(arrays["x_volume"]),
        "coordinate_semantics": "cell_centers",
    }
    return arrays, metadata


def collect_statistics(
    case_paths: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    surface_bounds = {
        "min": np.full(3, np.inf, dtype=np.float64),
        "max": np.full(3, -np.inf, dtype=np.float64),
    }
    volume_domain_bounds = {
        "min": np.full(3, np.inf, dtype=np.float64),
        "max": np.full(3, -np.inf, dtype=np.float64),
    }
    moments = {
        "y_spressure": RunningMoments(),
        "y_velocity": RunningMoments(),
        "y_wss": RunningMoments(),
    }
    records: list[dict[str, Any]] = []
    for position, case_path in enumerate(case_paths, start=1):
        started = time.monotonic()
        arrays, metadata = extract_case(case_path)
        surface_bounds["min"] = np.minimum(
            surface_bounds["min"], arrays["x_surface"].min(axis=0)
        )
        surface_bounds["max"] = np.maximum(
            surface_bounds["max"], arrays["x_surface"].max(axis=0)
        )
        volume_domain_bounds["min"] = np.minimum(
            volume_domain_bounds["min"], arrays["x_volume"].min(axis=0)
        )
        volume_domain_bounds["max"] = np.maximum(
            volume_domain_bounds["max"], arrays["x_volume"].max(axis=0)
        )
        for field, accumulator in moments.items():
            accumulator.update(arrays[field])
        records.append(
            {
                "sample_id": case_id(case_path),
                "source_case": str(case_path),
                "surface_cells": metadata["surface_cells"],
                "volume_cells": metadata["volume_cells"],
                "surface_parts": metadata["surface_parts"],
                "volume_parts": metadata["volume_parts"],
            }
        )
        print(
            f"[statistics {position}/{len(case_paths)}] {case_path.name}: "
            f"surface={metadata['surface_cells']}, volume={metadata['volume_cells']}, "
            f"{time.monotonic() - started:.2f}s"
        )
        del arrays, metadata
        gc.collect()

    scalars: dict[str, Any] = {
        "schema_version": "drivaerstar-normalized-v3-surface-scale",
        "coordinate_fields": COORDINATE_FIELDS,
        "physical_fields": ("y_spressure", "y_velocity", "y_wss"),
        "field_mapping": {
            "Pressure": "y_spressure",
            "Velocity": "y_velocity",
            "WallShearStress": "y_wss",
            "surface_cell_centers": "x_surface",
            "volume_cell_centers": "x_volume",
        },
        "normalization": {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_surface_minmax",
            "y_spressure": "global_componentwise_mean_std",
            "y_velocity": "global_componentwise_mean_std",
            "y_wss": "global_componentwise_mean_std",
        },
        "coordinate_normalization_scopes": {
            "x_surface": {
                "statistics_prefix": "surface_coords",
                "population": "all selected automotive surface cell centers",
            },
            "x_volume": {
                "statistics_prefix": "surface_coords",
                "population": "all selected automotive surface cell centers",
            },
        },
        "coordinate_semantics": "cell_centers",
        "surface_selection": (
            "2D EnSight parts with Pressure and WallShearStress, excluding "
            "Block.* wind-tunnel boundaries and interface parts"
        ),
        "volume_selection": "3D EnSight parts with Velocity",
        "coordinate_scale_policy": "surface_scale_for_surface_and_volume",
        "volume_coordinates_may_exceed_unit_interval": True,
        "samples": records,
        "num_samples": len(records),
    }
    surface_range = surface_bounds["max"] - surface_bounds["min"]
    surface_range[surface_range <= np.finfo(np.float32).eps] = 1.0
    scalars["surface_coords_min"] = surface_bounds["min"].astype(np.float32)
    scalars["surface_coords_max"] = surface_bounds["max"].astype(np.float32)
    scalars["surface_coords_range"] = surface_range.astype(np.float32)
    # Compatibility aliases: downstream volume loaders use volume_coords_*.
    # These intentionally contain the same automotive-surface scale.
    scalars["volume_coords_min"] = scalars["surface_coords_min"].copy()
    scalars["volume_coords_max"] = scalars["surface_coords_max"].copy()
    scalars["volume_coords_range"] = scalars["surface_coords_range"].copy()
    scalars["volume_domain_min"] = volume_domain_bounds["min"].astype(np.float32)
    scalars["volume_domain_max"] = volume_domain_bounds["max"].astype(np.float32)
    for field, accumulator in moments.items():
        mean, std = accumulator.statistics()
        scalars[f"{field}_mean"] = mean
        scalars[f"{field}_std"] = std
        scalars[f"{field}_count"] = accumulator.count
    return scalars, records


def normalized_sample(
    case_path: Path,
    scalars: dict[str, Any],
) -> dict[str, Any]:
    arrays, metadata = extract_case(case_path)
    for field in COORDINATE_FIELDS:
        coords_min, coords_range = coordinate_scalars(scalars, field)
        arrays[field] = (
            (arrays[field] - coords_min) / coords_range
        ).astype(np.float32, copy=False)
    for field in ("y_spressure", "y_velocity", "y_wss"):
        arrays[field] = (
            (arrays[field] - scalars[f"{field}_mean"])
            / scalars[f"{field}_std"]
        ).astype(np.float32, copy=False)
    return {
        "schema_version": scalars["schema_version"],
        "sample_id": case_id(case_path),
        **arrays,
        "metadata": metadata,
        "normalization": scalars["normalization"].copy(),
    }


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_sample(path: Path, sample_id: int) -> None:
    with path.open("rb") as stream:
        sample = pickle.load(stream)
    if sample.get("sample_id") != sample_id:
        raise RuntimeError(f"Incorrect sample ID in {path}")
    expected = {
        "x_surface": 3,
        "y_spressure": 1,
        "x_volume": 3,
        "y_velocity": 3,
        "y_wss": 3,
    }
    for field, components in expected.items():
        values = sample.get(field)
        if not isinstance(values, np.ndarray) or values.ndim != 2:
            raise RuntimeError(f"Missing or invalid {field} in {path}")
        if values.shape[1] != components or values.dtype != np.float32:
            raise RuntimeError(f"Invalid shape/dtype for {field} in {path}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"NaN/Inf in {field} of {path}")
        if field == "x_surface" and (
            np.any(values < -1.0e-5) or np.any(values > 1.0 + 1.0e-5)
        ):
            raise RuntimeError(f"Normalized {field} is outside [0,1] in {path}")
    if not (
        len(sample["x_surface"])
        == len(sample["y_spressure"])
        == len(sample["y_wss"])
    ):
        raise RuntimeError(f"Surface fields are misaligned in {path}")
    if len(sample["x_volume"]) != len(sample["y_velocity"]):
        raise RuntimeError(f"Volume fields are misaligned in {path}")


def numbered_pkl_paths(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def migrate_processed_pkls(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> int:
    """Recover physical coordinates from old PKLs and write v2 coordinates.

    Physical arrays are deliberately copied byte-for-byte at the ndarray level;
    only ``x_surface`` and ``x_volume`` are transformed.
    """

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source PKL directory not found: {source_dir}")
    if source_dir == output_dir:
        raise ValueError(
            "Migration output must differ from --source-pkl-dir so the old "
            "recoverable PKLs cannot be overwritten"
        )
    scalar_path = source_dir / SCALARS_FILENAME
    if not scalar_path.is_file():
        raise FileNotFoundError(f"Missing source normalization scalars: {scalar_path}")
    with scalar_path.open("rb") as stream:
        source_scalars = pickle.load(stream)
    if not isinstance(source_scalars, dict):
        raise TypeError(f"Expected a scalar dictionary in {scalar_path}")

    sample_paths = numbered_pkl_paths(source_dir)
    if not sample_paths:
        raise FileNotFoundError(f"No numbered PKLs found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_outputs = [output_dir / path.name for path in sample_paths]
    pending_outputs.append(output_dir / SCALARS_FILENAME)
    existing = [path for path in pending_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Migration output already exists: {existing[0]}; use --overwrite"
        )
    if overwrite:
        for path in numbered_pkl_paths(output_dir):
            path.unlink()
        output_scalar_path = output_dir / SCALARS_FILENAME
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
    print(f"Samples              : {len(sample_paths)}")

    # Pass 1: reverse the existing coordinate transform and get independent
    # physical bounds without retaining more than one large sample in memory.
    for position, path in enumerate(sample_paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        for field in COORDINATE_FIELDS:
            if field not in sample:
                raise KeyError(f"{path}: missing {field!r}")
            old_min, old_range = coordinate_scalars(source_scalars, field)
            normalized = np.asarray(sample[field], dtype=np.float32)
            physical = normalized * old_range + old_min
            bounds[field]["min"] = np.minimum(
                bounds[field]["min"], physical.min(axis=0)
            )
            bounds[field]["max"] = np.maximum(
                bounds[field]["max"], physical.max(axis=0)
            )
        records.append(
            {
                "sample_id": int(path.stem),
                "surface_cells": int(len(sample["x_surface"])),
                "volume_cells": int(len(sample["x_volume"])),
            }
        )
        print(f"[recover statistics {position}/{len(sample_paths)}] {path.name}")
        del sample, normalized, physical
        gc.collect()

    new_scalars = copy.deepcopy(source_scalars)
    new_scalars.pop("coords_min", None)
    new_scalars.pop("coords_max", None)
    new_scalars.pop("coords_range", None)
    new_scalars["schema_version"] = "drivaerstar-normalized-v3-surface-scale"
    new_scalars["coordinate_fields"] = COORDINATE_FIELDS
    normalization = dict(new_scalars.get("normalization", {}))
    normalization.update(
        {
            "x_surface": "global_surface_minmax",
            "x_volume": "global_surface_minmax",
        }
    )
    new_scalars["normalization"] = normalization
    new_scalars["coordinate_normalization_scopes"] = {
        "x_surface": {
            "statistics_prefix": "surface_coords",
            "population": "all surface coordinates reconstructed from source PKLs",
        },
        "x_volume": {
            "statistics_prefix": "surface_coords",
            "population": "all surface coordinates reconstructed from source PKLs",
        },
    }
    new_scalars["coordinate_scale_policy"] = (
        "surface_scale_for_surface_and_volume"
    )
    new_scalars["volume_coordinates_may_exceed_unit_interval"] = True
    new_scalars["migrated_from_schema"] = source_scalars.get("schema_version")
    new_scalars["migrated_from_directory"] = str(source_dir)
    new_scalars["output_dir"] = str(output_dir)
    new_scalars["num_samples"] = len(sample_paths)
    # Keep rich original manifest records, but only for files in this migration.
    migrated_ids = {int(path.stem) for path in sample_paths}
    source_records = source_scalars.get("samples")
    if isinstance(source_records, list):
        filtered_records = [
            record
            for record in source_records
            if isinstance(record, dict)
            and "sample_id" in record
            and int(record["sample_id"]) in migrated_ids
        ]
        new_scalars["samples"] = filtered_records or records
    else:
        new_scalars["samples"] = records
    surface_range = bounds["x_surface"]["max"] - bounds["x_surface"]["min"]
    surface_range[surface_range <= np.finfo(np.float32).eps] = 1.0
    new_scalars["surface_coords_min"] = bounds["x_surface"]["min"].astype(np.float32)
    new_scalars["surface_coords_max"] = bounds["x_surface"]["max"].astype(np.float32)
    new_scalars["surface_coords_range"] = surface_range.astype(np.float32)
    new_scalars["volume_coords_min"] = new_scalars["surface_coords_min"].copy()
    new_scalars["volume_coords_max"] = new_scalars["surface_coords_max"].copy()
    new_scalars["volume_coords_range"] = new_scalars["surface_coords_range"].copy()
    new_scalars["volume_domain_min"] = bounds["x_volume"]["min"].astype(np.float32)
    new_scalars["volume_domain_max"] = bounds["x_volume"]["max"].astype(np.float32)

    # Pass 2: reconstruct with old statistics and normalize with v2 statistics.
    for position, path in enumerate(sample_paths, start=1):
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        for field in COORDINATE_FIELDS:
            old_min, old_range = coordinate_scalars(source_scalars, field)
            physical = np.asarray(sample[field], dtype=np.float32) * old_range + old_min
            new_min, new_range = coordinate_scalars(new_scalars, field)
            sample[field] = ((physical - new_min) / new_range).astype(
                np.float32, copy=False
            )
            if field == "x_surface" and (
                np.any(sample[field] < -1.0e-5)
                or np.any(sample[field] > 1.0 + 1.0e-5)
            ):
                raise RuntimeError(f"Migrated {field} outside [0,1] in {path}")
        sample["schema_version"] = new_scalars["schema_version"]
        sample_normalization = dict(sample.get("normalization", {}))
        sample_normalization.update(
            {
                "x_surface": "global_surface_minmax",
                "x_volume": "global_surface_minmax",
            }
        )
        sample["normalization"] = sample_normalization
        destination = output_dir / path.name
        atomic_pickle_dump(sample, destination)
        verify_sample(destination, int(path.stem))
        print(f"[migrate {position}/{len(sample_paths)}] {path.name}")
        del sample, physical
        gc.collect()

    atomic_pickle_dump(new_scalars, output_dir / SCALARS_FILENAME)
    print(f"Migration complete: {len(sample_paths)} samples")
    print(f"Normalization scalars: {output_dir / SCALARS_FILENAME}")
    print("Physical fields were copied without re-normalization.")
    return 0


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print("Preprocessing configuration: process_DrivAerStar.py")
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
        if args.output_dir is None:
            raise ValueError(
                "PKL migration requires --output-dir different from the source"
            )
        return migrate_processed_pkls(
            args.source_pkl_dir, output_dir, args.overwrite
        )
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered_cases = sorted(
        data_dir.rglob("*.case"),
        key=case_id,
    )
    if not discovered_cases:
        raise FileNotFoundError(f"No EnSight .case files found below {data_dir}")
    incomplete_cases = [
        record
        for case_path in discovered_cases
        if (record := incomplete_case_record(case_path)) is not None
    ]
    case_paths = [
        case_path
        for case_path in discovered_cases
        if incomplete_case_record(case_path) is None
    ]
    if not case_paths:
        raise RuntimeError("No complete EnSight cases are available for conversion")
    identifiers = [case_id(path) for path in case_paths]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate case IDs found: {identifiers}")

    print(f"Dataset    : {data_dir}")
    print(f"PKL output : {output_dir}")
    print(f"Discovered : {len(discovered_cases)} cases")
    print(f"Complete   : {len(case_paths)} cases")
    print(f"Incomplete : {len(incomplete_cases)} cases")
    for record in incomplete_cases:
        print(
            f"[skip incomplete] {record['source_case']}: "
            f"missing={record['missing_files']}"
        )
    scalars, records = collect_statistics(case_paths)
    scalars["num_discovered_cases"] = len(discovered_cases)
    scalars["num_complete_cases"] = len(case_paths)
    scalars["num_incomplete_cases"] = len(incomplete_cases)
    scalars["incomplete_cases"] = incomplete_cases
    scalars["source_files_preserved"] = True

    for position, case_path in enumerate(case_paths, start=1):
        identifier = case_id(case_path)
        destination = output_dir / f"{identifier}.pkl"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {destination}; use --overwrite to replace it"
            )
        started = time.monotonic()
        sample = normalized_sample(case_path, scalars)
        atomic_pickle_dump(sample, destination)
        verify_sample(destination, identifier)
        print(
            f"[convert {position}/{len(case_paths)}] {case_path.name} -> "
            f"{destination.name}, surface={len(sample['x_surface'])}, "
            f"volume={len(sample['x_volume'])}, {time.monotonic() - started:.2f}s"
        )
        del sample
        gc.collect()

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
    if verified_scalars.get("num_samples") != len(case_paths):
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

    for case_path in case_paths:
        if not (output_dir / f"{case_id(case_path)}.pkl").is_file():
            raise RuntimeError(f"Missing output for {case_path}")

    print(f"Conversion complete: {len(case_paths)} samples")
    print(f"Normalization scalars: {scalars_path}")
    print("Original EnSight case directories were preserved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
