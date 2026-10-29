#!/usr/bin/env python3
"""Convert DrivAerStar EnSight Gold cases into normalized PKL samples.

The EnSight variables are cell fields (``per element``), so coordinates are
cell centers rather than mesh vertices. Automotive surface parts are selected
from two-dimensional parts carrying both Pressure and WallShearStress while
excluding wind-tunnel ``Block.*`` boundaries. Three-dimensional parts carrying
Velocity form the volume field.
"""

from __future__ import annotations

import argparse
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
import vtk
from vtk.util.numpy_support import vtk_to_numpy


DEFAULT_DATA_DIR = Path(
    "/Users/zhangbojun/ML_Turbulent/preprocess/DrivAerStar"
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
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data-dir so numbered PKLs are placed at its root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--remove-source-directories",
        action="store_true",
        help=(
            "After verification, retain a small number of cases in raw_data/ "
            "and permanently remove the original case_E_batch_* directories."
        ),
    )
    parser.add_argument(
        "--keep-source-cases",
        type=int,
        default=10,
        help=(
            "Number of complete raw EnSight cases retained for topology-based "
            "visualization when --remove-source-directories is enabled."
        ),
    )
    return parser.parse_args()


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
    coords_min = np.full(3, np.inf, dtype=np.float64)
    coords_max = np.full(3, -np.inf, dtype=np.float64)
    moments = {
        "y_spressure": RunningMoments(),
        "y_velocity": RunningMoments(),
        "y_wss": RunningMoments(),
    }
    records: list[dict[str, Any]] = []
    for position, case_path in enumerate(case_paths, start=1):
        started = time.monotonic()
        arrays, metadata = extract_case(case_path)
        for field in ("x_surface", "x_volume"):
            coords_min = np.minimum(coords_min, arrays[field].min(axis=0))
            coords_max = np.maximum(coords_max, arrays[field].max(axis=0))
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

    coords_range = coords_max - coords_min
    coords_range[coords_range <= np.finfo(np.float32).eps] = 1.0
    scalars: dict[str, Any] = {
        "schema_version": "drivaerstar-normalized-v1",
        "coords_min": coords_min.astype(np.float32),
        "coords_max": coords_max.astype(np.float32),
        "coords_range": coords_range.astype(np.float32),
        "coordinate_fields": ("x_surface", "x_volume"),
        "physical_fields": ("y_spressure", "y_velocity", "y_wss"),
        "field_mapping": {
            "Pressure": "y_spressure",
            "Velocity": "y_velocity",
            "WallShearStress": "y_wss",
            "surface_cell_centers": "x_surface",
            "volume_cell_centers": "x_volume",
        },
        "normalization": {
            "x_surface": "global_shared_minmax",
            "x_volume": "global_shared_minmax",
            "y_spressure": "global_componentwise_mean_std",
            "y_velocity": "global_componentwise_mean_std",
            "y_wss": "global_componentwise_mean_std",
        },
        "coordinate_semantics": "cell_centers",
        "surface_selection": (
            "2D EnSight parts with Pressure and WallShearStress, excluding "
            "Block.* wind-tunnel boundaries and interface parts"
        ),
        "volume_selection": "3D EnSight parts with Velocity",
        "samples": records,
        "num_samples": len(records),
    }
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
    for field in ("x_surface", "x_volume"):
        arrays[field] = (
            (arrays[field] - scalars["coords_min"]) / scalars["coords_range"]
        ).astype(np.float32, copy=False)
    for field in ("y_spressure", "y_velocity", "y_wss"):
        arrays[field] = (
            (arrays[field] - scalars[f"{field}_mean"])
            / scalars[f"{field}_std"]
        ).astype(np.float32, copy=False)
    retained_case = scalars.get("retained_source_cases", {}).get(case_id(case_path))
    if retained_case is not None:
        metadata["visualization_case"] = retained_case
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
    if not (
        len(sample["x_surface"])
        == len(sample["y_spressure"])
        == len(sample["y_wss"])
    ):
        raise RuntimeError(f"Surface fields are misaligned in {path}")
    if len(sample["x_volume"]) != len(sample["y_velocity"]):
        raise RuntimeError(f"Volume fields are misaligned in {path}")


def main() -> int:
    args = parse_args()
    if args.keep_source_cases < 0:
        raise ValueError("--keep-source-cases must be greater than or equal to 0")
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir
    )
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir = output_dir / "raw_data"
    discovered_cases = sorted(
        (
            path
            for path in data_dir.rglob("*.case")
            if not path.is_relative_to(raw_data_dir)
        ),
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
    retained_case_paths = (
        case_paths[: min(args.keep_source_cases, len(case_paths))]
        if args.remove_source_directories
        else case_paths
    )
    retained_case_ids = [case_id(path) for path in retained_case_paths]
    scalars["source_cleanup_requested"] = args.remove_source_directories
    scalars["retained_source_case_ids"] = retained_case_ids
    if args.remove_source_directories:
        scalars["retained_source_cases"] = {
            case_id(path): str(
                raw_data_dir / f"{case_id(path):05d}" / path.name
            )
            for path in retained_case_paths
        }
    else:
        scalars["retained_source_cases"] = {
            case_id(path): str(path) for path in retained_case_paths
        }

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

    for case_path in case_paths:
        if not (output_dir / f"{case_id(case_path)}.pkl").is_file():
            raise RuntimeError(f"Missing output for {case_path}")

    if args.remove_source_directories:
        # Keep deletion all-or-nothing: first verify every PKL, then move the
        # selected raw cases to raw_data/<id>/ and remove all batch folders.
        if incomplete_cases:
            raise RuntimeError(
                "Refusing to remove source directories because incomplete "
                "cases were discovered"
            )
        batch_directories = sorted(
            path
            for path in data_dir.glob("case_E_batch_*")
            if path.is_dir()
        )
        if not batch_directories:
            raise RuntimeError("No case_E_batch_* source directories resolved")

        raw_data_dir.mkdir(parents=True, exist_ok=True)
        retained_ids = set(retained_case_ids)
        for position, case_path in enumerate(retained_case_paths, start=1):
            identifier = case_id(case_path)
            destination = raw_data_dir / f"{identifier:05d}"
            if destination.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        f"Retained source output exists: {destination}; "
                        "use --overwrite to replace it"
                    )
                shutil.rmtree(destination)
            shutil.move(str(case_path.parent), str(destination))
            expected_case = destination / case_path.name
            retained_files = {
                ".case": expected_case,
                **case_companions(expected_case),
            }
            missing_retained = [
                suffix
                for suffix, path in retained_files.items()
                if not path.is_file()
            ]
            if missing_retained:
                raise RuntimeError(
                    f"Retained visualization case is incomplete: "
                    f"{expected_case}, missing={missing_retained}"
                )
            print(
                f"[retain {position}/{len(retained_case_paths)}] "
                f"case {identifier} -> {destination}"
            )

        for position, directory in enumerate(batch_directories, start=1):
            shutil.rmtree(directory)
            print(
                f"[remove {position}/{len(batch_directories)}] {directory}"
            )
        print(
            f"Retained {len(retained_ids)} raw cases for visualization in "
            f"{raw_data_dir}"
        )

    print(f"Conversion complete: {len(case_paths)} samples")
    print(f"Normalization scalars: {scalars_path}")
    if not args.remove_source_directories:
        print("Original EnSight case directories were preserved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
