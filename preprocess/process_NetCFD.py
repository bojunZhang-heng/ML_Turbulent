"""Preprocess full-volume DrivAerNet-CFD legacy VTK files.

The DrivAerNet-CFD files in this project are unstructured CFD meshes rather
than car-only surface meshes.  They contain point and cell fields named
``p``, ``wallShearStress`` and ``U``.  This script supports two regions:

``car_surface`` (default)
    Extract the exterior boundary, identify the vehicle wall from non-zero
    wall-shear stress while excluding the ground plane, and save normalized
    coordinates, unit point normals and the selected physical fields.

``volume``
    Save all volume points and the selected point fields.  Normals are not
    defined for volume points and are therefore not written.

Coordinates are normalized with one global min-max transform.  Every
physical field is normalized component-wise with one global mean/std
transform.  Statistics are accumulated over all selected files and stored in
``normalization_scalars.pkl``.  The source VTK files are never modified.

Examples
--------
Car-surface pressure, compatible with the existing DrivAerNet pressure
training data format::

    python preprocess/process_NetCFD.py --overwrite

Car-surface pressure and WSS::

    python preprocess/process_NetCFD.py \
        --fields pressure wss --overwrite

Full-volume pressure and velocity::

    python preprocess/process_NetCFD.py \
        --region volume --fields pressure velocity \
        --output-dir preprocess/DrivAerNet-CFD/volume_pklFile --overwrite
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "DrivAerNet-CFD"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "pklFile"
LOGGER = logging.getLogger("process_NetCFD")

FIELD_ALIASES = {
    "pressure": ("p", "pressure", "static_pressure", "surface_pressure"),
    "wss": (
        "wallShearStress",
        "wss",
        "wall_shear_stress",
        "wallshearstress",
        "wall_shear_stress_vector",
    ),
    "velocity": ("U", "velocity", "flow_velocity", "cell_velocity"),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def print_run_configuration(args: argparse.Namespace) -> None:
    """Print the effective CLI configuration without changing processing."""
    print("=" * 72)
    print("Preprocessing configuration: process_NetCFD.py")
    for name, value in sorted(vars(args).items()):
        print(f"{name:<24}: {value}")
    print("=" * 72)


def canonical_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def list_arrays(attributes: vtk.vtkDataSetAttributes) -> list[str]:
    return [
        attributes.GetArrayName(index) or f"<unnamed:{index}>"
        for index in range(attributes.GetNumberOfArrays())
    ]


def find_array(
    attributes: vtk.vtkDataSetAttributes, aliases: Iterable[str]
) -> tuple[str, vtk.vtkDataArray] | None:
    aliases = tuple(aliases)
    canonical_aliases = {canonical_name(alias) for alias in aliases}
    for alias in aliases:
        array = attributes.GetArray(alias)
        if array is not None:
            return alias, array
    for index in range(attributes.GetNumberOfArrays()):
        name = attributes.GetArrayName(index)
        if canonical_name(name) in canonical_aliases:
            return name or f"<unnamed:{index}>", attributes.GetArray(index)
    return None


def read_dataset(path: Path) -> vtk.vtkDataSet:
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise ValueError(f"Unable to read a non-empty VTK dataset: {path}")
    return dataset


def ensure_point_fields(
    dataset: vtk.vtkDataSet, required_fields: Iterable[str], path: Path
) -> vtk.vtkDataSet:
    """Ensure required fields exist as point data, converting cell data once."""

    missing = [
        field
        for field in required_fields
        if find_array(dataset.GetPointData(), FIELD_ALIASES[field]) is None
    ]
    if not missing:
        return dataset

    unavailable = [
        field
        for field in missing
        if find_array(dataset.GetCellData(), FIELD_ALIASES[field]) is None
    ]
    if unavailable:
        raise KeyError(
            f"Missing fields {unavailable} in {path.name}. "
            f"Point arrays={list_arrays(dataset.GetPointData())}; "
            f"cell arrays={list_arrays(dataset.GetCellData())}"
        )

    LOGGER.info(
        "%s: converting cell fields %s to point data", path.name, missing
    )
    converter = vtk.vtkCellDataToPointData()
    converter.SetInputData(dataset)
    converter.PassCellDataOn()
    converter.Update()
    output = converter.GetOutput()
    for field in required_fields:
        if find_array(output.GetPointData(), FIELD_ALIASES[field]) is None:
            raise RuntimeError(
                f"VTK failed to provide point field {field!r} for {path.name}"
            )
    return output


def point_coordinates(dataset: vtk.vtkDataSet, path: Path) -> np.ndarray:
    points = dataset.GetPoints()
    if points is None:
        raise ValueError(f"Dataset has no points: {path}")
    coords = vtk_to_numpy(points.GetData()).astype(np.float32, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected XYZ coordinates, got {coords.shape} in {path.name}")
    if not np.isfinite(coords).all():
        raise ValueError(f"Coordinates contain NaN/Inf in {path.name}")
    return coords


def numpy_point_field(
    dataset: vtk.vtkDataSet, field: str, path: Path
) -> tuple[np.ndarray, str]:
    match = find_array(dataset.GetPointData(), FIELD_ALIASES[field])
    if match is None:
        raise KeyError(
            f"Could not find point field {field!r} in {path.name}; "
            f"arrays={list_arrays(dataset.GetPointData())}"
        )
    vtk_name, vtk_array = match
    values = vtk_to_numpy(vtk_array).astype(np.float32, copy=False)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != dataset.GetNumberOfPoints():
        raise ValueError(
            f"Unexpected {field} shape {values.shape} in {path.name}; "
            f"points={dataset.GetNumberOfPoints()}"
        )
    expected_components = 1 if field == "pressure" else 3
    if values.shape[1] != expected_components:
        raise ValueError(
            f"{field} must have {expected_components} components, got "
            f"{values.shape[1]} from VTK array {vtk_name!r}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{vtk_name!r} contains NaN/Inf in {path.name}")
    return values, vtk_name


def extract_surface(dataset: vtk.vtkDataSet) -> vtk.vtkPolyData:
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(dataset)
    if hasattr(surface_filter, "PassThroughPointIdsOn"):
        surface_filter.PassThroughPointIdsOn()
        if hasattr(surface_filter, "SetOriginalPointIdsName"):
            surface_filter.SetOriginalPointIdsName("source_point_ids")
    surface_filter.Update()
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(surface_filter.GetOutput())
    if surface.GetNumberOfPoints() == 0:
        raise ValueError("The VTK dataset has no extractable exterior surface")
    return surface


def add_unit_normals(surface: vtk.vtkPolyData, path: Path) -> vtk.vtkPolyData:
    """Compute point normals without splitting or reordering surface points."""

    input_coords = point_coordinates(surface, path)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(surface)
    normal_filter.ComputePointNormalsOn()
    normal_filter.ComputeCellNormalsOff()
    normal_filter.SplittingOff()
    normal_filter.ConsistencyOn()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.NonManifoldTraversalOn()
    normal_filter.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(normal_filter.GetOutput())

    output_coords = point_coordinates(output, path)
    if output_coords.shape != input_coords.shape or not np.allclose(
        output_coords, input_coords, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            f"VTK changed point count/order while computing normals for {path.name}"
        )
    normals_array = output.GetPointData().GetNormals()
    if normals_array is None:
        raise ValueError(f"Unable to compute point normals for {path.name}")
    normals = vtk_to_numpy(normals_array).astype(np.float32, copy=False)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.isfinite(normals).all():
        raise ValueError(f"Computed normals contain NaN/Inf for {path.name}")
    # A full CFD boundary may contain degenerate far-field cells.  Do not fail
    # before the vehicle mask is applied; normalize valid normals here and
    # strictly validate the selected vehicle normals in extract_sample().
    valid = lengths[:, 0] > 1e-12
    if not np.all(valid):
        LOGGER.warning(
            "%s: full exterior contains %d zero-normal points; vehicle points "
            "will be validated after wall selection",
            path.name,
            int(np.count_nonzero(~valid)),
        )
    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[valid] = normals[valid] / lengths[valid]
    normals_array.DeepCopy(numpy_to_vtk(normalized, deep=True))
    return output


def car_surface_mask(
    surface: vtk.vtkPolyData,
    path: Path,
    ground_axis: int,
    ground_tolerance: float,
    wss_threshold: float,
) -> np.ndarray:
    """Identify vehicle-wall points using WSS and remove the ground plane."""

    coords = point_coordinates(surface, path)
    wss, _ = numpy_point_field(surface, "wss", path)
    ground_coordinate = float(coords[:, ground_axis].min())
    height_mask = coords[:, ground_axis] > ground_coordinate + ground_tolerance
    wss_mask = np.linalg.norm(wss, axis=1) > wss_threshold
    mask = height_mask & wss_mask
    if not np.any(mask):
        raise ValueError(
            f"Vehicle mask is empty for {path.name}. ground_axis={ground_axis}, "
            f"ground={ground_coordinate}, ground_tolerance={ground_tolerance}, "
            f"wss_threshold={wss_threshold}"
        )
    LOGGER.info(
        "%s: exterior points=%d, selected vehicle points=%d, ground=%g",
        path.name,
        coords.shape[0],
        int(mask.sum()),
        ground_coordinate,
    )
    return mask


@dataclass
class ExtractedSample:
    coords: np.ndarray
    fields: dict[str, np.ndarray]
    vtk_field_names: dict[str, str]
    normals: np.ndarray | None = None
    source_point_ids: np.ndarray | None = None


def extract_sample(
    path: Path,
    region: str,
    fields: tuple[str, ...],
    ground_axis: int,
    ground_tolerance: float,
    wss_threshold: float,
    save_point_ids: bool,
) -> ExtractedSample:
    dataset = read_dataset(path)
    # car_surface selection always needs WSS, even if WSS is not an output.
    required = tuple(dict.fromkeys((*fields, *(('wss',) if region == 'car_surface' else ()))))
    dataset = ensure_point_fields(dataset, required, path)

    if region == "volume":
        coords = point_coordinates(dataset, path)
        arrays: dict[str, np.ndarray] = {}
        names: dict[str, str] = {}
        for field in fields:
            arrays[field], names[field] = numpy_point_field(dataset, field, path)
        source_ids = np.arange(coords.shape[0], dtype=np.int64) if save_point_ids else None
        return ExtractedSample(coords, arrays, names, source_point_ids=source_ids)

    surface = add_unit_normals(extract_surface(dataset), path)
    mask = car_surface_mask(
        surface, path, ground_axis, ground_tolerance, wss_threshold
    )
    coords = point_coordinates(surface, path)[mask].copy()
    normals_vtk = surface.GetPointData().GetNormals()
    normals = vtk_to_numpy(normals_vtk).astype(np.float32, copy=False)[mask].copy()
    selected_lengths = np.linalg.norm(normals, axis=1)
    invalid_normals = ~np.isfinite(selected_lengths) | (selected_lengths <= 1e-12)
    if np.any(invalid_normals):
        raise ValueError(
            f"Selected vehicle surface contains {int(invalid_normals.sum())} "
            f"invalid normals in {path.name}"
        )
    arrays = {}
    names = {}
    for field in fields:
        values, vtk_name = numpy_point_field(surface, field, path)
        arrays[field] = values[mask].copy()
        names[field] = vtk_name

    source_ids = None
    if save_point_ids:
        id_match = find_array(surface.GetPointData(), ("source_point_ids", "vtkOriginalPointIds"))
        if id_match is None:
            raise RuntimeError(
                "--save-point-ids was requested, but VTK did not preserve original IDs"
            )
        source_ids = vtk_to_numpy(id_match[1]).astype(np.int64, copy=False)[mask].copy()
    return ExtractedSample(coords, arrays, names, normals, source_ids)


@dataclass
class RunningMoments:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray, chunk_size: int = 1_000_000) -> None:
        """Update stable moments in chunks to limit float64 temporary memory."""

        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"Expected non-empty 2D values, got {values.shape}")
        for start in range(0, values.shape[0], chunk_size):
            batch = np.asarray(values[start : start + chunk_size], dtype=np.float64)
            batch_count = batch.shape[0]
            batch_mean = batch.mean(axis=0)
            centered = batch - batch_mean
            batch_m2 = np.square(centered).sum(axis=0)
            if self.count == 0:
                self.count = batch_count
                self.mean = batch_mean
                self.m2 = batch_m2
                continue
            if batch_mean.shape != self.mean.shape:
                raise ValueError("Physical field component count changed")
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * (batch_count / total)
            self.m2 = (
                self.m2
                + batch_m2
                + np.square(delta) * self.count * batch_count / total
            )
            self.count = total

    def population_std(self) -> np.ndarray:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("No values were accumulated")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std[std <= np.finfo(np.float32).eps] = 1.0
        return std


def collect_statistics(
    vtk_files: list[Path],
    region: str,
    fields: tuple[str, ...],
    ground_axis: int,
    ground_tolerance: float,
    wss_threshold: float,
) -> dict:
    coords_min = np.full(3, np.inf, dtype=np.float64)
    coords_max = np.full(3, -np.inf, dtype=np.float64)
    moments = {field: RunningMoments() for field in fields}
    inspected: list[dict] = []

    for index, path in enumerate(vtk_files, start=1):
        started = time.monotonic()
        sample = extract_sample(
            path,
            region,
            fields,
            ground_axis,
            ground_tolerance,
            wss_threshold,
            save_point_ids=False,
        )
        coords_min = np.minimum(coords_min, sample.coords.min(axis=0))
        coords_max = np.maximum(coords_max, sample.coords.max(axis=0))
        for field in fields:
            moments[field].update(sample.fields[field])
        inspected.append(
            {
                "source_file": path.name,
                "points": int(sample.coords.shape[0]),
                "vtk_fields": sample.vtk_field_names,
            }
        )
        LOGGER.info(
            "Statistics [%d/%d] %s: points=%d, %.2fs",
            index,
            len(vtk_files),
            path.name,
            sample.coords.shape[0],
            time.monotonic() - started,
        )
        del sample

    coords_range = coords_max - coords_min
    constant = coords_range <= np.finfo(np.float32).eps
    if np.any(constant):
        LOGGER.warning("Constant coordinate axes %s get range=1", np.flatnonzero(constant).tolist())
        coords_range[constant] = 1.0

    scalars: dict = {
        "schema_version": "drivaernet-cfd-v1",
        "region": region,
        "fields": list(fields),
        "coords_min": coords_min.astype(np.float32)[None, :],
        "coords_max": coords_max.astype(np.float32)[None, :],
        "coords_range": coords_range.astype(np.float32)[None, :],
        "samples": inspected,
        "normalization": {
            "coor": "global_min_max",
            "normals": "unit_vectors_unchanged" if region == "car_surface" else None,
            **{field: "global_mean_std_per_component" for field in fields},
        },
    }
    for field, state in moments.items():
        scalars[f"{field}_mean"] = state.mean.astype(np.float32)[None, :]
        scalars[f"{field}_std"] = state.population_std().astype(np.float32)[None, :]
        scalars[f"{field}_point_count"] = int(state.count)
    return scalars


def sample_id(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    return str(int(match.group(1))) if match else path.stem


def atomic_pickle_dump(value: object, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_normalized_samples(
    vtk_files: list[Path],
    output_dir: Path,
    region: str,
    fields: tuple[str, ...],
    scalars: dict,
    ground_axis: int,
    ground_tolerance: float,
    wss_threshold: float,
    save_point_ids: bool,
) -> list[Path]:
    written: list[Path] = []
    used_names: set[str] = set()
    for index, path in enumerate(vtk_files, start=1):
        started = time.monotonic()
        extracted = extract_sample(
            path,
            region,
            fields,
            ground_axis,
            ground_tolerance,
            wss_threshold,
            save_point_ids,
        )
        normalized_coords = (
            (extracted.coords - scalars["coords_min"]) / scalars["coords_range"]
        ).astype(np.float32, copy=False)
        sample: dict[str, np.ndarray] = {"coor": normalized_coords}
        if extracted.normals is not None:
            sample["normals"] = extracted.normals.astype(np.float32, copy=False)
        for field in fields:
            sample[field] = (
                (extracted.fields[field] - scalars[f"{field}_mean"])
                / scalars[f"{field}_std"]
            ).astype(np.float32, copy=False)
        if extracted.source_point_ids is not None:
            sample["source_point_ids"] = extracted.source_point_ids

        arrays = list(sample.values())
        if not all(array.shape[0] == normalized_coords.shape[0] for array in arrays):
            raise RuntimeError(f"Point arrays are not aligned for {path.name}")
        if not all(np.isfinite(array).all() for array in arrays):
            raise RuntimeError(f"Normalized sample contains NaN/Inf for {path.name}")
        if np.any(normalized_coords < -1e-5) or np.any(normalized_coords > 1.0 + 1e-5):
            raise RuntimeError(f"Normalized coordinates outside [0, 1] for {path.name}")

        output_name = f"{sample_id(path)}.pkl"
        if output_name in used_names:
            raise ValueError(f"Duplicate output name {output_name!r}")
        used_names.add(output_name)
        output_path = output_dir / output_name
        atomic_pickle_dump(sample, output_path)
        written.append(output_path)
        LOGGER.info(
            "Write [%d/%d] %s -> %s: points=%d, %.2fs",
            index,
            len(vtk_files),
            path.name,
            output_path.name,
            normalized_coords.shape[0],
            time.monotonic() - started,
        )
        del extracted, sample
    return written


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.pkl"))
    if existing and not overwrite:
        preview = ", ".join(path.name for path in existing[:5])
        raise FileExistsError(
            f"{output_dir} already contains PKL files ({preview}). "
            "Use --overwrite to replace generated files."
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    region: str,
    fields: tuple[str, ...],
    pattern: str,
    overwrite: bool,
    ground_axis: int,
    ground_tolerance: float,
    wss_threshold: float,
    save_point_ids: bool,
) -> dict:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    vtk_files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not vtk_files:
        raise FileNotFoundError(
            f"No VTK files matching {pattern!r} were found in {input_dir}"
        )
    if len(set(fields)) != len(fields):
        raise ValueError(f"Duplicate --fields are not allowed: {fields}")

    prepare_output_directory(output_dir, overwrite)
    LOGGER.info(
        "Found %d VTK files; region=%s; fields=%s; output=%s",
        len(vtk_files),
        region,
        fields,
        output_dir,
    )
    started = time.monotonic()
    scalars = collect_statistics(
        vtk_files,
        region,
        fields,
        ground_axis,
        ground_tolerance,
        wss_threshold,
    )
    written = write_normalized_samples(
        vtk_files,
        output_dir,
        region,
        fields,
        scalars,
        ground_axis,
        ground_tolerance,
        wss_threshold,
        save_point_ids,
    )
    atomic_pickle_dump(scalars, output_dir / "normalization_scalars.pkl")
    LOGGER.info(
        "Completed %d samples in %.1fs. Scalars: %s",
        len(written),
        time.monotonic() - started,
        output_dir / "normalization_scalars.pkl",
    )
    return scalars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--region", choices=("car_surface", "volume"), default="car_surface")
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=tuple(FIELD_ALIASES),
        default=["pressure"],
        help="One or more fields to normalize and save",
    )
    parser.add_argument("--pattern", default="*.vtk", help="Input filename glob")
    parser.add_argument(
        "--ground-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Vertical coordinate axis used to remove the ground (default: Z=2)",
    )
    parser.add_argument(
        "--ground-tolerance",
        type=float,
        default=1e-4,
        help="Remove points within this distance above the minimum ground coordinate",
    )
    parser.add_argument(
        "--wss-threshold",
        type=float,
        default=1e-12,
        help="Minimum WSS magnitude used to identify vehicle-wall points",
    )
    parser.add_argument(
        "--save-point-ids",
        action="store_true",
        help="Save original VTK point IDs for mapping predictions back to the source mesh",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing PKL files in output-dir before processing",
    )
    args = parser.parse_args()
    if args.ground_tolerance < 0 or args.wss_threshold < 0:
        parser.error("ground-tolerance and wss-threshold must be non-negative")
    args.fields = tuple(args.fields)
    return args


def main() -> None:
    args = parse_args()
    print_run_configuration(args)
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        region=args.region,
        fields=args.fields,
        pattern=args.pattern,
        overwrite=args.overwrite,
        ground_axis=args.ground_axis,
        ground_tolerance=args.ground_tolerance,
        wss_threshold=args.wss_threshold,
        save_point_ids=args.save_point_ids,
    )


if __name__ == "__main__":
    configure_logging()
    main()
