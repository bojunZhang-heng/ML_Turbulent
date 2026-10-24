"""Preprocess DrivAerNet legacy VTK surface files into normalized PKL samples.

Each sample PKL contains exactly three arrays:

    coor:     (N, 3), global min-max normalized coordinates
    normals:  (N, 3), unit surface normals (not statistically normalized)
    pressure: (N, 1), global mean-std normalized pressure

or, when ``--field wss`` is selected:

    wss:      (N, C), global per-component mean-std normalized WSS

The statistics are computed across every point of every selected VTK file.
Raw VTK files are never modified.
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
from vtk.util.numpy_support import vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path("/Users/zhangbojun/Desktop/E_S_WW_WM_small")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "DrivAerNet_data"
LOGGER = logging.getLogger("process_DrivAerNet")

FIELD_ALIASES = {
    "pressure": (
        "pressure",
        "p",
        "static_pressure",
        "staticpressure",
        "surface_pressure",
    ),
    "wss": (
        "wss",
        "wall_shear_stress",
        "wallshearstress",
        "wall_shear",
        "wallshear",
        "wall_shear_stress_vector",
    ),
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def canonical_name(name: str | None) -> str:
    """Return a case/punctuation-insensitive VTK array name."""

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

    # Prefer exact names so that an explicitly named field wins when multiple
    # case-insensitive variants exist.
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


def cell_field_to_point_field(
    dataset: vtk.vtkDataSet,
    aliases: Iterable[str],
) -> tuple[vtk.vtkDataSet, str] | None:
    found = find_array(dataset.GetCellData(), aliases)
    if found is None:
        return None

    field_name, _ = found
    converter = vtk.vtkCellDataToPointData()
    converter.SetInputData(dataset)
    converter.PassCellDataOn()
    converter.Update()

    output = converter.GetOutput()
    if find_array(output.GetPointData(), (field_name,)) is None:
        raise RuntimeError(
            f"VTK failed to convert cell field {field_name!r} to point data"
        )
    return output, field_name


def extract_surface(dataset: vtk.vtkDataSet) -> vtk.vtkPolyData:
    if isinstance(dataset, vtk.vtkPolyData):
        surface = vtk.vtkPolyData()
        surface.ShallowCopy(dataset)
        return surface

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(dataset)
    surface_filter.Update()
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(surface_filter.GetOutput())
    if surface.GetNumberOfPoints() == 0:
        raise ValueError("The VTK dataset has no extractable surface")
    return surface


def read_surface_and_field(
    path: Path, field: str
) -> tuple[vtk.vtkPolyData, np.ndarray, str, str]:
    aliases = FIELD_ALIASES[field]
    dataset = read_dataset(path)

    point_match = find_array(dataset.GetPointData(), aliases)
    association = "point"
    if point_match is None:
        converted = cell_field_to_point_field(dataset, aliases)
        if converted is None:
            raise KeyError(
                f"Could not find {field!r} in {path.name}. "
                f"Point arrays={list_arrays(dataset.GetPointData())}; "
                f"cell arrays={list_arrays(dataset.GetCellData())}; "
                f"accepted aliases={list(aliases)}"
            )
        dataset, converted_name = converted
        point_match = find_array(dataset.GetPointData(), (converted_name,))
        association = "cell_to_point"

    surface = extract_surface(dataset)
    surface_match = find_array(surface.GetPointData(), (point_match[0], *aliases))
    if surface_match is None:
        raise KeyError(
            f"Field {point_match[0]!r} disappeared during surface extraction of "
            f"{path.name}; surface point arrays={list_arrays(surface.GetPointData())}"
        )

    field_name, vtk_array = surface_match
    values = vtk_to_numpy(vtk_array).astype(np.float32, copy=False)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != surface.GetNumberOfPoints():
        raise ValueError(
            f"Unexpected {field} shape {values.shape} in {path.name}; "
            f"surface points={surface.GetNumberOfPoints()}"
        )
    if field == "pressure" and values.shape[1] != 1:
        raise ValueError(
            f"Pressure must be scalar, but {field_name!r} has "
            f"{values.shape[1]} components in {path.name}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{field_name!r} contains NaN/Inf in {path.name}")
    return surface, values, field_name, association


def point_coordinates(surface: vtk.vtkPolyData, path: Path) -> np.ndarray:
    points = surface.GetPoints()
    if points is None:
        raise ValueError(f"Surface has no points: {path}")
    coords = vtk_to_numpy(points.GetData()).astype(np.float32, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected XYZ coordinates, got {coords.shape} in {path.name}")
    if not np.isfinite(coords).all():
        raise ValueError(f"Coordinates contain NaN/Inf in {path.name}")
    return coords


def unit_point_normals(surface: vtk.vtkPolyData, path: Path) -> np.ndarray:
    """Compute unit point normals without splitting or reordering vertices."""

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

    normal_surface = normal_filter.GetOutput()
    normals_array = normal_surface.GetPointData().GetNormals()
    if normals_array is None:
        raise ValueError(f"Unable to compute point normals for {path.name}")

    output_coords = point_coordinates(normal_surface, path)
    if output_coords.shape != input_coords.shape or not np.allclose(
        output_coords, input_coords, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            f"VTK changed point count/order while computing normals for {path.name}"
        )

    normals = vtk_to_numpy(normals_array).astype(np.float32, copy=False)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.isfinite(normals).all() or np.any(lengths <= 1e-12):
        bad_count = int(np.count_nonzero(lengths <= 1e-12))
        raise ValueError(
            f"Computed normals contain invalid or zero vectors in {path.name}; "
            f"zero_count={bad_count}"
        )
    normals = normals / lengths
    return normals.astype(np.float32, copy=False)


@dataclass
class RunningMoments:
    """Numerically stable per-component moments for differently sized meshes."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.ndim != 2 or values64.shape[0] == 0:
            raise ValueError(f"Expected non-empty 2D values, got {values64.shape}")

        batch_count = values64.shape[0]
        batch_mean = values64.mean(axis=0)
        centered = values64 - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        if batch_mean.shape != self.mean.shape:
            raise ValueError(
                f"Physical field component count changed from {self.mean.shape} "
                f"to {batch_mean.shape}"
            )
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
            raise RuntimeError("No physical values were accumulated")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std[std <= np.finfo(np.float32).eps] = 1.0
        return std


def collect_statistics(
    vtk_files: list[Path], field: str
) -> tuple[dict, list[dict]]:
    coords_min = np.full(3, np.inf, dtype=np.float64)
    coords_max = np.full(3, -np.inf, dtype=np.float64)
    moments = RunningMoments()
    inspected: list[dict] = []

    for index, path in enumerate(vtk_files, start=1):
        started = time.monotonic()
        surface, values, vtk_field_name, association = read_surface_and_field(
            path, field
        )
        coords = point_coordinates(surface, path)
        coords_min = np.minimum(coords_min, coords.min(axis=0))
        coords_max = np.maximum(coords_max, coords.max(axis=0))
        moments.update(values)
        inspected.append(
            {
                "source_file": path.name,
                "points": int(coords.shape[0]),
                "vtk_field": vtk_field_name,
                "association": association,
                "components": int(values.shape[1]),
            }
        )
        LOGGER.info(
            "Statistics [%d/%d] %s: points=%d, field=%s, components=%d, %.2fs",
            index,
            len(vtk_files),
            path.name,
            coords.shape[0],
            vtk_field_name,
            values.shape[1],
            time.monotonic() - started,
        )

    coords_range = coords_max - coords_min
    zero_range = coords_range <= np.finfo(np.float32).eps
    if np.any(zero_range):
        LOGGER.warning(
            "Constant coordinate axes detected at indices %s; their range is set to 1",
            np.flatnonzero(zero_range).tolist(),
        )
        coords_range[zero_range] = 1.0

    field_mean = moments.mean.astype(np.float32)[None, :]
    field_std = moments.population_std().astype(np.float32)[None, :]
    scalars = {
        "coords_min": coords_min.astype(np.float32)[None, :],
        "coords_max": coords_max.astype(np.float32)[None, :],
        "coords_range": coords_range.astype(np.float32)[None, :],
        f"{field}_mean": field_mean,
        f"{field}_std": field_std,
        "field_name": field,
        "field_components": int(field_mean.shape[1]),
        "physical_point_count": int(moments.count),
        "samples": inspected,
        "normalization": {
            "coor": "global_min_max",
            field: "global_mean_std_per_component",
            "normals": "unit_vectors_unchanged",
        },
    }
    return scalars, inspected


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


def validate_normalized_sample(sample: dict, field: str, path: Path) -> None:
    if set(sample) != {"coor", "normals", field}:
        raise RuntimeError(f"Unexpected sample keys for {path.name}: {set(sample)}")

    coords = sample["coor"]
    normals = sample["normals"]
    values = sample[field]
    if not (coords.shape[0] == normals.shape[0] == values.shape[0]):
        raise RuntimeError(f"Node arrays are not aligned for {path.name}")
    if coords.shape[1] != 3 or normals.shape[1] != 3:
        raise RuntimeError(f"Invalid coordinate/normal shape for {path.name}")
    if not all(np.isfinite(array).all() for array in sample.values()):
        raise RuntimeError(f"Normalized sample contains NaN/Inf for {path.name}")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"Normals are not unit vectors for {path.name}")
    if np.any(coords < -1e-5) or np.any(coords > 1.0 + 1e-5):
        raise RuntimeError(f"Min-max coordinates are outside [0, 1] for {path.name}")


def write_normalized_samples(
    vtk_files: list[Path], output_dir: Path, field: str, scalars: dict
) -> list[Path]:
    written: list[Path] = []
    used_names: set[str] = set()
    for index, path in enumerate(vtk_files, start=1):
        started = time.monotonic()
        surface, values, _, _ = read_surface_and_field(path, field)
        coords = point_coordinates(surface, path)
        normals = unit_point_normals(surface, path)
        if not (coords.shape[0] == normals.shape[0] == values.shape[0]):
            raise RuntimeError(
                f"Coordinates, normals and {field} are not aligned in {path.name}"
            )

        normalized_coords = (
            (coords - scalars["coords_min"]) / scalars["coords_range"]
        ).astype(np.float32, copy=False)
        normalized_values = (
            (values - scalars[f"{field}_mean"]) / scalars[f"{field}_std"]
        ).astype(np.float32, copy=False)
        sample = {
            "coor": normalized_coords,
            "normals": normals,
            field: normalized_values,
        }
        validate_normalized_sample(sample, field, path)

        output_name = f"{sample_id(path)}.pkl"
        if output_name in used_names:
            raise ValueError(
                f"Duplicate output sample name {output_name!r}; source={path.name}"
            )
        used_names.add(output_name)
        output_path = output_dir / output_name
        atomic_pickle_dump(sample, output_path)
        written.append(output_path)
        LOGGER.info(
            "Write [%d/%d] %s -> %s: shape=%s, %.2fs",
            index,
            len(vtk_files),
            path.name,
            output_path.name,
            normalized_values.shape,
            time.monotonic() - started,
        )
    return written


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.pkl"))
    if existing and not overwrite:
        preview = ", ".join(path.name for path in existing[:5])
        raise FileExistsError(
            f"{output_dir} already contains PKL files ({preview}). "
            "Use --overwrite to replace generated PKL files."
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for path in output_dir.glob("*.pkl.tmp"):
            path.unlink()


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    field: str,
    pattern: str = "*.vtk",
    overwrite: bool = False,
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

    prepare_output_directory(output_dir, overwrite)
    LOGGER.info(
        "Found %d VTK files; field=%s; output=%s",
        len(vtk_files),
        field,
        output_dir,
    )
    started = time.monotonic()
    scalars, _ = collect_statistics(vtk_files, field)
    written = write_normalized_samples(vtk_files, output_dir, field, scalars)
    atomic_pickle_dump(scalars, output_dir / "normalization_scalars.pkl")
    LOGGER.info(
        "Completed %d samples in %.1fs. Normalization scalars: %s",
        len(written),
        time.monotonic() - started,
        output_dir / "normalization_scalars.pkl",
    )
    return scalars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing legacy VTK files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for normalized PKL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--field",
        choices=tuple(FIELD_ALIASES),
        default="pressure",
        help="Physical field to extract and normalize",
    )
    parser.add_argument("--pattern", default="*.vtk", help="Input filename glob")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing PKL files in output-dir before processing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        field=args.field,
        pattern=args.pattern,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    configure_logging()
    main()
