"""Preprocess ShapeNet-Car MLCFD data and map predictions back to VTK meshes.

The input may contain extracted ``param*/<case>/`` directories, the original
``param*.tar.gz`` archives, or both.  One pickle is written per CFD case.
The node-index maps stored in every pickle make model predictions reversible.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


DEFAULT_INPUT = Path.home() / "Desktop" / "mlcfd_data" / "training_data"
DEFAULT_OUTPUT = Path.home() / "Desktop" / "mlcfd_data" / "pkl_data"
LOGGER = logging.getLogger("mlcfd_preprocess")
PRESSURE_FILE = "quadpress_smpl.vtk"
VELOCITY_FILE = "hexvelo_smpl.vtk"
SCHEMA_VERSION = "mlcfd-shapenet-car-v2"


@dataclass(frozen=True)
class CaseSource:
    fold: str
    case_id: str
    kind: str
    root: Path
    pressure_ref: str
    velocity_ref: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _pickle_dump(value, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def discover_cases(input_dir: Path, folds: set[str] | None = None) -> list[CaseSource]:
    cases: dict[tuple[str, str], CaseSource] = {}

    for fold_dir in sorted(input_dir.glob("param[0-8]")):
        if not fold_dir.is_dir() or (folds and fold_dir.name not in folds):
            continue
        for case_dir in sorted(path for path in fold_dir.iterdir() if path.is_dir()):
            pressure_path = case_dir / PRESSURE_FILE
            velocity_path = case_dir / VELOCITY_FILE
            if pressure_path.exists() and velocity_path.exists():
                cases[(fold_dir.name, case_dir.name)] = CaseSource(
                    fold=fold_dir.name,
                    case_id=case_dir.name,
                    kind="directory",
                    root=input_dir.resolve(),
                    pressure_ref=str(pressure_path.resolve()),
                    velocity_ref=str(velocity_path.resolve()),
                )

    for archive_path in sorted(input_dir.glob("param[0-8].tar.gz")):
        fold = archive_path.name.removesuffix(".tar.gz")
        if folds and fold not in folds:
            continue
        members: dict[str, set[str]] = {}
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile() or len(path.parts) != 3 or path.parts[0] != fold:
                    continue
                if path.name in {PRESSURE_FILE, VELOCITY_FILE}:
                    members.setdefault(path.parts[1], set()).add(path.name)
        for case_id, filenames in members.items():
            if {PRESSURE_FILE, VELOCITY_FILE}.issubset(filenames):
                key = (fold, case_id)
                cases.setdefault(
                    key,
                    CaseSource(
                        fold=fold,
                        case_id=case_id,
                        kind="archive",
                        root=archive_path.resolve(),
                        pressure_ref=f"{fold}/{case_id}/{PRESSURE_FILE}",
                        velocity_ref=f"{fold}/{case_id}/{VELOCITY_FILE}",
                    ),
                )

    return [cases[key] for key in sorted(cases)]


def _copy_archive_member(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    member = archive.getmember(member_name)
    if not member.isfile():
        raise ValueError(f"Archive member is not a file: {member_name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Unable to read archive member: {member_name}")
    with source, destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def materialize_case(source: CaseSource, directory: Path) -> tuple[Path, Path]:
    pressure_path = directory / PRESSURE_FILE
    velocity_path = directory / VELOCITY_FILE
    if source.kind == "directory":
        return Path(source.pressure_ref), Path(source.velocity_ref)
    with tarfile.open(source.root, "r:gz") as archive:
        _copy_archive_member(archive, source.pressure_ref, pressure_path)
        _copy_archive_member(archive, source.velocity_ref, velocity_path)
    return pressure_path, velocity_path


def read_grid(path: Path) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    output = vtk.vtkUnstructuredGrid()
    output.DeepCopy(reader.GetOutput())
    if output.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not read a non-empty unstructured grid: {path}")
    return output


def point_array(grid: vtk.vtkUnstructuredGrid, components: int) -> np.ndarray:
    point_data = grid.GetPointData()
    preferred = point_data.GetVectors() if components == 3 else point_data.GetScalars()
    if preferred is not None and preferred.GetNumberOfComponents() == components:
        return vtk_to_numpy(preferred).astype(np.float32, copy=False)
    for index in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(index)
        if array is not None and array.GetNumberOfComponents() == components:
            return vtk_to_numpy(array).astype(np.float32, copy=False)
    names = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    raise ValueError(f"No {components}-component point array found; available={names}")


def surface_normals(surface_grid: vtk.vtkUnstructuredGrid) -> np.ndarray:
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(surface_grid)
    geometry.Update()
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputConnection(geometry.GetOutputPort())
    normal_filter.ComputePointNormalsOn()
    normal_filter.ComputeCellNormalsOff()
    normal_filter.SplittingOff()
    normal_filter.ConsistencyOn()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.Update()
    output = normal_filter.GetOutput()
    normals_array = output.GetPointData().GetNormals()
    if normals_array is None or output.GetNumberOfPoints() != surface_grid.GetNumberOfPoints():
        raise ValueError("Unable to compute one surface normal per pressure-grid point")
    output_points = vtk_to_numpy(output.GetPoints().GetData())
    input_points = vtk_to_numpy(surface_grid.GetPoints().GetData())
    if not np.allclose(output_points, input_points, rtol=0.0, atol=1e-7):
        raise ValueError("VTK changed the surface point order while computing normals")
    normals = vtk_to_numpy(normals_array).astype(np.float32, copy=False)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0.0, 1.0, lengths)


def build_sample(
    source: CaseSource,
    mapping_tolerance: float,
    sample_index: int,
    raw_data_dir: Path,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="mlcfd_case_") as temporary_dir:
        pressure_path, velocity_path = materialize_case(source, Path(temporary_dir))
        raw_pressure_path = raw_data_dir / f"{sample_index}_pressure.vtk"
        raw_velocity_path = raw_data_dir / f"{sample_index}_velocity.vtk"
        shutil.copy2(pressure_path, raw_pressure_path)
        shutil.copy2(velocity_path, raw_velocity_path)
        pressure_grid = read_grid(pressure_path)
        velocity_grid = read_grid(velocity_path)

    surface_points = vtk_to_numpy(pressure_grid.GetPoints().GetData()).astype(np.float32, copy=False)
    volume_points = vtk_to_numpy(velocity_grid.GetPoints().GetData()).astype(np.float32, copy=False)
    pressure = point_array(pressure_grid, 1).reshape(-1, 1)
    velocity = point_array(velocity_grid, 3)

    volume_tree = cKDTree(volume_points)
    surface_distances, surface_to_volume = volume_tree.query(surface_points, k=1)
    scale = max(float(np.ptp(volume_points, axis=0).max()), 1.0)
    absolute_tolerance = mapping_tolerance * scale
    unmatched = surface_distances > absolute_tolerance
    surface_to_volume = surface_to_volume.astype(np.int64)
    surface_to_volume[unmatched] = -1

    matched_volume_ids = surface_to_volume[surface_to_volume >= 0]
    if len(np.unique(matched_volume_ids)) != len(matched_volume_ids):
        raise ValueError(f"Surface-to-volume mapping is not one-to-one for {source.case_id}")
    exterior_mask = np.ones(len(volume_points), dtype=bool)
    exterior_mask[matched_volume_ids] = False
    exterior_volume_ids = np.flatnonzero(exterior_mask).astype(np.int64)

    surface_tree = cKDTree(surface_points)
    distances, nearest_surface = surface_tree.query(volume_points, k=1)
    volume_normals = (volume_points - surface_points[nearest_surface]) / (distances[:, None] + 1e-8)
    volume_normals = volume_normals.astype(np.float32, copy=False)
    sdf = distances.astype(np.float32, copy=False)
    normals_surface = surface_normals(pressure_grid)

    exterior_points = volume_points[exterior_volume_ids]
    exterior_input = np.column_stack(
        (exterior_points, sdf[exterior_volume_ids], volume_normals[exterior_volume_ids])
    ).astype(np.float32, copy=False)
    surface_input = np.column_stack(
        (surface_points, np.zeros(len(surface_points), dtype=np.float32), normals_surface)
    ).astype(np.float32, copy=False)

    exterior_output = np.column_stack(
        (velocity[exterior_volume_ids], np.zeros(len(exterior_volume_ids), dtype=np.float32))
    ).astype(np.float32, copy=False)
    surface_velocity = np.zeros((len(surface_points), 3), dtype=np.float32)
    matched = surface_to_volume >= 0
    surface_velocity[matched] = velocity[surface_to_volume[matched]]
    surface_output = np.column_stack((surface_velocity, pressure)).astype(np.float32, copy=False)

    combined_input = np.concatenate((exterior_input, surface_input), axis=0)
    combined_output = np.concatenate((exterior_output, surface_output), axis=0)
    surface_mask = np.concatenate(
        (np.zeros(len(exterior_input), dtype=bool), np.ones(len(surface_input), dtype=bool))
    )
    combined_to_volume = np.concatenate((exterior_volume_ids, surface_to_volume))
    combined_to_surface = np.concatenate(
        (np.full(len(exterior_input), -1, dtype=np.int64), np.arange(len(surface_points), dtype=np.int64))
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_index": sample_index,
        "sample_id": source.case_id,
        "fold": source.fold,
        "input": combined_input,
        "output": combined_output,
        "x": combined_input,
        "y": combined_output,
        "pos": np.concatenate((exterior_points, surface_points), axis=0).astype(np.float32, copy=False),
        "surface_mask": surface_mask,
        "surf": surface_mask,
        "mapping": {
            "combined_to_volume_point": combined_to_volume,
            "combined_to_surface_point": combined_to_surface,
            "surface_to_volume_point": surface_to_volume,
            "surface_match_distance": surface_distances.astype(np.float32),
        },
        "source": {
            "kind": "directory",
            "root": str(raw_data_dir),
            "pressure_ref": str(raw_pressure_path),
            "velocity_ref": str(raw_velocity_path),
            "pressure_num_points": len(surface_points),
            "velocity_num_points": len(volume_points),
            "original_kind": source.kind,
            "original_root": str(source.root),
            "original_pressure_ref": source.pressure_ref,
            "original_velocity_ref": source.velocity_ref,
        },
        "normalization": "raw",
    }


def compute_statistics(sample_paths: Iterable[Path]) -> dict:
    coords_min = np.full(3, np.inf, dtype=np.float64)
    coords_max = np.full(3, -np.inf, dtype=np.float64)
    sdf_min = np.inf
    sdf_max = -np.inf
    output_sum = np.zeros(4, dtype=np.float64)
    output_sum_sq = np.zeros(4, dtype=np.float64)
    count = 0
    for path in sample_paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        model_input = np.asarray(sample["input"], dtype=np.float64)
        target = np.asarray(sample["output"], dtype=np.float64)
        coords_min = np.minimum(coords_min, model_input[:, :3].min(axis=0))
        coords_max = np.maximum(coords_max, model_input[:, :3].max(axis=0))
        sdf_min = min(sdf_min, float(model_input[:, 3].min()))
        sdf_max = max(sdf_max, float(model_input[:, 3].max()))
        output_sum += target.sum(axis=0)
        output_sum_sq += np.square(target).sum(axis=0)
        count += len(model_input)
    if count == 0:
        raise ValueError("Cannot compute statistics from an empty dataset")

    coords_range = coords_max - coords_min
    coords_range[coords_range == 0.0] = 1.0
    sdf_range = sdf_max - sdf_min
    if sdf_range == 0.0:
        sdf_range = 1.0
    output_mean = output_sum / count
    output_std = np.sqrt(np.maximum(output_sum_sq / count - np.square(output_mean), 0.0))
    output_std[output_std == 0.0] = 1.0
    return {
        "schema_version": SCHEMA_VERSION,
        "coords_min": coords_min.astype(np.float32),
        "coords_max": coords_max.astype(np.float32),
        "coords_range": coords_range.astype(np.float32),
        "sdf_min": np.float32(sdf_min),
        "sdf_max": np.float32(sdf_max),
        "sdf_range": np.float32(sdf_range),
        "output_mean": output_mean.astype(np.float32),
        "output_std": output_std.astype(np.float32),
        "input_features": ("x", "y", "z", "sdf", "normal_x", "normal_y", "normal_z"),
        "output_features": ("velocity_x", "velocity_y", "velocity_z", "pressure"),
        "num_points": count,
    }


def normalize_samples(sample_paths: Iterable[Path], scalars: dict) -> None:
    for path in sample_paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        model_input = np.asarray(sample["input"], dtype=np.float32).copy()
        model_input[:, :3] = (
            (model_input[:, :3] - scalars["coords_min"]) / scalars["coords_range"]
        )
        model_input[:, 3] = (
            (model_input[:, 3] - scalars["sdf_min"]) / scalars["sdf_range"]
        )
        sample["input"] = model_input
        sample["output"] = ((sample["output"] - scalars["output_mean"]) / scalars["output_std"]).astype(np.float32)
        sample["x"] = sample["input"]
        sample["y"] = sample["output"]
        sample["pos"] = sample["input"][:, :3].copy()
        sample["normalization"] = {
            "coordinates": "minmax",
            "sdf": "minmax",
            "normals": "unit_vector_unchanged",
            "physical_quantities": "mean_std",
        }
        _pickle_dump(sample, path)


def preprocess(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw_data_dir = (
        args.raw_data_dir.expanduser().resolve()
        if args.raw_data_dir is not None
        else output_dir.parent / "raw_data"
    )
    folds = set(args.folds) if args.folds else None
    cases = discover_cases(input_dir, folds)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError(f"No valid MLCFD cases found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    sample_paths: list[Path] = []
    sample_records: list[dict] = []
    started = time.monotonic()
    for index, source in enumerate(cases, start=1):
        case_started = time.monotonic()
        output_path = output_dir / f"{index}.pkl"
        raw_pressure_path = raw_data_dir / f"{index}_pressure.vtk"
        raw_velocity_path = raw_data_dir / f"{index}_velocity.vtk"
        existing_paths = [
            path for path in (output_path, raw_pressure_path, raw_velocity_path) if path.exists()
        ]
        if existing_paths and not args.overwrite:
            raise FileExistsError(
                "Output exists; pass --overwrite to replace it: "
                + ", ".join(str(path) for path in existing_paths)
            )
        sample = build_sample(source, args.mapping_tolerance, index, raw_data_dir)
        _pickle_dump(sample, output_path)
        sample_paths.append(output_path)
        sample_records.append(
            {
                "sample_index": index,
                "fold": source.fold,
                "sample_id": source.case_id,
                "pkl": output_path.name,
                "pressure_vtk": raw_pressure_path.name,
                "velocity_vtk": raw_velocity_path.name,
                "num_nodes": len(sample["input"]),
                "num_surface_nodes": int(sample["surface_mask"].sum()),
            }
        )
        LOGGER.info(
            "[%d/%d] %s/%s: nodes=%d, surface=%d, %.2fs",
            index,
            len(cases),
            source.fold,
            source.case_id,
            len(sample["input"]),
            int(sample["surface_mask"].sum()),
            time.monotonic() - case_started,
        )

    if args.statistics_folds:
        statistics_folds = set(args.statistics_folds)
        statistics_paths = [
            path
            for path, record in zip(sample_paths, sample_records)
            if record["fold"] in statistics_folds
        ]
    else:
        statistics_folds = {record["fold"] for record in sample_records}
        statistics_paths = sample_paths
    if not statistics_paths:
        raise ValueError("None of the processed samples belong to --statistics-folds")
    scalars = compute_statistics(statistics_paths)
    scalars["statistics_folds"] = tuple(sorted(statistics_folds))
    scalars["samples"] = sample_records
    scalars["input_dir"] = str(input_dir)
    scalars["raw_data_dir"] = str(raw_data_dir)
    scalars["normalization"] = {
        "coordinates": "minmax",
        "sdf": "minmax",
        "normals": "unit_vector_unchanged",
        "physical_quantities": "mean_std",
    }
    normalize_samples(sample_paths, scalars)
    _pickle_dump(scalars, output_dir / "normalization_scalars.pkl")
    LOGGER.info("Processed %d cases into %s in %.1fs", len(cases), output_dir, time.monotonic() - started)


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        prediction = np.load(path)
    else:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if isinstance(value, dict):
            for key in ("prediction", "predictions", "output", "y"):
                if key in value:
                    value = value[key]
                    break
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        prediction = np.asarray(value)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.ndim != 2:
        raise ValueError(f"Prediction must have shape (N, C) or (1, N, C), got {prediction.shape}")
    return prediction.astype(np.float32, copy=False)


def add_point_array(grid: vtk.vtkDataSet, values: np.ndarray, name: str) -> None:
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    array.SetName(name)
    grid.GetPointData().AddArray(array)
    if values.ndim == 2 and values.shape[1] == 3:
        grid.GetPointData().SetActiveVectors(name)
    elif values.ndim == 1 or values.shape[1] == 1:
        grid.GetPointData().SetActiveScalars(name)


def write_grid(grid: vtk.vtkUnstructuredGrid, path: Path) -> None:
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise OSError(f"VTK failed to write {path}")


def export_prediction(args: argparse.Namespace) -> None:
    with args.sample_pkl.expanduser().open("rb") as handle:
        sample = pickle.load(handle)
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The sample pickle was not created by this script")

    prediction = load_prediction(args.prediction.expanduser())
    if args.normalized:
        scalar_path = args.scalars or args.sample_pkl.expanduser().parent / "normalization_scalars.pkl"
        with scalar_path.open("rb") as handle:
            scalars = pickle.load(handle)
        if prediction.shape[1] == 4:
            prediction = prediction * scalars["output_std"] + scalars["output_mean"]
        elif prediction.shape[1] == 1:
            prediction = prediction * scalars["output_std"][-1] + scalars["output_mean"][-1]

    source_dict = sample["source"]
    source = CaseSource(
        fold=sample["fold"],
        case_id=sample["sample_id"],
        kind=source_dict["kind"],
        root=Path(source_dict["root"]),
        pressure_ref=source_dict["pressure_ref"],
        velocity_ref=source_dict["velocity_ref"],
    )
    with tempfile.TemporaryDirectory(prefix="mlcfd_export_") as temporary_dir:
        pressure_path, velocity_path = materialize_case(source, Path(temporary_dir))
        pressure_grid = read_grid(pressure_path)
        velocity_grid = read_grid(velocity_path)

    combined_to_volume = sample["mapping"]["combined_to_volume_point"]
    combined_to_surface = sample["mapping"]["combined_to_surface_point"]
    surface_rows = np.flatnonzero(sample["surface_mask"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if prediction.shape[0] == len(combined_to_volume) and prediction.shape[1] == 4:
        volume_prediction = np.full((velocity_grid.GetNumberOfPoints(), 3), np.nan, dtype=np.float32)
        valid_volume = combined_to_volume >= 0
        volume_prediction[combined_to_volume[valid_volume]] = prediction[valid_volume, :3]
        pressure_prediction = np.full(pressure_grid.GetNumberOfPoints(), np.nan, dtype=np.float32)
        valid_surface = combined_to_surface >= 0
        pressure_prediction[combined_to_surface[valid_surface]] = prediction[valid_surface, 3]
    elif prediction.shape[1] == 1 and prediction.shape[0] in {len(surface_rows), len(combined_to_surface)}:
        pressure_prediction = np.full(pressure_grid.GetNumberOfPoints(), np.nan, dtype=np.float32)
        values = prediction[:, 0] if len(prediction) == len(surface_rows) else prediction[surface_rows, 0]
        pressure_prediction[combined_to_surface[surface_rows]] = values
        volume_prediction = None
    else:
        raise ValueError(
            "Expected a full (combined_nodes, 4) prediction or pressure-only "
            f"(surface_nodes, 1)/(combined_nodes, 1), got {prediction.shape}"
        )

    if volume_prediction is not None:
        add_point_array(velocity_grid, volume_prediction, args.velocity_name)
        write_grid(velocity_grid, output_dir / f"{sample['sample_index']}_velocity_prediction.vtk")
    add_point_array(pressure_grid, pressure_prediction, args.pressure_name)
    write_grid(pressure_grid, output_dir / f"{sample['sample_index']}_pressure_prediction.vtk")
    LOGGER.info("Prediction mapped back to the original grids in %s", output_dir)


def inspect_pickle(args: argparse.Namespace) -> None:
    with args.sample_pkl.expanduser().open("rb") as handle:
        sample = pickle.load(handle)
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for nested_key, nested_value in value.items():
                description = (
                    f"shape={nested_value.shape}, dtype={nested_value.dtype}"
                    if isinstance(nested_value, np.ndarray)
                    else repr(nested_value)
                )
                print(f"  {nested_key}: {description}")
        else:
            print(f"{key}: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Convert MLCFD cases to per-case pickle files")
    preprocess_parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    preprocess_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    preprocess_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Directory for numbered original VTK grids (default: sibling raw_data)",
    )
    preprocess_parser.add_argument("--folds", nargs="+", choices=[f"param{i}" for i in range(9)])
    preprocess_parser.add_argument("--limit", type=int, help="Process only the first N cases for a smoke test")
    preprocess_parser.add_argument("--mapping-tolerance", type=float, default=1e-5)
    preprocess_parser.add_argument(
        "--statistics-folds",
        nargs="+",
        choices=[f"param{i}" for i in range(9)],
        help="Folds used to compute mean/std (normally the training folds only)",
    )
    preprocess_parser.add_argument("--overwrite", action="store_true")
    preprocess_parser.set_defaults(handler=preprocess)

    export_parser = subparsers.add_parser("export", help="Write a model prediction back to the original VTK grids")
    export_parser.add_argument("--sample-pkl", type=Path, required=True)
    export_parser.add_argument("--prediction", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--normalized", action="store_true", help="Prediction is z-score normalized")
    export_parser.add_argument("--scalars", type=Path)
    export_parser.add_argument("--velocity-name", default="predicted_velocity")
    export_parser.add_argument("--pressure-name", default="predicted_pressure")
    export_parser.set_defaults(handler=export_prediction)

    inspect_parser = subparsers.add_parser("inspect", help="Print the structure of one generated pickle")
    inspect_parser.add_argument("--sample-pkl", type=Path, required=True)
    inspect_parser.set_defaults(handler=inspect_pickle)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    configure_logging()
    main()
