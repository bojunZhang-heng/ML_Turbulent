from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

try:
    from .segmentation import create_seg_matrix
except ImportError:
    from segmentation import create_seg_matrix


DEFAULT_INPUT = Path("/Users/zhangbojun/Downloads/ShapeNet_dataset/vtk")
DEFAULT_OUTPUT = DEFAULT_INPUT / "data"
LOGGER = logging.getLogger("shapenet_preprocess")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _id_filter():
    filter_type = getattr(vtk, "vtkIdFilter", None)
    if filter_type is None:
        filter_type = vtk.vtkGenerateIds
    return filter_type()


def read_legacy_dataset(file_path):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(file_path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not read a non-empty VTK dataset from {file_path}")
    return dataset


def extract_surface(dataset):
    id_filter = _id_filter()
    id_filter.SetInputData(dataset)
    id_filter.PointIdsOn()
    id_filter.SetPointIdsArrayName("OriginalPointIds")
    id_filter.Update()

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(id_filter.GetOutputPort())
    surface_filter.Update()

    surface = vtk.vtkPolyData()
    surface.ShallowCopy(surface_filter.GetOutput())
    if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfPolys() == 0:
        raise ValueError("VTK dataset does not contain an extractable polygonal surface")
    return surface


def unit_normals(surface):
    normals_array = surface.GetPointData().GetArray("normals")
    if normals_array is None:
        normals_array = surface.GetPointData().GetNormals()

    if normals_array is None:
        normal_filter = vtk.vtkPolyDataNormals()
        normal_filter.SetInputData(surface)
        normal_filter.ComputePointNormalsOn()
        normal_filter.ComputeCellNormalsOff()
        normal_filter.SplittingOff()
        normal_filter.ConsistencyOn()
        normal_filter.AutoOrientNormalsOn()
        normal_filter.Update()
        normals_array = normal_filter.GetOutput().GetPointData().GetNormals()

    if normals_array is None:
        raise ValueError("Unable to read or compute surface normals")

    normals = numpy_support.vtk_to_numpy(normals_array).astype(np.float32, copy=False)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return normals / lengths


def find_pressure_array(point_data, names):
    for name in names:
        pressure_array = point_data.GetArray(name)
        if pressure_array is not None:
            return name, pressure_array
    available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    raise ValueError(
        f"Pressure field not found. Tried {names}; available point arrays: {available}"
    )


def read_shapenet_vtk(file_path, pressure_names=("pressure", "p")):
    volume = read_legacy_dataset(file_path)
    surface = extract_surface(volume)

    coords = numpy_support.vtk_to_numpy(surface.GetPoints().GetData()).astype(
        np.float32, copy=False
    )
    pressure_name, pressure_array = find_pressure_array(
        surface.GetPointData(), pressure_names
    )
    pressure = numpy_support.vtk_to_numpy(pressure_array).astype(
        np.float32, copy=False
    ).reshape(-1, 1)
    normals = unit_normals(surface)

    original_ids_array = surface.GetPointData().GetArray("OriginalPointIds")
    if original_ids_array is None:
        original_point_ids = np.arange(len(coords), dtype=np.int64)
    else:
        original_point_ids = numpy_support.vtk_to_numpy(original_ids_array).astype(
            np.int64, copy=False
        )

    if not (len(coords) == len(pressure) == len(normals) == len(original_point_ids)):
        raise ValueError(f"Point-data arrays are inconsistent in {file_path}")

    metadata = {
        "source_dataset_type": volume.GetClassName(),
        "source_num_points": volume.GetNumberOfPoints(),
        "source_num_cells": volume.GetNumberOfCells(),
        "surface_num_points": surface.GetNumberOfPoints(),
        "surface_num_cells": surface.GetNumberOfCells(),
        "pressure_field": pressure_name,
    }
    return coords, pressure, normals, original_point_ids, surface, metadata


def sample_id(file_path):
    match = re.search(r"(\d+)$", file_path.stem)
    return str(int(match.group(1))) if match else file_path.stem


def compute_global_statistics(samples):
    coords = np.vstack([sample["coor"] for sample in samples])
    pressure = np.vstack([sample["pressure"] for sample in samples])

    coords_min = coords.min(axis=0, keepdims=True)
    coords_max = coords.max(axis=0, keepdims=True)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1.0

    pressure_mean = np.asarray([pressure.mean()], dtype=np.float32)
    pressure_std = np.asarray([pressure.std()], dtype=np.float32)
    pressure_std[pressure_std == 0] = 1.0

    return {
        "coords_min": coords_min,
        "coords_max": coords_max,
        "coords_range": coords_range,
        "pressure_mean": pressure_mean,
        "pressure_std": pressure_std,
    }


def normalize_sample(sample, scalars):
    coords = (sample["coor"] - scalars["coords_min"]) / scalars["coords_range"]
    pressure = (
        sample["pressure"] - scalars["pressure_mean"]
    ) / scalars["pressure_std"]
    sample["coor"] = coords.astype(np.float32, copy=False)
    sample["pressure"] = pressure.astype(np.float32, copy=False)
    sample["features_6d"] = np.concatenate(
        [sample["coor"], sample["normals"]], axis=1
    ).astype(np.float32, copy=False)


def process_folder(
    input_dir,
    output_dir,
    threshold_angles=(8, 7, 6),
    min_graph_size_ratio=0.0001,
    max_segments=256,
    pattern="*.vtk",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    vtk_files = sorted(input_dir.glob(pattern))
    if not vtk_files:
        raise ValueError(f"No VTK files matching {pattern!r} found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.pkl"):
        old_file.unlink()

    LOGGER.info("Found %d VTK files in %s", len(vtk_files), input_dir)
    samples = []
    for index, vtk_file in enumerate(vtk_files, start=1):
        started_at = time.monotonic()
        coords, pressure, normals, original_ids, surface, metadata = read_shapenet_vtk(
            vtk_file
        )
        seg_matrix, labels = create_seg_matrix(
            coords,
            surface,
            threshold_angles=threshold_angles,
            min_graph_size_ratio=min_graph_size_ratio,
            max_segments=max_segments,
        )
        sample = {
            "coor": coords,
            "node_cluster_flags": labels,
            "pressure": pressure,
            "normals": normals,
            "seg_matrix": seg_matrix,
            "original_point_ids": original_ids,
            "source_file": vtk_file.name,
            "metadata": metadata,
        }
        samples.append(sample)
        LOGGER.info(
            "Prepared %d/%d %s: volume_points=%d, surface_points=%d, segments=%d, %.2fs",
            index,
            len(vtk_files),
            vtk_file.name,
            metadata["source_num_points"],
            len(coords),
            seg_matrix.shape[0],
            time.monotonic() - started_at,
        )

    scalars = compute_global_statistics(samples)
    with (output_dir / "normalization_scalars.pkl").open("wb") as handle:
        pickle.dump(scalars, handle)

    for vtk_file, sample in zip(vtk_files, samples):
        normalize_sample(sample, scalars)
        output_file = output_dir / f"{sample_id(vtk_file)}.pkl"
        with output_file.open("wb") as handle:
            pickle.dump(sample, handle)

    LOGGER.info("Saved %d samples and normalization scalars to %s", len(samples), output_dir)
    return scalars


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ShapeNet car legacy VTK volume meshes to PGD-NO surface samples."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", default="*.vtk")
    parser.add_argument("--threshold-angles", type=float, nargs="+", default=[8, 7, 6])
    parser.add_argument("--min-graph-size-ratio", type=float, default=0.0001)
    parser.add_argument("--max-segments", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        threshold_angles=args.threshold_angles,
        min_graph_size_ratio=args.min_graph_size_ratio,
        max_segments=args.max_segments,
    )


if __name__ == "__main__":
    configure_logging()
    main()
