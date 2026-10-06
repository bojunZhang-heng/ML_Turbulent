"""Unified PGD preprocessing for DrivAer and ShapeNet legacy VTK files."""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import vtk
from scipy.sparse import csr_matrix
from vtk.util import numpy_support

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"
LOGGER = logging.getLogger("pgd_preprocess")


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)


def _id_filter():
    return (vtk.vtkIdFilter if hasattr(vtk, "vtkIdFilter") else vtk.vtkGenerateIds)()


def read_dataset(path):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path)); reader.ReadAllScalarsOn(); reader.ReadAllVectorsOn(); reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not read a non-empty VTK file: {path}")
    return dataset


def as_surface(dataset):
    if isinstance(dataset, vtk.vtkPolyData):
        surface = vtk.vtkPolyData(); surface.ShallowCopy(dataset); return surface
    ids = _id_filter(); ids.SetInputData(dataset); ids.PointIdsOn(); ids.SetPointIdsArrayName("OriginalPointIds"); ids.Update()
    surface_filter = vtk.vtkDataSetSurfaceFilter(); surface_filter.SetInputConnection(ids.GetOutputPort()); surface_filter.Update()
    surface = vtk.vtkPolyData(); surface.ShallowCopy(surface_filter.GetOutput())
    if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfPolys() == 0:
        raise ValueError("The VTK dataset has no extractable polygonal surface")
    return surface


def get_point_array(point_data, names):
    for name in names:
        array = point_data.GetArray(name)
        if array is not None: return name, array
    available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    raise ValueError(f"None of {names} found in point data; available={available}")


def get_normals(surface):
    normals_array = surface.GetPointData().GetArray("normals") or surface.GetPointData().GetNormals()
    if normals_array is None:
        normal_filter = vtk.vtkPolyDataNormals(); normal_filter.SetInputData(surface)
        normal_filter.ComputePointNormalsOn(); normal_filter.ComputeCellNormalsOff(); normal_filter.SplittingOff(); normal_filter.ConsistencyOn(); normal_filter.AutoOrientNormalsOn(); normal_filter.Update()
        normals_array = normal_filter.GetOutput().GetPointData().GetNormals()
    if normals_array is None: raise ValueError("Unable to read or compute point normals")
    normals = numpy_support.vtk_to_numpy(normals_array).astype(np.float32, copy=False)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0, 1.0, lengths)


def read_vtk(path, input_format="auto"):
    dataset = read_dataset(path)
    if input_format == "auto": input_format = "shapenet" if not isinstance(dataset, vtk.vtkPolyData) else "drivaer"
    surface = as_surface(dataset)
    coords = numpy_support.vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float32, copy=False)
    pressure_name, pressure_array = get_point_array(surface.GetPointData(), ("pressure", "p"))
    pressure = numpy_support.vtk_to_numpy(pressure_array).astype(np.float32, copy=False).reshape(-1, 1)
    normals = get_normals(surface)
    ids_array = surface.GetPointData().GetArray("OriginalPointIds")
    original_ids = np.arange(len(coords), dtype=np.int64) if ids_array is None else numpy_support.vtk_to_numpy(ids_array).astype(np.int64, copy=False)
    if not (len(coords) == len(normals) == len(pressure) == len(original_ids)):
        raise ValueError(f"Point arrays have inconsistent lengths in {path}")
    metadata = {"input_format": input_format, "source_dataset_type": dataset.GetClassName(), "source_num_points": dataset.GetNumberOfPoints(), "source_num_cells": dataset.GetNumberOfCells(), "surface_num_points": surface.GetNumberOfPoints(), "surface_num_cells": surface.GetNumberOfCells(), "pressure_field": pressure_name}
    return coords, pressure, normals, original_ids, surface, metadata


def mesh_graph(polydata):
    graph = nx.Graph(); graph.add_nodes_from(range(polydata.GetNumberOfPoints()))
    for cell_index in range(polydata.GetNumberOfCells()):
        ids = polydata.GetCell(cell_index).GetPointIds(); vertices = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        for i, source in enumerate(vertices):
            if len(vertices) > 1: graph.add_edge(source, vertices[(i + 1) % len(vertices)])
    return graph


def sharp_flags(polydata, angle):
    ids = _id_filter(); ids.SetInputData(polydata); ids.PointIdsOn(); ids.SetPointIdsArrayName("SegPointIds"); ids.Update()
    edges = vtk.vtkFeatureEdges(); edges.SetInputConnection(ids.GetOutputPort()); edges.FeatureEdgesOn(); edges.BoundaryEdgesOff(); edges.NonManifoldEdgesOn(); edges.SetFeatureAngle(float(angle)); edges.Update()
    flags = np.zeros(polydata.GetNumberOfPoints(), dtype=np.uint8)
    point_ids = edges.GetOutput().GetPointData().GetArray("SegPointIds")
    if point_ids is not None:
        values = numpy_support.vtk_to_numpy(point_ids).astype(np.int64, copy=False)
        if values.size: flags[np.unique(values)] = 1
    return flags


def create_seg_matrix(coords, polydata, threshold_angles=(8, 7, 6), min_graph_size_ratio=0.0001, max_segments=256):
    num_nodes = len(coords); graph = mesh_graph(polydata)
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]; min_size = int(min_graph_size_ratio * num_nodes)
    for angle in threshold_angles:
        flags = sharp_flags(polydata, angle); next_components = []
        for component in components:
            removed = [node for node in component if flags[node]]; smooth = component.copy(); smooth.remove_nodes_from(removed)
            node_sets = list(nx.connected_components(smooth)); node_sets.extend(nx.connected_components(component.subgraph(removed)))
            next_components.extend(component.subgraph(nodes).copy() for nodes in node_sets if len(nodes) > min_size)
        if not next_components: break
        components = next_components
    assigned = set().union(*(set(c.nodes) for c in components)) if components else set()
    for nodes in nx.connected_components(graph.subgraph(set(graph.nodes) - assigned)): components.append(graph.subgraph(nodes).copy())
    components.sort(key=len, reverse=True)
    if max_segments is not None and len(components) > max_segments:
        kept = components[:max_segments - 1]; overflow = set().union(*(set(c.nodes) for c in components[max_segments - 1:])); kept.append(graph.subgraph(overflow).copy()); components = kept
    rows, cols, values = [], [], []; labels = np.full(num_nodes, -1, dtype=np.int32)
    for segment_index, component in enumerate(components):
        nodes = np.fromiter(component.nodes, dtype=np.int64)
        if not len(nodes): continue
        rows.extend([segment_index] * len(nodes)); cols.extend(nodes.tolist()); values.extend([1.0 / len(nodes)] * len(nodes)); labels[nodes] = segment_index
    if np.any(labels < 0): raise RuntimeError("Geometry segmentation left nodes unassigned")
    matrix = csr_matrix((values, (rows, cols)), shape=(len(components), num_nodes), dtype=np.float32)
    return matrix, labels[:, None]


def sample_id(path):
    match = re.search(r"(\d+)$", path.stem)
    return str(int(match.group(1))) if match else path.stem


def compute_statistics(output_files):
    coords_min = np.full(3, np.inf, dtype=np.float64); coords_max = np.full(3, -np.inf, dtype=np.float64); count = 0; pressure_sum = 0.0; pressure_sum_sq = 0.0
    for file_path in output_files:
        with file_path.open("rb") as handle: sample = pickle.load(handle)
        coords = np.asarray(sample["coor"], dtype=np.float64); pressure = np.asarray(sample["pressure"], dtype=np.float64).ravel()
        coords_min = np.minimum(coords_min, coords.min(axis=0)); coords_max = np.maximum(coords_max, coords.max(axis=0)); pressure_sum += float(pressure.sum()); pressure_sum_sq += float(np.square(pressure).sum()); count += pressure.size
    mean = pressure_sum / count; std = np.sqrt(max(pressure_sum_sq / count - mean ** 2, 0.0)) or 1.0; ranges = coords_max - coords_min; ranges[ranges == 0] = 1.0
    return {"coords_min": coords_min[None, :].astype(np.float32), "coords_max": coords_max[None, :].astype(np.float32), "coords_range": ranges[None, :].astype(np.float32), "pressure_mean": np.asarray([mean], dtype=np.float32), "pressure_std": np.asarray([std], dtype=np.float32)}


def normalize_files(output_files, scalars):
    for file_path in output_files:
        with file_path.open("rb") as handle: sample = pickle.load(handle)
        sample["coor"] = ((sample["coor"] - scalars["coords_min"]) / scalars["coords_range"]).astype(np.float32, copy=False)
        sample["pressure"] = ((sample["pressure"] - scalars["pressure_mean"]) / scalars["pressure_std"]).astype(np.float32, copy=False)
        sample["features_6d"] = np.concatenate([sample["coor"], sample["normals"]], axis=1).astype(np.float32, copy=False)
        with file_path.open("wb") as handle: pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_folder(input_dir, output_dir=DEFAULT_OUTPUT, input_format="auto", pattern="*.vtk", threshold_angles=(8, 7, 6), min_graph_size_ratio=0.0001, max_segments=256):
    input_dir = Path(input_dir); output_dir = Path(output_dir); vtk_files = sorted(input_dir.glob(pattern))
    if not vtk_files: raise ValueError(f"No VTK files matching {pattern!r} found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.pkl"): old_file.unlink()
    output_files = []; started = time.monotonic()
    for index, vtk_file in enumerate(vtk_files, start=1):
        file_started = time.monotonic(); coords, pressure, normals, original_ids, surface, metadata = read_vtk(vtk_file, input_format)
        seg_matrix, labels = create_seg_matrix(coords, surface, threshold_angles, min_graph_size_ratio, max_segments)
        sample = {"coor": coords, "normals": normals, "pressure": pressure, "seg_matrix": seg_matrix, "node_cluster_flags": labels, "original_point_ids": original_ids, "features_6d": np.concatenate([coords, normals], axis=1).astype(np.float32), "source_file": vtk_file.name, "metadata": metadata}
        output_file = output_dir / f"{sample_id(vtk_file)}.pkl"
        with output_file.open("wb") as handle: pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output_files.append(output_file)
        LOGGER.info("[%d/%d] %s: %s, points=%d, segments=%d, %.2fs", index, len(vtk_files), vtk_file.name, metadata["source_dataset_type"], len(coords), seg_matrix.shape[0], time.monotonic() - file_started)
    scalars = compute_statistics(output_files)
    with (output_dir / "normalization_scalars.pkl").open("wb") as handle: pickle.dump(scalars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    normalize_files(output_files, scalars)
    LOGGER.info("Processed %d files into %s in %.1fs", len(output_files), output_dir, time.monotonic() - started)
    return scalars


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--format", choices=("auto", "drivaer", "shapenet"), default="auto")
    parser.add_argument("--pattern", default="*.vtk")
    parser.add_argument("--threshold-angles", type=float, nargs="+", default=[8, 7, 6])
    parser.add_argument("--min-graph-size-ratio", type=float, default=0.0001)
    parser.add_argument("--max-segments", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    process_folder(args.input_dir, args.output_dir, args.format, args.pattern, args.threshold_angles, args.min_graph_size_ratio, args.max_segments)


if __name__ == "__main__":
    configure_logging(); main()
