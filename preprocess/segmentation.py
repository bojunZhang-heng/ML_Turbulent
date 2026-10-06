from __future__ import annotations

import networkx as nx
import numpy as np
import vtk
from scipy.sparse import csr_matrix
from vtk.util import numpy_support


def _id_filter():
    filter_type = getattr(vtk, "vtkIdFilter", None)
    if filter_type is None:
        filter_type = vtk.vtkGenerateIds
    return filter_type()


def mark_sharp_edges(polydata, angle_threshold):
    id_filter = _id_filter()
    id_filter.SetInputData(polydata)
    id_filter.PointIdsOn()
    id_filter.SetPointIdsArrayName("SegPointIds")
    id_filter.Update()

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputConnection(id_filter.GetOutputPort())
    feature_edges.FeatureEdgesOn()
    feature_edges.BoundaryEdgesOff()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.SetFeatureAngle(float(angle_threshold))
    feature_edges.Update()

    flags = np.zeros(polydata.GetNumberOfPoints(), dtype=np.uint8)
    edges = feature_edges.GetOutput()
    point_ids = edges.GetPointData().GetArray("SegPointIds")
    if point_ids is None:
        return flags

    original_ids = numpy_support.vtk_to_numpy(point_ids).astype(np.int64, copy=False)
    if original_ids.size:
        flags[np.unique(original_ids)] = 1
    return flags


def mesh_graph(polydata):
    graph = nx.Graph()
    graph.add_nodes_from(range(polydata.GetNumberOfPoints()))

    for cell_index in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(cell_index)
        point_ids = cell.GetPointIds()
        count = point_ids.GetNumberOfIds()
        if count < 2:
            continue
        vertices = [point_ids.GetId(index) for index in range(count)]
        for index, source in enumerate(vertices):
            target = vertices[(index + 1) % count]
            graph.add_edge(source, target)
    return graph


def _split_component(graph, sharp_flags, min_graph_size):
    sharp_nodes = [node for node in graph.nodes if sharp_flags[node]]
    smooth_graph = graph.copy()
    smooth_graph.remove_nodes_from(sharp_nodes)

    components = list(nx.connected_components(smooth_graph))
    sharp_graph = graph.subgraph(sharp_nodes)
    components.extend(nx.connected_components(sharp_graph))

    return [
        graph.subgraph(nodes).copy()
        for nodes in components
        if len(nodes) > min_graph_size
    ]


def _include_unassigned_nodes(subgraphs, graph):
    assigned = set()
    for subgraph in subgraphs:
        assigned.update(subgraph.nodes)

    unassigned = set(graph.nodes) - assigned
    if not unassigned:
        return subgraphs

    remainder = graph.subgraph(unassigned)
    for nodes in nx.connected_components(remainder):
        subgraphs.append(graph.subgraph(nodes).copy())
    return subgraphs


def _limit_segments(subgraphs, graph, max_segments):
    subgraphs = sorted(subgraphs, key=len, reverse=True)
    if max_segments is None or len(subgraphs) <= max_segments:
        return subgraphs

    retained = subgraphs[: max_segments - 1]
    overflow_nodes = set()
    for subgraph in subgraphs[max_segments - 1 :]:
        overflow_nodes.update(subgraph.nodes)
    retained.append(graph.subgraph(overflow_nodes).copy())
    return retained


def create_seg_matrix(
    coords,
    polydata,
    threshold_angles=(8, 7, 6),
    min_graph_size_ratio=0.0001,
    max_segments=256,
):
    num_nodes = len(coords)
    if num_nodes != polydata.GetNumberOfPoints():
        raise ValueError("Coordinate count does not match surface point count")

    graph = mesh_graph(polydata)
    subgraphs = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)]
    min_graph_size = int(min_graph_size_ratio * num_nodes)

    for threshold_angle in threshold_angles:
        sharp_flags = mark_sharp_edges(polydata, threshold_angle)
        next_subgraphs = []
        for subgraph in subgraphs:
            next_subgraphs.extend(
                _split_component(subgraph, sharp_flags, min_graph_size)
            )
        if not next_subgraphs:
            break
        subgraphs = next_subgraphs

    subgraphs = _include_unassigned_nodes(subgraphs, graph)
    subgraphs = _limit_segments(subgraphs, graph, max_segments)

    row_indices = []
    column_indices = []
    values = []
    labels = np.full(num_nodes, -1, dtype=np.int32)

    for segment_index, subgraph in enumerate(subgraphs):
        nodes = np.fromiter(subgraph.nodes, dtype=np.int64)
        if nodes.size == 0:
            continue
        row_indices.extend([segment_index] * nodes.size)
        column_indices.extend(nodes.tolist())
        values.extend([1.0 / nodes.size] * nodes.size)
        labels[nodes] = segment_index

    if np.any(labels < 0):
        raise RuntimeError("Segmentation left surface nodes unassigned")

    seg_matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(subgraphs), num_nodes),
        dtype=np.float32,
    )
    return seg_matrix, labels.reshape(-1, 1)
