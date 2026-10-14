from pathlib import Path

import numpy as np
import vtk
from scipy.spatial import ConvexHull
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _read_grid(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pressure VTK file not found: {path}")
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(reader.GetOutput())
    if grid.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not read a non-empty VTK grid: {path}")
    return grid


def _surface_polydata(grid):
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(grid)
    surface_filter.Update()
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(surface_filter.GetOutput())
    return polydata


def _add_point_array(dataset, values, name):
    vtk_array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    vtk_array.SetName(name)
    dataset.GetPointData().AddArray(vtk_array)


def _cell_areas(polydata):
    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(polydata)
    cell_size.ComputeAreaOn()
    cell_size.ComputeLengthOff()
    cell_size.ComputeVolumeOff()
    cell_size.Update()
    area_array = cell_size.GetOutput().GetCellData().GetArray("Area")
    if area_array is None:
        raise ValueError("VTK could not compute surface-cell areas")
    return vtk_to_numpy(area_array).astype(np.float64, copy=False)


def _cell_normals(polydata):
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(polydata)
    normal_filter.ComputeCellNormalsOn()
    normal_filter.ComputePointNormalsOff()
    normal_filter.ConsistencyOn()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.SplittingOff()
    normal_filter.Update()
    normals = normal_filter.GetOutput().GetCellData().GetNormals()
    if normals is None:
        raise ValueError("VTK could not compute surface-cell normals")
    return vtk_to_numpy(normals).astype(np.float64, copy=False)


def _point_to_cell(polydata, values, name):
    values = np.asarray(values, dtype=np.float64)
    vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    vtk_array.SetName(name)
    polydata.GetPointData().AddArray(vtk_array)
    converter = vtk.vtkPointDataToCellData()
    converter.SetInputData(polydata)
    converter.PassPointDataOn()
    converter.Update()
    cell_values = converter.GetOutput().GetCellData().GetArray(name)
    if cell_values is None:
        raise ValueError(f"VTK could not convert {name} from points to cells")
    return vtk_to_numpy(cell_values).astype(np.float64, copy=False)


def _velocity_gradient(polydata, velocity):
    velocity = np.asarray(velocity, dtype=np.float64)
    vtk_velocity = numpy_to_vtk(np.ascontiguousarray(velocity), deep=True)
    vtk_velocity.SetName("predicted_velocity")
    polydata.GetPointData().AddArray(vtk_velocity)

    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(polydata)
    gradient_filter.SetInputScalars(
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "predicted_velocity"
    )
    gradient_filter.SetResultArrayName("velocity_gradient")
    gradient_filter.Update()
    gradient = gradient_filter.GetOutput().GetPointData().GetArray("velocity_gradient")
    if gradient is None:
        raise ValueError("VTK could not compute the surface velocity gradient")
    point_gradient = vtk_to_numpy(gradient).reshape(-1, 3, 3)
    return _point_to_cell(polydata, point_gradient.reshape(-1, 9), "velocity_gradient")


def calculate_cd(
    pressure_vtk_path,
    pressure,
    surface_velocity,
    flow_speed=20.0,
    air_density=0.3,
    dynamic_viscosity=1.8e-5,
    flow_axis=2,
):
    """Compute drag coefficient from denormalized surface fields.

    The defaults match the reference LinearNO ShapeNet-Car implementation.
    ``pressure`` and ``surface_velocity`` must follow the pressure-grid point
    order saved by ``process_mlcfd.py``.
    """
    grid = _read_grid(pressure_vtk_path)
    if len(pressure) != grid.GetNumberOfPoints():
        raise ValueError(
            f"Pressure length {len(pressure)} does not match VTK points "
            f"{grid.GetNumberOfPoints()}"
        )
    if np.asarray(surface_velocity).shape != (grid.GetNumberOfPoints(), 3):
        raise ValueError(
            "surface_velocity must have shape "
            f"({grid.GetNumberOfPoints()}, 3), got {np.asarray(surface_velocity).shape}"
        )

    _add_point_array(grid, np.asarray(pressure).reshape(-1, 1), "input_pressure")
    _add_point_array(grid, np.asarray(surface_velocity), "input_velocity")
    polydata = _surface_polydata(grid)
    surface_pressure_array = polydata.GetPointData().GetArray("input_pressure")
    surface_velocity_array = polydata.GetPointData().GetArray("input_velocity")
    if surface_pressure_array is None or surface_velocity_array is None:
        raise ValueError("VTK did not preserve input fields during surface extraction")
    surface_pressure = vtk_to_numpy(surface_pressure_array).reshape(-1)
    surface_velocity = vtk_to_numpy(surface_velocity_array).reshape(-1, 3)

    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64, copy=False)
    projected_axes = [axis for axis in range(3) if axis != flow_axis]
    reference_area = float(ConvexHull(points[:, projected_axes]).volume)
    if reference_area <= 0.0:
        raise ValueError("Projected frontal area is zero")

    areas = _cell_areas(polydata)
    normals = _cell_normals(polydata)
    cell_pressure = _point_to_cell(
        polydata, surface_pressure.reshape(-1, 1), "pressure"
    ).reshape(-1)
    velocity_gradient = _velocity_gradient(polydata, surface_velocity).reshape(-1, 3, 3)

    pressure_drag = -np.sum(cell_pressure * areas * normals[:, flow_axis])
    normal_velocity_gradient = velocity_gradient[:, flow_axis, flow_axis]
    viscous_drag = -dynamic_viscosity * np.sum(
        normal_velocity_gradient * areas * normals[:, flow_axis]
    )
    drag_force = pressure_drag + viscous_drag
    return float(2.0 * drag_force / (air_density * flow_speed**2 * reference_area))
