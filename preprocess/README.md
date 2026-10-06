# Unified VTK preprocessing

`preprocess/process_data.py` converts both supported VTK datasets into one
pickle format used by `SegLinearNO/driver/dataloader.py`.

## Supported inputs

### DrivAer surface VTK

```text
/Users/zhangbojun/Desktop/E_S_WW_WM_small/*.vtk
```

- VTK type: `vtkPolyData`
- pressure field: `p`
- surface points are used directly

### ShapeNet car VTK

```text
/Users/zhangbojun/Downloads/ShapeNet_dataset/vtk/*.vtk
```

- VTK type: `vtkUnstructuredGrid`
- cells: tetrahedra
- pressure field: `pressure`
- the exterior triangular surface is extracted automatically

The script accepts either pressure field name (`p` or `pressure`) and can
automatically detect the VTK dataset type with `--format auto`.

## Running the preprocessing

The output directory is overwritten on every run: existing `.pkl` files are
removed before new samples are generated.

### DrivAer

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  /Users/zhangbojun/ML_Turbulent/preprocess/process_data.py \
  --input-dir /Users/zhangbojun/Desktop/E_S_WW_WM_small \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/data \
  --format drivaer
```

### ShapeNet

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  /Users/zhangbojun/ML_Turbulent/preprocess/process_data.py \
  --input-dir /Users/zhangbojun/Downloads/ShapeNet_dataset/vtk \
  --output-dir /Users/zhangbojun/ML_Turbulent/preprocess/data \
  --format shapenet
```

For a quick single-file test:

```bash
/opt/anaconda3/envs/pgd-no/bin/python \
  /Users/zhangbojun/ML_Turbulent/preprocess/process_data.py \
  --input-dir /Users/zhangbojun/Downloads/ShapeNet_dataset/vtk \
  --pattern 'car_00000.vtk' \
  --output-dir /tmp/shapenet_check
```

## Geometry partition parameters

The default geometry segmentation uses feature-edge angles `[8, 7, 6]`:

```bash
--threshold-angles 8 7 6
```

Other options:

```bash
--min-graph-size-ratio 0.0001
--max-segments 256
```

`max-segments` is an upper bound, not a request for an exact number of
regions. For example, if natural segmentation produces 178 regions,
`--max-segments 128` merges overflow regions, while `--max-segments 256`
leaves 178 regions unchanged.

## Output format

Every input VTK file produces one pickle file with the same schema:

```python
{
    "coor": normalized_coordinates,        # (N, 3), float32
    "normals": unit_normals,               # (N, 3), float32
    "features_6d": coordinates_and_normals,# (N, 6), float32
    "pressure": normalized_pressure,       # (N, 1), float32
    "seg_matrix": geometry_matrix,         # (S, N), scipy CSR matrix
    "node_cluster_flags": segment_labels,  # (N, 1), int32
    "original_point_ids": source_ids,      # (N,), int64
    "source_file": input_filename,
    "metadata": source_mesh_metadata,
}
```

The model uses:

```text
input:    features_6d = [x, y, z, nx, ny, nz]
geometry: seg_matrix
target:   pressure
```

`original_point_ids` maps ShapeNet exterior-surface points to source volume
mesh points. For native `vtkPolyData`, it is the surface point index.

## Normalization

`normalization_scalars.pkl` is written beside the sample files and contains:

```python
{
    "coords_min": ...,
    "coords_max": ...,
    "coords_range": ...,
    "pressure_mean": ...,
    "pressure_std": ...,
}
```

Coordinates use global min-max normalization:

```python
coor_normalized = (coor - coords_min) / coords_range
```

Pressure uses global mean/std standardization:

```python
pressure_normalized = (pressure - pressure_mean) / pressure_std
```

Statistics are computed across all VTK files in the current input folder.

## DataLoader usage

The unified pickle files can be passed directly to the SegLinearNO DataLoader:

```python
from dataloader import create_data_loaders

train_loader, val_loader, test_loader, _ = create_data_loaders(
    "/Users/zhangbojun/ML_Turbulent/preprocess/data",
    batch_size=1,
    train_index=[1, 2, 3],
    val_index=[4],
    test_index=[5],
    predicted_feature_name="pressure",
)
```

Each batch has four elements:

```python
features_6d, seg_matrix, pressure, sample_id = batch
```

Because different surface meshes can have different `N` and `S`, use
`batch_size=1` unless padding and masking are implemented.

## Example output sizes

```text
DrivAer E_S_WW_WM_001.vtk:
    features_6d: (482634, 6)
    seg_matrix:  (256, 482634)

ShapeNet car_00000.vtk:
    features_6d: (3585, 6)
    seg_matrix:  (178, 3585)
```
