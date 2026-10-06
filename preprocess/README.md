# ShapeNet car preprocessing

This standalone pipeline converts ShapeNet legacy VTK unstructured volume meshes
into the per-sample pickle format used by the PGD-NO driver models.

The ShapeNet files differ from the original DrivAer files in two important ways:

- they contain a `vtkUnstructuredGrid` of tetrahedra instead of `vtkPolyData`;
- their pressure point-data array is named `pressure` instead of `p`.

The pipeline extracts the exterior triangular surface, preserves the original volume
point IDs, computes PGD-style geometry segments, normalizes coordinates with global
min/max values, and standardizes pressure with global mean/std values.

Run all files with the default paths:

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_data.py
```

Run only the example file:

```bash
/opt/anaconda3/envs/pgd-no/bin/python preprocess/process_data.py \
  --pattern 'car_00000.vtk' \
  --max-segments 128
```

Outputs are written to `preprocess/data/`. Every run replaces existing `.pkl`
files in that output directory.

Each sample contains:

- `coor`: normalized exterior-surface coordinates, shape `(N, 3)`;
- `normals`: unit surface normals, shape `(N, 3)`;
- `features_6d`: `[x, y, z, nx, ny, nz]`, shape `(N, 6)`;
- `pressure`: standardized surface pressure, shape `(N, 1)`;
- `seg_matrix`: CSR geometry aggregation matrix, shape `(S, N)`;
- `node_cluster_flags`: segment label per surface point, shape `(N, 1)`;
- `original_point_ids`: mapping from surface points to source volume-grid points;
- `metadata` and `source_file`: source-mesh traceability.
