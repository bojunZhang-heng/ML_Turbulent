# ShapeNet model comparison

This directory is synchronized from the DrivAerStar postprocessing framework
and adapted to the local ShapeNet DataLoader and model implementations.

## Compare many models

Edit `postprocess/runs.example.yml` so every entry points to the exact YAML and
checkpoint used for training, then run from `RETO++/ShapeNet`:

```bash
python postprocess/evaluate_models.py \
  --runs-file postprocess/runs.example.yml \
  --data-path /data/zhangbojun/mlcfd_data/pkl_data_v2 \
  --output-dir postprocess_results/all_models \
  --device cuda --amp \
  2>&1 | tee postprocess_all_models.log
```

The compared runs must use the same task and test split. ShapeNet supports
`surface_pressure`, `volume_velocity`, and `p_and_v` when the PKL schema
contains the corresponding fields.

For a future custom model, set `model_module`, `model_class`, `model_kwargs`
and optionally `forward_mode` in its YAML; see the generic section in
`README_EVALUATION.md`.

