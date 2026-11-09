# DrivAerML model comparison

This directory is synchronized from the DrivAerStar postprocessing framework
and supports `surface_pressure`, `surface_wss`, and `volume_velocity`.

Edit `postprocess/runs.example.yml`, then run from `RETO++/DrivAerML`:

```bash
python postprocess/evaluate_models.py \
  --runs-file postprocess/runs.example.yml \
  --data-path /data/zhangbojun/DrivAerML/pklFile_10k \
  --output-dir postprocess_results/all_models \
  --device cuda --amp \
  2>&1 | tee postprocess_all_models.log
```

All models in one comparison must predict the same field and use the same test
cases. Use separate output directories when comparing pressure and velocity.

