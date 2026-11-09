# DrivAerNet model comparison

This directory is synchronized from the DrivAerStar postprocessing framework
and adapted to DrivAerNet pressure PKLs and multi-vehicle datasets.

Edit `postprocess/runs.example.yml`, then run from `RETO++/DrivAerNet`:

```bash
python postprocess/evaluate_models.py \
  --runs-file postprocess/runs.example.yml \
  --data-path /data/zhangbojun/DrivAerNet/pklFile \
  --output-dir postprocess_results/all_models \
  --device cuda --amp \
  2>&1 | tee postprocess_all_models.log
```

All YAML files must reconstruct the same test split. Repeated `--run` remains
available for quick comparisons.

