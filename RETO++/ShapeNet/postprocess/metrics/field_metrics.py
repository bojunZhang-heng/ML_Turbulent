"""Physical-unit scalar/vector field metrics.

Relative L2 is reported both globally (every node has equal weight) and per
case (every vehicle has equal weight).  This distinction matters when meshes
have different point counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _flatten(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return values.reshape(-1, values.shape[-1])


def case_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    """Return metrics for one case in physical units."""
    prediction, target = _flatten(prediction), _flatten(target)
    error = prediction - target
    absolute = np.abs(error)
    sse = float(np.square(error).sum())
    target_energy = float(np.square(target).sum())
    centered = target - target.mean()
    sst = float(np.square(centered).sum())
    pearson = np.corrcoef(prediction.ravel(), target.ravel())[0, 1]
    return {
        "relative_l2": _finite(np.sqrt(sse / max(target_energy, 1.0e-24))),
        "r2": _finite(1.0 - sse / sst) if sst > 1.0e-24 else None,
        "mae": _finite(absolute.mean()),
        "rmse": _finite(np.sqrt(np.square(error).mean())),
        "bias": _finite(error.mean()),
        "pearson_r": _finite(pearson),
        "max_absolute_error": _finite(absolute.max()),
    }


@dataclass
class FieldAccumulator:
    """Streaming sufficient statistics for a complete test set."""

    components: int

    def __post_init__(self):
        c = self.components
        self.count = np.zeros(c, np.int64)
        self.sum_y = np.zeros(c)
        self.sum_p = np.zeros(c)
        self.sum_y2 = np.zeros(c)
        self.sum_p2 = np.zeros(c)
        self.sum_yp = np.zeros(c)
        self.sum_error = np.zeros(c)
        self.sum_error2 = np.zeros(c)
        self.sum_absolute = np.zeros(c)
        self.max_absolute = np.zeros(c)
        self.minimum = np.full(c, np.inf)
        self.maximum = np.full(c, -np.inf)

    def update(self, prediction, target):
        prediction, target = _flatten(prediction), _flatten(target)
        if prediction.shape != target.shape or prediction.shape[1] != self.components:
            raise ValueError(f"Incompatible field shapes: {prediction.shape}, {target.shape}")
        error = prediction - target
        absolute = np.abs(error)
        self.count += target.shape[0]
        self.sum_y += target.sum(0)
        self.sum_p += prediction.sum(0)
        self.sum_y2 += np.square(target).sum(0)
        self.sum_p2 += np.square(prediction).sum(0)
        self.sum_yp += (prediction * target).sum(0)
        self.sum_error += error.sum(0)
        self.sum_error2 += np.square(error).sum(0)
        self.sum_absolute += absolute.sum(0)
        self.max_absolute = np.maximum(self.max_absolute, absolute.max(0))
        self.minimum = np.minimum(self.minimum, target.min(0))
        self.maximum = np.maximum(self.maximum, target.max(0))

    def _arrays(self):
        count = np.maximum(self.count.astype(float), 1.0)
        sst = self.sum_y2 - np.square(self.sum_y) / count
        var_p = self.sum_p2 - np.square(self.sum_p) / count
        covariance = self.sum_yp - self.sum_y * self.sum_p / count
        rmse = np.sqrt(self.sum_error2 / count)
        target_std = np.sqrt(np.maximum(sst / count, 0.0))
        return {
            "relative_l2": np.sqrt(self.sum_error2 / np.maximum(self.sum_y2, 1e-24)),
            "r2": np.where(sst > 1e-24, 1.0 - self.sum_error2 / sst, np.nan),
            "mae": self.sum_absolute / count,
            "rmse": rmse,
            "nrmse_std": rmse / np.maximum(target_std, 1e-12),
            "nrmse_range": rmse / np.maximum(self.maximum - self.minimum, 1e-12),
            "bias": self.sum_error / count,
            "pearson_r": covariance / np.maximum(np.sqrt(np.maximum(sst, 0) * np.maximum(var_p, 0)), 1e-24),
            "max_absolute_error": self.max_absolute,
        }

    def finish(self, component_names: list[str]):
        arrays = self._arrays()
        components = {
            name: {key: _finite(values[index]) for key, values in arrays.items()}
            for index, name in enumerate(component_names)
        }
        # Flattened overall metrics need scalar sufficient statistics.
        overall = FieldAccumulator(1)
        overall.count[0] = self.count.sum()
        for name in ("sum_y", "sum_p", "sum_y2", "sum_p2", "sum_yp",
                     "sum_error", "sum_error2", "sum_absolute"):
            getattr(overall, name)[0] = getattr(self, name).sum()
        overall.max_absolute[0] = self.max_absolute.max()
        overall.minimum[0] = self.minimum.min()
        overall.maximum[0] = self.maximum.max()
        return {
            "overall": {key: _finite(value[0]) for key, value in overall._arrays().items()},
            "components": components,
        }


def summarize_cases(rows: list[dict]) -> dict:
    """Equal-case mean/median/P95/worst summary."""
    result = {}
    for metric in ("relative_l2", "r2", "mae", "rmse", "bias", "pearson_r", "max_absolute_error"):
        values = np.asarray([row[metric] for row in rows if row.get(metric) is not None], float)
        if not values.size:
            result[metric] = {key: None for key in ("mean", "std", "median", "p95", "worst")}
            continue
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "worst": float(values.min() if metric in {"r2", "pearson_r"} else values.max()),
        }
    return result
