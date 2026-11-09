"""Uncertainty summaries for repeated/case-level model evaluation."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values, samples=2000, confidence=0.95, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": None, "lower": None, "upper": None}
    if samples <= 0:
        return {"mean": float(values.mean()), "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": samples,
    }


def paired_bootstrap_difference(first, second, samples=2000, seed=42):
    """CI of mean(first - second); useful for paired test-case errors."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape:
        raise ValueError("Paired arrays must have identical shapes")
    difference = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(samples, difference.size))
    means = difference[indices].mean(1)
    return {
        "mean_difference": float(difference.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "probability_first_better": float((means < 0).mean()),
    }
