"""Aerodynamic coefficient accuracy and ranking metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _rankdata(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rho(first, second):
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


def read_coefficient_csv(path: Path, labels: list[str]):
    """Read case_id,true_coefficient,<model labels...> and compute metrics."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"case_id", "true_coefficient", *labels}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Coefficient CSV is missing columns: {sorted(missing)}")
    truth = np.asarray([float(row["true_coefficient"]) for row in rows])
    result = {"case_id": [row["case_id"] for row in rows], "truth": truth}
    metrics = {}
    for label in labels:
        prediction = np.asarray([float(row[label]) for row in rows])
        result[label] = prediction
        metrics[label] = {
            "relative_l2": float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-12)),
            "mae": float(np.abs(prediction - truth).mean()),
            "spearman_rho": spearman_rho(prediction, truth),
        }
    return result, metrics
