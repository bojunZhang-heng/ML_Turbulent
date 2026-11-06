"""Aerodynamic coefficient accuracy and ranking plots."""

from pathlib import Path

import numpy as np


def plot_coefficients(data: dict, metrics: dict, output_dir: Path):
    import matplotlib.pyplot as plt

    labels = [key for key in data if key not in {"case_id", "truth"}]
    truth = np.asarray(data["truth"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5.5))
    lower, upper = float(truth.min()), float(truth.max())
    for label in labels:
        axis.scatter(truth, data[label], s=20, alpha=0.65, label=f"{label}, rho={metrics[label]['spearman_rho']:.3f}")
        lower = min(lower, float(np.min(data[label])))
        upper = max(upper, float(np.max(data[label])))
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axis.set_xlabel("True coefficient")
    axis.set_ylabel("Predicted coefficient")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "coefficient_scatter.png", dpi=220)
    plt.close(figure)

    order = np.argsort(truth)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(truth[order], label="Ground truth", color="black", linewidth=2)
    for label in labels:
        axis.plot(np.asarray(data[label])[order], label=label, alpha=0.8)
    axis.set_xlabel("Cases sorted by true coefficient")
    axis.set_ylabel("Coefficient")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "coefficient_rank.png", dpi=220)
    plt.close(figure)
