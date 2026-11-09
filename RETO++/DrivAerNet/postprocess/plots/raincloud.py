"""CANO-style relative-L2 distribution plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_error_raincloud(values_by_model: dict[str, list[float]], output: Path, seed=42):
    import matplotlib.pyplot as plt

    labels = list(values_by_model)
    values = [np.asarray(values_by_model[label], float) for label in labels]
    violin_values = []
    for model_values in values:
        if model_values.size == 1:
            violin_values.append(np.array([model_values[0] - 1e-12, model_values[0] + 1e-12]))
        elif np.ptp(model_values) == 0:
            offsets = np.linspace(-1e-12, 1e-12, model_values.size)
            violin_values.append(model_values + offsets)
        else:
            violin_values.append(model_values)
    figure, axis = plt.subplots(figsize=(max(7, 1.35 * len(labels)), 5.2))
    positions = np.arange(1, len(labels) + 1)
    violin = axis.violinplot(
        violin_values, positions=positions, showextrema=False, widths=0.75
    )
    colors = plt.get_cmap("tab10").colors
    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(colors[index % len(colors)])
        body.set_edgecolor("none")
        body.set_alpha(0.35)
        # Keep the left half: violin + box + jitter forms a raincloud.
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.minimum(vertices[:, 0], positions[index])
    axis.boxplot(
        values, positions=positions, widths=0.16, patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    rng = np.random.default_rng(seed)
    for index, model_values in enumerate(values):
        x = positions[index] + 0.10 + rng.uniform(0.0, 0.18, model_values.size)
        axis.scatter(x, model_values, s=13, alpha=0.55, color=colors[index % len(colors)], linewidths=0)
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Per-case relative L2 error")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
