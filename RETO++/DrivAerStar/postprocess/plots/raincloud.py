"""CANO-style relative-L2 distribution plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ==================== PAPER FIGURE TUNING: START ====================
# Edit this block when adapting the figure to a journal or thesis template.
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 12             # Base text size.
AXIS_LABEL_SIZE = 13       # Axis-title size.
TICK_LABEL_SIZE = 11       # Model names and y-axis tick labels.

FIGURE_MIN_WIDTH = 7.0     # Minimum figure width in inches.
WIDTH_PER_MODEL = 1.35     # Extra horizontal space for each model.
FIGURE_HEIGHT = 5.2        # Figure height in inches.

VIOLIN_WIDTH = 0.75
VIOLIN_ALPHA = 0.35
BOX_WIDTH = 0.16
BOX_LINE_WIDTH = 1.0
MEDIAN_LINE_WIDTH = 1.5
SCATTER_SIZE = 13
SCATTER_ALPHA = 0.55
X_LABEL_ROTATION = 15
GRID_ALPHA = 0.25
OUTPUT_DPI = 300           # Use 300-600 dpi for publication raster figures.
# ===================== PAPER FIGURE TUNING: END =====================


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
    paper_style = {
        "font.family": "serif",
        "font.serif": [FONT_FAMILY],
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
    with plt.rc_context(paper_style):
        figure_width = max(FIGURE_MIN_WIDTH, WIDTH_PER_MODEL * len(labels))
        figure, axis = plt.subplots(figsize=(figure_width, FIGURE_HEIGHT))
        positions = np.arange(1, len(labels) + 1)
        violin = axis.violinplot(
            violin_values,
            positions=positions,
            showextrema=False,
            widths=VIOLIN_WIDTH,
        )
        colors = plt.get_cmap("tab10").colors
        for index, body in enumerate(violin["bodies"]):
            body.set_facecolor(colors[index % len(colors)])
            body.set_edgecolor("none")
            body.set_alpha(VIOLIN_ALPHA)
            # Keep the left half: violin + box + jitter forms a raincloud.
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.minimum(vertices[:, 0], positions[index])
        axis.boxplot(
            values,
            positions=positions,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": BOX_LINE_WIDTH,
            },
            whiskerprops={"linewidth": BOX_LINE_WIDTH},
            capprops={"linewidth": BOX_LINE_WIDTH},
            medianprops={"color": "black", "linewidth": MEDIAN_LINE_WIDTH},
        )
        rng = np.random.default_rng(seed)
        for index, model_values in enumerate(values):
            x = positions[index] + 0.10 + rng.uniform(0.0, 0.18, model_values.size)
            axis.scatter(
                x,
                model_values,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                color=colors[index % len(colors)],
                linewidths=0,
            )
        axis.set_xticks(
            positions,
            labels,
            rotation=X_LABEL_ROTATION,
            ha="right",
        )
        axis.set_ylabel("Per-case relative L2 error")
        axis.grid(axis="y", alpha=GRID_ALPHA)
        figure.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=OUTPUT_DPI, bbox_inches="tight")
        plt.close(figure)
