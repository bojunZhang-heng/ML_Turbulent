"""Ground-truth, prediction and absolute-error point-cloud maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _scalar_view(values):
    values = np.asarray(values)
    if values.ndim == 1 or values.shape[-1] == 1:
        return values.reshape(-1), "field"
    return np.linalg.norm(values, axis=-1), "magnitude"


def plot_field_comparison(case_id, coords, truth, predictions, output: Path, point_size=2.0):
    import matplotlib.pyplot as plt

    coords = np.asarray(coords)
    truth_scalar, suffix = _scalar_view(truth)
    prediction_scalars = {label: _scalar_view(value)[0] for label, value in predictions.items()}
    field_values = np.concatenate([truth_scalar, *prediction_scalars.values()])
    field_min, field_max = np.nanpercentile(field_values, [1, 99])
    errors = {label: np.abs(value - truth_scalar) for label, value in prediction_scalars.items()}
    error_max = max(float(np.nanpercentile(value, 99)) for value in errors.values())
    rows, columns = 2, len(predictions) + 1
    figure = plt.figure(figsize=(4.2 * columns, 7.5))

    def scatter(panel, values, title, minimum, maximum, cmap):
        axis = figure.add_subplot(rows, columns, panel, projection="3d")
        artist = axis.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], c=values, s=point_size,
            cmap=cmap, vmin=minimum, vmax=maximum, rasterized=True,
        )
        axis.set_title(title)
        axis.set_axis_off()
        axis.view_init(elev=18, azim=-55)
        figure.colorbar(artist, ax=axis, fraction=0.025, pad=0.01)

    scatter(1, truth_scalar, f"Ground truth ({suffix})", field_min, field_max, "viridis")
    for column, (label, prediction) in enumerate(prediction_scalars.items(), start=2):
        scatter(column, prediction, f"{label} prediction", field_min, field_max, "viridis")
    # Bottom-left is deliberately a textual legend, then one shared-scale error map/model.
    text_axis = figure.add_subplot(rows, columns, columns + 1)
    text_axis.axis("off")
    text_axis.text(0.5, 0.5, f"Case: {case_id}\nAbsolute error\n(common color scale)", ha="center", va="center")
    for column, (label, error) in enumerate(errors.items(), start=2):
        scatter(columns + column, error, f"{label} absolute error", 0.0, error_max, "magma")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
