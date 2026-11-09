"""Visualize externally exported query/key/cluster/slice weights.

NPZ format: ``coords`` (N,3), plus any number of arrays shaped (N,) or (N,K).
For (N,K), ``--attention-index`` selects K.
"""

from pathlib import Path

import numpy as np


def plot_attention_npz(path: Path, output_dir: Path, index=0):
    import matplotlib.pyplot as plt

    archive = np.load(path)
    if "coords" not in archive:
        raise ValueError("Attention NPZ must contain coords with shape (N,3)")
    coords = archive["coords"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in archive.files:
        if key == "coords":
            continue
        values = np.asarray(archive[key])
        if values.ndim == 2:
            if not 0 <= index < values.shape[1]:
                raise IndexError(f"attention index {index} outside {key} shape {values.shape}")
            values = values[:, index]
        if values.shape != (coords.shape[0],):
            raise ValueError(f"{key} must have shape (N,) or (N,K), got {values.shape}")
        figure = plt.figure(figsize=(6.5, 5))
        axis = figure.add_subplot(111, projection="3d")
        artist = axis.scatter(*coords.T, c=values, s=3, cmap="viridis", rasterized=True)
        axis.set_axis_off()
        axis.set_title(f"{key}, channel {index}")
        figure.colorbar(artist, ax=axis, fraction=0.03)
        figure.tight_layout()
        figure.savefig(output_dir / f"attention_{key}.png", dpi=220)
        plt.close(figure)
