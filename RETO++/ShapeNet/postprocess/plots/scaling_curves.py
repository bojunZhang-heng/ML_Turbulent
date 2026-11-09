"""Plot point-count scaling from a CSV benchmark."""

import csv
from pathlib import Path


def plot_scaling_csv(path: Path, output_dir: Path):
    import matplotlib.pyplot as plt

    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"model", "points", "time_ms", "memory_gb"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Scaling CSV columns must include {sorted(required)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for field, ylabel, filename in (
        ("time_ms", "Inference time (ms)", "scaling_time.png"),
        ("memory_gb", "Peak CUDA memory (GiB)", "scaling_memory.png"),
    ):
        figure, axis = plt.subplots(figsize=(6.5, 4.8))
        for model in sorted({row["model"] for row in rows}):
            selected = sorted((row for row in rows if row["model"] == model), key=lambda row: int(row["points"]))
            axis.plot([int(row["points"]) for row in selected], [float(row[field]) for row in selected], marker="o", label=model)
        axis.set_xlabel("Input mesh points")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=220)
        plt.close(figure)
