"""Plot cluster/slice/layer ablations from a tidy CSV file."""

import csv
from pathlib import Path


def plot_ablation_csv(path: Path, output: Path):
    import matplotlib.pyplot as plt

    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"model", "setting", "relative_l2"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Ablation CSV columns must include {sorted(required)}")
    figure, axis = plt.subplots(figsize=(7, 4.8))
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        axis.plot([float(row["setting"]) for row in selected], [float(row["relative_l2"]) for row in selected], marker="o", label=model)
    axis.set_xlabel("Ablation setting (clusters / slices / layers)")
    axis.set_ylabel("Relative L2 error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)
