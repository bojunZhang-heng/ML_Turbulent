"""Transolver-style accuracy/efficiency bubble plot."""

from pathlib import Path


def plot_efficiency_bubble(summaries: dict, output: Path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    for label, summary in summaries.items():
        efficiency = summary["efficiency"]
        error = summary["physical_metrics"]["case_equal_weighted"]["relative_l2"]["mean"]
        memory = efficiency.get("peak_cuda_memory_gb")
        # CPU/MPS runs have no comparable CUDA peak; place them on y=0 and annotate.
        memory_value = float(memory) if memory is not None else 0.0
        parameters_million = efficiency["parameter_count"] / 1e6
        area = 80.0 + 45.0 * max(parameters_million, 0.1) ** 0.5
        axis.scatter(efficiency["milliseconds_per_case"], memory_value, s=area, alpha=0.65)
        axis.annotate(
            f"{label}\nL2={error:.4g}",
            (efficiency["milliseconds_per_case"], memory_value),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    axis.set_xlabel("Inference time per case (ms)")
    axis.set_ylabel("Peak CUDA memory (GiB; 0 for CPU/MPS)")
    axis.set_title("Bubble area represents parameter count")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
