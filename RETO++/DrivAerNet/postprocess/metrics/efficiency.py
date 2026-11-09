"""Small helpers for model efficiency reporting."""

from pathlib import Path


def model_size_summary(model, checkpoint: Path):
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint_size_mb": Path(checkpoint).stat().st_size / 1024**2,
    }
