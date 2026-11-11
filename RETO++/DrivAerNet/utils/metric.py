"""Physical-unit metrics for normalized DrivAerNet fields."""

import numpy as np
import torch


TARGET_ALIASES = {
    "surface_pressure": "y_spressure",
    "pressure": "y_spressure",
    "y_spressure": "y_spressure",
    "volume_velocity": "y_velocity",
    "velocity": "y_velocity",
    "y_velocity": "y_velocity",
}


def target_field_name(predicted_feature_name):
    key = str(predicted_feature_name).strip().lower()
    if key not in TARGET_ALIASES:
        raise ValueError(f"Unsupported DrivAerNet target: {predicted_feature_name!r}")
    return TARGET_ALIASES[key]


def _mean_std(normalization_scalars, predicted_feature_name):
    field = target_field_name(predicted_feature_name)
    candidates = [(f"{field}_mean", f"{field}_std")]
    if field == "y_spressure":
        candidates.append(("pressure_mean", "pressure_std"))
    elif field == "y_velocity":
        candidates.append(("velocity_mean", "velocity_std"))
    for mean_key, std_key in candidates:
        if mean_key in normalization_scalars and std_key in normalization_scalars:
            return normalization_scalars[mean_key], normalization_scalars[std_key]
    expected = ", ".join(f"{mean}/{std}" for mean, std in candidates)
    raise KeyError(
        f"Missing normalization statistics for {predicted_feature_name!r}; "
        f"expected one of: {expected}"
    )


def denormalize_field(
    values, normalization_scalars, predicted_feature_name="surface_pressure"
):
    """Undo dataset-wide, component-wise mean/std standardization."""
    mean, std = _mean_std(normalization_scalars, predicted_feature_name)
    if isinstance(values, torch.Tensor):
        mean = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(std, device=values.device, dtype=values.dtype)
    else:
        values = np.asarray(values)
        mean = np.asarray(mean, dtype=values.dtype)
        std = np.asarray(std, dtype=values.dtype)
    return values * std + mean


def denormalize_pressure(values, normalization_scalars):
    return denormalize_field(values, normalization_scalars, "surface_pressure")


def compute_relative_error(
    predicted,
    target,
    normalization_scalars,
    predicted_feature_name="surface_pressure",
    eps=1.0e-12,
):
    """Return relative L2 error after restoring physical units."""
    pred = denormalize_field(
        np.asarray(predicted), normalization_scalars, predicted_feature_name
    )
    truth = denormalize_field(
        np.asarray(target), normalization_scalars, predicted_feature_name
    )
    denominator = max(float(np.linalg.norm(truth)), eps)
    return float(np.linalg.norm(pred - truth) / denominator)


def compute_component_relative_errors(
    predicted, target, normalization_scalars, predicted_feature_name
):
    """Return physical-scale relative L2 error for each output component."""
    pred = denormalize_field(
        np.asarray(predicted), normalization_scalars, predicted_feature_name
    )
    truth = denormalize_field(
        np.asarray(target), normalization_scalars, predicted_feature_name
    )
    if pred.ndim == 1:
        pred, truth = pred[:, None], truth[:, None]
    axes = tuple(range(pred.ndim - 1))
    numerator = np.linalg.norm(pred - truth, axis=axes)
    denominator = np.maximum(np.linalg.norm(truth, axis=axes), 1.0e-12)
    return np.asarray(numerator / denominator, dtype=np.float64)
