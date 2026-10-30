"""Metrics for normalized DrivAerStar fields."""

import numpy as np
import torch

TARGET_ALIASES = {
    "surface_pressure": "y_spressure", "pressure": "y_spressure",
    "y_spressure": "y_spressure", "surface_wss": "y_wss",
    "wss": "y_wss", "y_wss": "y_wss",
    "volume_velocity": "y_velocity", "velocity": "y_velocity",
    "y_velocity": "y_velocity",
}

def target_field_name(predicted_feature_name):
    key = str(predicted_feature_name).strip().lower()
    if key not in TARGET_ALIASES:
        raise ValueError(f"Unsupported DrivAerStar target: {predicted_feature_name!r}")
    return TARGET_ALIASES[key]

def _mean_std(normalization_scalars, predicted_feature_name):
    field = target_field_name(predicted_feature_name)
    mean_key, std_key = f"{field}_mean", f"{field}_std"
    if field == "y_spressure" and mean_key not in normalization_scalars:
        mean_key, std_key = "pressure_mean", "pressure_std"
    if mean_key not in normalization_scalars or std_key not in normalization_scalars:
        raise KeyError(f"Missing {mean_key!r}/{std_key!r} for {predicted_feature_name!r}")
    return normalization_scalars[mean_key], normalization_scalars[std_key]

def denormalize_field(values, normalization_scalars, predicted_feature_name="surface_pressure"):
    """Undo dataset-wide, per-component mean/std standardization."""
    mean, std = _mean_std(normalization_scalars, predicted_feature_name)
    if isinstance(values, torch.Tensor):
        mean = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(std, device=values.device, dtype=values.dtype)
    else:
        mean = np.asarray(mean, dtype=np.asarray(values).dtype)
        std = np.asarray(std, dtype=np.asarray(values).dtype)
    return values * std + mean

def denormalize_pressure(values, normalization_scalars):
    return denormalize_field(values, normalization_scalars, "surface_pressure")

def compute_relative_error(predicted, target, normalization_scalars,
                           predicted_feature_name="surface_pressure", eps=1.0e-12):
    """Relative L2 error in physical units, for scalar or vector fields."""
    pred = denormalize_field(np.asarray(predicted), normalization_scalars,
                             predicted_feature_name)
    truth = denormalize_field(np.asarray(target), normalization_scalars,
                              predicted_feature_name)
    return float(np.linalg.norm(pred - truth) / max(float(np.linalg.norm(truth)), eps))

def compute_component_relative_errors(predicted, target, normalization_scalars,
                                      predicted_feature_name):
    """Physical-scale relative L2 for each output component."""
    pred = denormalize_field(np.asarray(predicted), normalization_scalars,
                             predicted_feature_name)
    truth = denormalize_field(np.asarray(target), normalization_scalars,
                              predicted_feature_name)
    if pred.ndim == 1:
        pred, truth = pred[:, None], truth[:, None]
    axes = tuple(range(pred.ndim - 1))
    num = np.linalg.norm(pred - truth, axis=axes)
    den = np.maximum(np.linalg.norm(truth, axis=axes), 1.0e-12)
    return np.asarray(num / den, dtype=np.float64)
