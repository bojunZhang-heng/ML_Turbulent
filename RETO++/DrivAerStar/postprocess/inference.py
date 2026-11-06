"""Unified inference and physical-unit metric collection."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from SATO_dataset.Dataset import pos_to_order_inverse_index
from dataloader import create_data_loaders
from utils.metric import denormalize_field
from utils.ml_SATO import SEGMENTED_MODELS, SERIALIZED_MODELS, configure_amp, forward_model, serialization_coordinates

from data_split import display_case_id, resolve_test_data
from metrics.efficiency import model_size_summary
from metrics.field_metrics import FieldAccumulator, case_metrics, summarize_cases
from metrics.statistics import bootstrap_mean_ci
from model_registry import build_model, load_checkpoint

OUTPUT_COMPONENTS = {
    "surface_pressure": 1, "pressure": 1, "y_spressure": 1,
    "surface_wss": 3, "wss": 3, "y_wss": 3,
    "volume_velocity": 3, "velocity": 3, "y_velocity": 3,
}


def select_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _sample_scalars(all_scalars, sim_id):
    if isinstance(all_scalars, dict) and "by_vehicle" in all_scalars:
        return all_scalars["by_vehicle"][sim_id[0]]
    return all_scalars


def _component_names(task, count):
    if count == 1:
        return ["pressure" if "pressure" in task else "scalar"]
    prefix = "wss" if "wss" in task else "velocity"
    return [f"{prefix}_{axis}" for axis in "xyz"] if count == 3 else [f"component_{i}" for i in range(count)]


def evaluate_run(label, config_path, checkpoint, raw_cfg, cfg, args):
    model_name = str(cfg.get("model_name", ""))
    task = str(cfg.get("predicted_feature_name", "surface_pressure")).lower()
    if task not in OUTPUT_COMPONENTS:
        raise ValueError(f"Unsupported task {task!r}")
    cfg = dict(cfg)
    cfg["out_dim"] = OUTPUT_COMPONENTS[task]
    data_source, test_ids = resolve_test_data(raw_cfg, cfg, args.data_path)
    if args.max_test_samples:
        test_ids = test_ids[:args.max_test_samples]
    if not test_ids:
        raise ValueError("Reconstructed test split is empty")
    _, _, loader, scalars = create_data_loaders(
        data_source, batch_size=1, train_index=[], val_index=[], test_index=test_ids,
        shuffle=False, predicted_feature_name=task, coordinate_only=True,
    )
    if scalars is None:
        raise FileNotFoundError("normalization_scalars.pkl is required")

    device = select_device(args.device)
    model = build_model(model_name, cfg)
    load_checkpoint(model, checkpoint, expected_model_name=model_name)
    model = model.float().to(device).eval()
    amp_available, amp_dtype, _ = configure_amp(device)
    amp_enabled = bool(args.amp and amp_available)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    accumulator = FieldAccumulator(OUTPUT_COMPONENTS[task])
    magnitude_accumulator = FieldAccumulator(1)
    direction_cosine = direction_angle = 0.0
    direction_count = 0
    rows, selected_case = [], None
    elapsed = normalized_sse = 0.0
    normalized_count = total_points = 0
    warmed_up = False

    def invoke(coorf, raw_coords, seg_matrix):
        order = inverse = None
        if model_name in SERIALIZED_MODELS:
            positions = serialization_coordinates(model_name, coorf, raw_coords)
            order, inverse = pos_to_order_inverse_index(positions, device, tensor=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            return forward_model(model_name, model, coorf, raw_coords, seg_matrix, order, inverse)

    prediction_dir = Path(args.output_dir) / "predictions"
    with torch.inference_mode():
        for coorf, raw_coords, seg_matrix, target, sim_ids in tqdm(loader, desc=f"{label} ({model_name})", unit="case"):
            coorf = coorf[..., :3].to(device)
            raw_coords_device = raw_coords.to(device)
            target_device = target.to(device)
            if model_name in SEGMENTED_MODELS:
                seg_matrix = seg_matrix.to(device)
            if not warmed_up:
                _ = invoke(coorf, raw_coords_device, seg_matrix)
                synchronize(device)
                warmed_up = True
            synchronize(device)
            start = time.perf_counter()
            output = invoke(coorf, raw_coords_device, seg_matrix)
            synchronize(device)
            elapsed += time.perf_counter() - start

            normalized_error = output.detach().float() - target_device.float()
            normalized_sse += float(normalized_error.square().sum().cpu())
            normalized_count += normalized_error.numel()
            total_points += target_device.shape[1]
            output_np = output.detach().float().cpu().numpy()
            target_np = target_device.detach().float().cpu().numpy()
            raw_np = raw_coords.numpy()

            for index, sim_id in enumerate(sim_ids):
                case_id = display_case_id(sim_id)
                sample_scalars = _sample_scalars(scalars, sim_id)
                prediction = np.asarray(denormalize_field(output_np[index], sample_scalars, task), float)
                truth = np.asarray(denormalize_field(target_np[index], sample_scalars, task), float)
                accumulator.update(prediction, truth)
                row = {"case_id": case_id, "points": int(truth.shape[0]), **case_metrics(prediction, truth)}
                rows.append(row)
                if prediction.shape[-1] == 3:
                    pred_mag = np.linalg.norm(prediction, axis=-1, keepdims=True)
                    true_mag = np.linalg.norm(truth, axis=-1, keepdims=True)
                    magnitude_accumulator.update(pred_mag, true_mag)
                    valid = (pred_mag[:, 0] > 1e-12) & (true_mag[:, 0] > 1e-12)
                    cosine = np.sum(prediction[valid] * truth[valid], axis=-1) / (pred_mag[valid, 0] * true_mag[valid, 0])
                    cosine = np.clip(cosine, -1, 1)
                    direction_cosine += float(cosine.sum())
                    direction_angle += float(np.degrees(np.arccos(cosine)).sum())
                    direction_count += cosine.size
                if selected_case is None and (args.plot_case_id is None or args.plot_case_id == case_id):
                    selected_case = {"case_id": case_id, "coords": raw_np[index], "truth": truth, "prediction": prediction}
                if args.save_predictions:
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    safe_case = case_id.replace("/", "_")
                    np.savez_compressed(prediction_dir / f"{label}_{safe_case}.npz", coords=raw_np[index], truth=truth, prediction=prediction)

    physical = accumulator.finish(_component_names(task, accumulator.components))
    physical["case_equal_weighted"] = summarize_cases(rows)
    physical["relative_l2_bootstrap_ci95"] = bootstrap_mean_ci(
        [row["relative_l2"] for row in rows], args.bootstrap_samples, seed=args.seed
    )
    if accumulator.components == 3:
        physical["vector_magnitude"] = magnitude_accumulator.finish(["magnitude"])["overall"]
        physical["vector_direction"] = {
            "mean_cosine_similarity": direction_cosine / direction_count if direction_count else None,
            "mean_angular_error_degrees": direction_angle / direction_count if direction_count else None,
            "valid_points": int(direction_count),
        }
    efficiency = {
        **model_size_summary(model, checkpoint), "device": str(device), "amp_enabled": amp_enabled,
        "inference_seconds": elapsed, "milliseconds_per_case": elapsed * 1000 / len(rows),
        "points_per_second": total_points / max(elapsed, 1e-12),
    }
    if device.type == "cuda":
        efficiency["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    result = {
        "label": label, "model": model_name, "task": task,
        "config": str(Path(config_path).resolve()), "checkpoint": str(Path(checkpoint).resolve()),
        "test_case_ids": [row["case_id"] for row in rows], "test_cases": len(rows),
        "test_points": int(total_points), "normalized_mse": normalized_sse / max(normalized_count, 1),
        "physical_metrics": physical, "efficiency": efficiency,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows, selected_case
