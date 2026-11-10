"""Unified inference and physical-unit metric collection."""

from __future__ import annotations

import time
import inspect
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from dataloader import create_data_loaders

from data_split import display_case_id, resolve_test_data
from metrics.efficiency import model_size_summary
from metrics.field_metrics import FieldAccumulator, case_metrics, summarize_cases
from metrics.statistics import bootstrap_mean_ci
from model_registry import build_model, load_checkpoint

OUTPUT_COMPONENTS = {
    "surface_pressure": 1, "pressure": 1, "y_spressure": 1,
    "surface_wss": 3, "wss": 3, "y_wss": 3,
    "volume_velocity": 3, "velocity": 3, "y_velocity": 3,
    "p_and_v": 4, "combined": 4, "output": 4,
}

SEGMENTED_MODELS = {
    "Transolver_seg", "Transolver_seg_v2", "transolver_seg",
    "transolver_seg_v2", "SegLinearNO", "SegLinearNO_v2",
}
# ShapeNet training calls every non-segmented model as ``model(model_input)``.
# In particular, RETO_plus/v2 therefore derive their positional coordinates
# from the normalized input. Passing reconstructed physical coordinates here
# would change the learned function at evaluation time. A future experiment
# that is also trained with explicit raw coordinates can still opt in through
# ``postprocess.forward_mode: raw_coords`` in its YAML.
RAW_COORD_MODELS = set()
SERIALIZED_MODELS = {"LinearNO_se", "SATO", "Transolver_seg_ser"}


def configure_amp(device):
    enabled = device.type == "cuda"
    dtype = (
        torch.bfloat16
        if enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return enabled, dtype


def _normalization_keys(task):
    task = str(task).strip().lower()
    if task in {"surface_pressure", "pressure", "y_spressure"}:
        return (("y_spressure_mean", "y_spressure_std"),
                ("pressure_mean", "pressure_std"))
    if task in {"surface_wss", "wss", "y_wss"}:
        return (("y_wss_mean", "y_wss_std"),)
    if task in {"volume_velocity", "velocity", "y_velocity"}:
        return (("y_velocity_mean", "y_velocity_std"),)
    if task in {"p_and_v", "combined", "output"}:
        return (("output_mean", "output_std"),)
    raise ValueError(f"Unsupported task {task!r}")


def denormalize_field(values, scalars, task):
    for mean_key, std_key in _normalization_keys(task):
        if mean_key in scalars and std_key in scalars:
            values = np.asarray(values)
            return (
                values * np.asarray(scalars[std_key], dtype=values.dtype)
                + np.asarray(scalars[mean_key], dtype=values.dtype)
            )
    expected = [f"{mean}/{std}" for mean, std in _normalization_keys(task)]
    raise KeyError(f"Missing normalization statistics for {task!r}; expected {expected}")


def _load_scalars(data_source):
    if isinstance(data_source, dict):
        by_vehicle = {}
        for vehicle, root in data_source.items():
            path = Path(root) / "normalization_scalars.pkl"
            with path.open("rb") as handle:
                by_vehicle[vehicle] = pickle.load(handle)
        return {"schema_version": "multi-vehicle-normalization-v1", "by_vehicle": by_vehicle}
    roots = [data_source] if isinstance(data_source, (str, Path)) else list(data_source)
    for root in roots:
        path = Path(root) / "normalization_scalars.pkl"
        if path.is_file():
            with path.open("rb") as handle:
                return pickle.load(handle)
    return None


def _coordinate_stats(scalars, task):
    task = str(task).lower()
    if task in {"surface_pressure", "pressure", "y_spressure", "surface_wss", "wss", "y_wss"}:
        candidates = (("surface_coords_min", "surface_coords_range"),
                      ("coords_min", "coords_range"))
    elif task in {"volume_velocity", "velocity", "y_velocity"}:
        candidates = (("volume_coords_min", "volume_coords_range"),
                      ("coords_min", "coords_range"))
    else:
        candidates = (("coords_min", "coords_range"),)
    for min_key, range_key in candidates:
        if min_key in scalars and range_key in scalars:
            return np.asarray(scalars[min_key]), np.asarray(scalars[range_key])
    return None


def _raw_from_normalized(coords, scalars, task, surface_mask=None):
    coords_np = coords.detach().cpu().numpy()
    if str(task).lower() in {"p_and_v", "combined", "output"} and surface_mask is not None:
        raw = coords_np.copy()
        mask = surface_mask.detach().cpu().numpy().astype(bool)
        for is_surface, prefix in ((True, "surface_coords"), (False, "volume_coords")):
            min_key, range_key = f"{prefix}_min", f"{prefix}_range"
            if min_key in scalars and range_key in scalars:
                selected = mask if is_surface else ~mask
                raw[selected] = raw[selected] * np.asarray(scalars[range_key]) + np.asarray(scalars[min_key])
        return torch.as_tensor(raw, dtype=coords.dtype)
    stats = _coordinate_stats(scalars, task)
    if stats is None:
        return coords.detach().cpu()
    minimum, values_range = stats
    return torch.as_tensor(coords_np * values_range + minimum, dtype=coords.dtype)


def _unpack_batch(batch, scalars, task):
    """Adapt DrivAerStar/ML/Net five-tuples and ShapeNet six-tuples."""
    if len(batch) == 5:
        coorf, raw_coords, seg_matrix, target, sim_ids = batch
        return coorf, raw_coords, seg_matrix, target, sim_ids
    if len(batch) == 6:
        coorf, seg_matrix, target, surface_mask, sim_ids, _metadata = batch
        raw_batches = []
        for index, sim_id in enumerate(sim_ids):
            sample_scalars = _sample_scalars(scalars, sim_id)
            raw_batches.append(
                _raw_from_normalized(
                    coorf[index, :, :3], sample_scalars, task,
                    surface_mask[index] if surface_mask is not None else None,
                )
            )
        return coorf, torch.stack(raw_batches), seg_matrix, target, sim_ids
    raise ValueError(f"Unsupported DataLoader batch with {len(batch)} fields")


def _serialization_indices(coords, device):
    try:
        from SATO_dataset.Dataset import pos_to_order_inverse_index
    except ImportError as error:
        raise ImportError(
            "This serialized model requires SATO_dataset in the training directory"
        ) from error
    return pos_to_order_inverse_index(coords, device, tensor=True)


def forward_model(model_name, model, coorf, raw_coords, seg_matrix, cfg):
    mode = str(cfg.get("forward_mode", "auto")).lower()
    if mode == "auto":
        if model_name in SEGMENTED_MODELS:
            mode = "segmented"
        elif model_name in RAW_COORD_MODELS:
            mode = "raw_coords"
        elif model_name in SERIALIZED_MODELS:
            mode = "serialized"
        else:
            mode = "direct"
    if mode == "direct":
        output = model(coorf)
    elif mode == "raw_coords":
        output = model(coorf, rope_coords=raw_coords)
    elif mode == "segmented":
        output = model((coorf, seg_matrix))
    elif mode == "serialized":
        order, inverse = _serialization_indices(raw_coords, coorf.device)
        if model_name == "Transolver_seg_ser":
            output = model(coorf, seg_matrix, order, inverse)
        else:
            output = model(coorf, order, inverse)
    else:
        raise ValueError(
            f"Unknown forward_mode={mode!r}; choose auto/direct/raw_coords/segmented/serialized"
        )
    if output.ndim == 2 and coorf.shape[0] == 1:
        output = output.unsqueeze(0)
    return output


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
    if count == 4:
        return ["velocity_x", "velocity_y", "velocity_z", "pressure"]
    if count == 1:
        return ["pressure" if "pressure" in task else "scalar"]
    prefix = "wss" if "wss" in task else "velocity"
    return [f"{prefix}_{axis}" for axis in "xyz"] if count == 3 else [f"component_{i}" for i in range(count)]


def _print_run_setup(label, model_name, task, config_path, checkpoint, device,
                     amp_enabled, test_ids, data_source, parameter_count):
    print(f"\n{'=' * 24} RUN START: {label} {'=' * 24}", flush=True)
    print(f"model_name          : {model_name}", flush=True)
    print(f"task                : {task}", flush=True)
    print(f"config              : {Path(config_path).resolve()}", flush=True)
    print(f"checkpoint          : {Path(checkpoint).resolve()}", flush=True)
    print(f"data_source         : {data_source}", flush=True)
    print(f"device              : {device}", flush=True)
    print(f"amp_enabled         : {amp_enabled}", flush=True)
    print(f"parameter_count     : {parameter_count}", flush=True)
    print(f"test_case_count     : {len(test_ids)}", flush=True)
    print(f"test_case_ids       : {test_ids}", flush=True)


def _print_case_metrics(label, row):
    values = "  ".join(
        f"{key}={value:.8g}" if isinstance(value, (int, float)) and value is not None
        else f"{key}={value}"
        for key, value in row.items()
    )
    print(f"[case][{label}] {values}", flush=True)


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
    loader_kwargs = dict(
        batch_size=1, train_index=[], val_index=[], test_index=test_ids,
        shuffle=False, predicted_feature_name=task,
    )
    if "coordinate_only" in inspect.signature(create_data_loaders).parameters:
        loader_kwargs["coordinate_only"] = True
    _, _, loader, scalars = create_data_loaders(data_source, **loader_kwargs)
    if scalars is None:
        scalars = _load_scalars(data_source)
    if scalars is None:
        raise FileNotFoundError("normalization_scalars.pkl is required")

    device = select_device(args.device)
    model = build_model(model_name, cfg)
    load_checkpoint(model, checkpoint, expected_model_name=model_name)
    model = model.float().to(device).eval()
    amp_available, amp_dtype = configure_amp(device)
    amp_enabled = bool(args.amp and amp_available)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _print_run_setup(
        label, model_name, task, config_path, checkpoint, device, amp_enabled,
        test_ids, data_source, parameter_count,
    )
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
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            return forward_model(model_name, model, coorf, raw_coords, seg_matrix, cfg)

    prediction_dir = Path(args.output_dir) / "predictions"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{label} ({model_name})", unit="case"):
            coorf, raw_coords, seg_matrix, target, sim_ids = _unpack_batch(
                batch, scalars, task
            )
            expected_space_dim = int(cfg.get("space_dim", coorf.shape[-1]))
            coorf = coorf[..., :expected_space_dim].to(device)
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
                if getattr(args, "print_per_case", True):
                    _print_case_metrics(label, row)
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
    print(f"\n[run complete] {label}: cases={len(rows)}, points={total_points}, "
          f"inference_seconds={elapsed:.8g}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows, selected_case
