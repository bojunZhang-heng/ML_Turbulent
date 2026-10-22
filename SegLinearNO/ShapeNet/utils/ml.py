import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from utils.drag_coefficient import calculate_cd


SEGMENTED_MODELS = {
    "transolver_seg",
    "transolver_seg_v2",
    "SegLinearNO",
    "SegLinearNO_v2",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_amp(device):
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = GradScaler(
        device="cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    return amp_enabled, amp_dtype, scaler


def register_mem_hooks(model):
    """Register temporary CUDA-memory hooks on leaf modules."""
    hooks = []
    layer_stats = []

    def make_hook(name):
        def hook(module, inputs, output):
            torch.cuda.synchronize()
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            if isinstance(output, torch.Tensor):
                output_shape = tuple(output.shape)
                output_mb = output.numel() * output.element_size() / 1024**2
            elif isinstance(output, (tuple, list)):
                output_shape = "tuple"
                output_mb = sum(
                    item.numel() * item.element_size() / 1024**2
                    for item in output
                    if isinstance(item, torch.Tensor)
                )
            else:
                output_shape = type(output).__name__
                output_mb = 0.0
            layer_stats.append(
                {
                    "name": name,
                    "module": module.__class__.__name__,
                    "output_shape": output_shape,
                    "output_mb": output_mb,
                    "allocated_gb": allocated_gb,
                }
            )

        return hook

    for name, module in model.named_modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(make_hook(name)))
    return hooks, layer_stats


def remove_mem_hooks(hooks):
    for hook in hooks:
        hook.remove()


def print_mem_report(layer_stats, top_k=20):
    """Print the CUDA-memory report once, during the first training batch."""
    print("\n" + "=" * 100)
    print(f"{'layer':50s} {'module':18s} {'output_shape':22s} {'out_MB':>8s} {'alloc_GB':>9s}")
    print("-" * 100)
    for stat in layer_stats:
        print(
            f"{stat['name'][:50]:50s} {stat['module'][:18]:18s} "
            f"{str(stat['output_shape'])[:22]:22s} "
            f"{stat['output_mb']:>8.1f} {stat['allocated_gb']:>9.2f}"
        )
    print("\nLargest output tensors:")
    for stat in sorted(layer_stats, key=lambda item: -item["output_mb"])[:top_k]:
        print(
            f"  {stat['name'][:60]:60s} out={stat['output_mb']:>8.1f}MB "
            f"shape={stat['output_shape']} alloc={stat['allocated_gb']:.2f}GB"
        )
    print("=" * 100 + "\n")


def _unpack_batch(batch_data):
    if len(batch_data) == 6:
        return batch_data
    if len(batch_data) == 5:
        model_input, seg_matrix, target, surface_mask, sim_ids = batch_data
        return model_input, seg_matrix, target, surface_mask, sim_ids, [{} for _ in sim_ids]
    if len(batch_data) == 4:
        model_input, seg_matrix, target, sim_ids = batch_data
        surface_mask = torch.ones(target.shape[:-1], dtype=torch.bool)
        return model_input, seg_matrix, target, surface_mask, sim_ids, [{} for _ in sim_ids]
    raise ValueError(f"Unsupported batch format with {len(batch_data)} fields")


def _forward_model(model_name, model, model_input, seg_matrix):
    if model_name == "Transolver":
        output = model(model_input)
    elif model_name in SEGMENTED_MODELS:
        output = model((model_input, seg_matrix))
    elif model_name in ("LinearNO", "RopeLinearNO", "RopeLinearNO_v2"):
        output = model(model_input)
    else:
        raise ValueError(f"Model name {model_name} not supported")
    if output.ndim == 2 and model_input.shape[0] == 1:
        output = output.unsqueeze(0)
    if output.ndim != 3:
        raise ValueError(f"Model output must have shape (B, N, C), got {tuple(output.shape)}")
    return output


def _physics_loss(output, target, surface_mask, pressure_weight):
    if output.shape != target.shape:
        raise ValueError(
            f"Model output shape {tuple(output.shape)} != target shape {tuple(target.shape)}. "
            "ShapeNet-Car requires out_dim=4."
        )
    if output.shape[-1] == 1:
        pressure_loss = nn.functional.mse_loss(output, target)
        return pressure_weight * pressure_loss, output.new_zeros(()), pressure_loss
    if output.shape[-1] != 4:
        raise ValueError(f"Expected [u, v, w, p] output, got {output.shape[-1]} channels")
    if not surface_mask.any():
        raise ValueError("surface_mask contains no surface nodes")
    velocity_loss = nn.functional.mse_loss(output[..., :3], target[..., :3])
    pressure_loss = nn.functional.mse_loss(
        output[..., 3][surface_mask], target[..., 3][surface_mask]
    )
    total_loss = velocity_loss + pressure_weight * pressure_loss
    return total_loss, velocity_loss, pressure_loss


def _output_statistics(normalization_scalars, reference):
    if normalization_scalars is None:
        raise ValueError("normalization_scalars.pkl is required for physical-scale metrics")
    if "output_mean" not in normalization_scalars or "output_std" not in normalization_scalars:
        raise KeyError("normalization_scalars must contain output_mean and output_std")
    mean = torch.as_tensor(
        normalization_scalars["output_mean"], device=reference.device, dtype=reference.dtype
    )
    std = torch.as_tensor(
        normalization_scalars["output_std"], device=reference.device, dtype=reference.dtype
    )
    expected_shape = (reference.shape[-1],)
    if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
        raise ValueError(
            f"Normalization shape mismatch: mean={tuple(mean.shape)}, "
            f"std={tuple(std.shape)}, expected={expected_shape}"
        )
    return mean, std


def denormalize_output(normalized, normalization_scalars):
    mean, std = _output_statistics(normalization_scalars, normalized)
    return normalized * std + mean


def _relative_l2(predicted, target):
    denominator = torch.linalg.vector_norm(target).clamp_min(1e-12)
    return (torch.linalg.vector_norm(predicted - target) / denominator).item()


def _empty_metrics():
    return {
        "samples": 0,
        "velocity_relative_l2": 0.0,
        "pressure_relative_l2": 0.0,
        "cd_relative_error": 0.0,
        "velocity_rmse": 0.0,
        "pressure_rmse": 0.0,
    }


def _accumulate_physics_metrics(
    totals, output, target, surface_mask, metadata, normalization_scalars
):
    output = denormalize_output(output.float(), normalization_scalars)
    target = denormalize_output(target.float(), normalization_scalars)
    for batch_index in range(output.shape[0]):
        mask = surface_mask[batch_index]
        volume_mask = ~mask
        if not mask.any() or not volume_mask.any():
            raise ValueError("Each ShapeNet-Car sample needs both volume and surface nodes")

        pred_velocity = output[batch_index, volume_mask, :3]
        true_velocity = target[batch_index, volume_mask, :3]
        pred_pressure = output[batch_index, mask, 3]
        true_pressure = target[batch_index, mask, 3]
        totals["velocity_relative_l2"] += _relative_l2(pred_velocity, true_velocity)
        totals["pressure_relative_l2"] += _relative_l2(pred_pressure, true_pressure)
        totals["velocity_rmse"] += torch.sqrt(
            nn.functional.mse_loss(pred_velocity, true_velocity)
        ).item()
        totals["pressure_rmse"] += torch.sqrt(
            nn.functional.mse_loss(pred_pressure, true_pressure)
        ).item()

        pressure_path = metadata[batch_index].get("pressure_path")
        if not pressure_path:
            raise ValueError("Sample metadata does not contain pressure_path required for Cd")
        pred_cd = calculate_cd(
            pressure_path,
            pred_pressure.detach().cpu().numpy(),
            output[batch_index, mask, :3].detach().cpu().numpy(),
        )
        true_cd = calculate_cd(
            pressure_path,
            true_pressure.detach().cpu().numpy(),
            target[batch_index, mask, :3].detach().cpu().numpy(),
        )
        totals["cd_relative_error"] += abs(pred_cd - true_cd) / max(abs(true_cd), 1e-12)
        totals["samples"] += 1


def _average_metrics(totals):
    samples = totals["samples"]
    if samples == 0:
        raise ValueError("No samples were evaluated")
    return {key: value / samples for key, value in totals.items() if key != "samples"}


@torch.no_grad()
def validate(
    model_name,
    model,
    data_loader,
    normalization_scalars,
    device,
    pressure_weight=0.5,
    description="Validation",
    collect_outputs=False,
):
    model.eval()
    amp_enabled, amp_dtype, _ = configure_amp(device)
    loss_totals = {"loss": 0.0, "velocity_loss": 0.0, "pressure_loss": 0.0}
    physics_totals = _empty_metrics()
    predictions = {}
    targets = {}

    progress = tqdm(
        data_loader,
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for batch_data in progress:
        model_input, seg_matrix, target, surface_mask, sim_ids, metadata = _unpack_batch(batch_data)
        model_input = model_input.to(device)
        target = target.to(device)
        surface_mask = surface_mask.to(device)
        if model_name in SEGMENTED_MODELS:
            seg_matrix = seg_matrix.to(device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = _forward_model(model_name, model, model_input, seg_matrix)
            loss, velocity_loss, pressure_loss = _physics_loss(
                output, target, surface_mask, pressure_weight
            )
        loss_totals["loss"] += loss.item()
        loss_totals["velocity_loss"] += velocity_loss.item()
        loss_totals["pressure_loss"] += pressure_loss.item()
        _accumulate_physics_metrics(
            physics_totals, output, target, surface_mask, metadata, normalization_scalars
        )
        if collect_outputs:
            output_denormalized = denormalize_output(output.float(), normalization_scalars).cpu()
            target_denormalized = denormalize_output(target.float(), normalization_scalars).cpu()
            for index, sim_id in enumerate(sim_ids):
                predictions[sim_id] = output_denormalized[index]
                targets[sim_id] = target_denormalized[index]
        progress.set_postfix(loss=f"{loss.item():.4e}")
    batch_count = len(data_loader)
    if batch_count == 0:
        raise ValueError(f"{description} loader is empty")
    results = {key: value / batch_count for key, value in loss_totals.items()}
    results.update(_average_metrics(physics_totals))
    if collect_outputs:
        results["predictions"] = predictions
        results["targets"] = targets
    return results


def _print_metrics(prefix, results):
    print(f"  {prefix} normalized loss: {results['loss']:.6e}")
    print(f"  {prefix} velocity relative L2: {results['velocity_relative_l2']:.6f}")
    print(f"  {prefix} pressure relative L2: {results['pressure_relative_l2']:.6f}")
    print(f"  {prefix} Cd relative error: {results['cd_relative_error']:.6f}")
    print(f"  {prefix} velocity RMSE: {results['velocity_rmse']:.6e}")
    print(f"  {prefix} pressure RMSE: {results['pressure_rmse']:.6e}")


def train(
    model_name,
    model,
    train_loader,
    val_loader,
    normalization_scalars,
    args,
    num_epochs=100,
    eval_freq=10,
    save_path="models/best_model.pth",
    predicted_feature_name="output",
):
    device = get_device()
    pressure_weight = getattr(args, "pressure_weight", 0.5)
    print(f"Using device: {device}")
    amp_enabled, amp_dtype, scaler = configure_amp(device)
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=max(1, len(train_loader) * num_epochs),
        final_div_factor=1000.0,
    )
    os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
    history = {
        "train_loss": [],
        "train_velocity_loss": [],
        "train_pressure_loss": [],
        "val_loss": [],
        "val_velocity_relative_l2": [],
        "val_pressure_relative_l2": [],
        "val_cd_relative_error": [],
        "best_val_error": float("inf"),
    }
    memory_report_pending = device.type == "cuda"

    for epoch in range(num_epochs):
        model.train()
        train_totals = {"loss": 0.0, "velocity_loss": 0.0, "pressure_loss": 0.0}
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Train",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch_data in progress:
            model_input, seg_matrix, target, surface_mask, _, _ = _unpack_batch(batch_data)
            model_input = model_input.to(device)
            target = target.to(device)
            surface_mask = surface_mask.to(device)
            if model_name in SEGMENTED_MODELS:
                seg_matrix = seg_matrix.to(device)
            optimizer.zero_grad(set_to_none=True)

            memory_hooks, memory_stats = [], []
            if memory_report_pending:
                torch.cuda.reset_peak_memory_stats(device)
                memory_hooks, memory_stats = register_mem_hooks(model)
            try:
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    output = _forward_model(model_name, model, model_input, seg_matrix)
                    loss, velocity_loss, pressure_loss = _physics_loss(
                        output, target, surface_mask, pressure_weight
                    )
            except torch.OutOfMemoryError:
                if memory_report_pending:
                    print_mem_report(memory_stats)
                    remove_mem_hooks(memory_hooks)
                raise
            if memory_report_pending:
                print_mem_report(memory_stats)
                remove_mem_hooks(memory_hooks)
                memory_report_pending = False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_totals["loss"] += loss.item()
            train_totals["velocity_loss"] += velocity_loss.item()
            train_totals["pressure_loss"] += pressure_loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.3e}",
                velocity=f"{velocity_loss.item():.3e}",
                pressure=f"{pressure_loss.item():.3e}",
            )
        batch_count = len(train_loader)
        if batch_count == 0:
            raise ValueError("Training loader is empty")
        train_results = {key: value / batch_count for key, value in train_totals.items()}
        history["train_loss"].append(train_results["loss"])
        history["train_velocity_loss"].append(train_results["velocity_loss"])
        history["train_pressure_loss"].append(train_results["pressure_loss"])

        print(f"Epoch {epoch + 1}/{num_epochs} | LR={optimizer.param_groups[0]['lr']:.3e}")
        print(
            f"  Train normalized loss: {train_results['loss']:.6e} "
            f"(velocity={train_results['velocity_loss']:.6e}, "
            f"pressure={train_results['pressure_loss']:.6e})"
        )

        if epoch % eval_freq == 0 or epoch == num_epochs - 1:
            val_results = validate(
                model_name,
                model,
                val_loader,
                normalization_scalars,
                device,
                pressure_weight=pressure_weight,
                description=f"Epoch {epoch + 1}/{num_epochs} - Val",
            )
            history["val_loss"].append(val_results["loss"])
            history["val_velocity_relative_l2"].append(val_results["velocity_relative_l2"])
            history["val_pressure_relative_l2"].append(val_results["pressure_relative_l2"])
            history["val_cd_relative_error"].append(val_results["cd_relative_error"])
            selection_error = (
                val_results["velocity_relative_l2"] + val_results["pressure_relative_l2"]
            )
            _print_metrics("Val", val_results)
            if selection_error < history["best_val_error"]:
                history["best_val_error"] = selection_error
                torch.save(model.state_dict(), save_path)
                print(f"  New best model saved: {save_path}")

    print(f"Training completed. Best velocity+pressure relative L2: {history['best_val_error']:.6f}")
    return history


def test(
    model_name,
    model,
    test_loader,
    normalization_scalars,
    model_path="models/best_model.pth",
    predicted_feature_name="output",
    pressure_weight=0.5,
):
    device = get_device()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: model file not found at {model_path}; testing current weights")
    model = model.float().to(device)
    results = validate(
        model_name,
        model,
        test_loader,
        normalization_scalars,
        device,
        pressure_weight=pressure_weight,
        description="Testing",
        collect_outputs=True,
    )
    print("Test Results (relative errors use denormalized physical values):")
    _print_metrics("Test", results)
    return results
