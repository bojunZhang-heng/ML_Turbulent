import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from utils.metric import (
    compute_component_relative_errors,
    compute_relative_error,
    denormalize_field,
)
from SATO_dataset.Dataset import pos_to_order_inverse_index

SEGMENTED_MODELS = {
    "Transolver_seg",
    "Transolver_seg_v2",
    "SegLinearNO",
    "SegLinearNO_v2",
}
LINEAR_MODELS = {"LinearNO", "CANO", "RETO"}
LINEAR_MODEL_SE = {"SATO"}
RAW_COORD_ROPE_MODELS = {
    "RETO_plus",
    "RETO_plus_v2",
}
SEGMENTED_SER = {"Transolver_seg_ser"}
SERIALIZED_MODELS = LINEAR_MODEL_SE | SEGMENTED_SER


def checkpoint_paths(save_path):
    """Return the val-loss and relative-L2 checkpoint paths."""
    base_path = Path(save_path)
    return {
        "val_loss": base_path.with_name(
            f"{base_path.stem}_val_loss{base_path.suffix}"
        ),
        "relative_error": base_path.with_name(
            f"{base_path.stem}_relative_error{base_path.suffix}"
        ),
    }


def serialization_coordinates(model_name, normalized_coords, raw_coords):
    """Return normalized coordinates for the remaining serialized models."""
    return normalized_coords


def batch_relative_error(outputs, targets, sim_ids, normalization_scalars,
                         predicted_feature_name):
    """Compute physical-scale error with the correct scalars for each vehicle."""
    outputs_np = outputs.detach().float().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    by_vehicle = (
        normalization_scalars.get("by_vehicle")
        if isinstance(normalization_scalars, dict)
        else None
    )
    if by_vehicle is None:
        return compute_relative_error(
            outputs_np, targets_np, normalization_scalars, predicted_feature_name
        )
    errors = []
    for batch_index, composite_id in enumerate(sim_ids):
        if not isinstance(composite_id, tuple) or len(composite_id) != 2:
            raise TypeError(
                "Multi-vehicle metrics require (vehicle_name, sample_id) IDs"
            )
        vehicle, _ = composite_id
        errors.append(
            compute_relative_error(
                outputs_np[batch_index:batch_index + 1],
                targets_np[batch_index:batch_index + 1],
                by_vehicle[vehicle],
                predicted_feature_name,
            )
        )
    return float(np.mean(errors))


def batch_physical_fields(
    outputs, targets, sim_ids, normalization_scalars, predicted_feature_name
):
    """Yield predictions and targets restored with each sample's scalars."""
    outputs_np = outputs.detach().float().cpu().numpy()
    targets_np = targets.detach().float().cpu().numpy()
    by_vehicle = (
        normalization_scalars.get("by_vehicle")
        if isinstance(normalization_scalars, dict)
        else None
    )
    for batch_index, sim_id in enumerate(sim_ids):
        scalars = normalization_scalars
        if by_vehicle is not None:
            if not isinstance(sim_id, tuple) or len(sim_id) != 2:
                raise TypeError(
                    "Multi-vehicle metrics require (vehicle_name, sample_id) IDs"
                )
            scalars = by_vehicle[sim_id[0]]
        predicted = denormalize_field(
            outputs_np[batch_index], scalars, predicted_feature_name
        )
        target = denormalize_field(
            targets_np[batch_index], scalars, predicted_feature_name
        )
        yield np.asarray(predicted, dtype=np.float64), np.asarray(
            target, dtype=np.float64
        )


def r2_from_sufficient_statistics(count, target_sum, target_sum_sq, error_sum_sq):
    """Compute component R2 without retaining all point predictions."""
    target_sum = np.asarray(target_sum, dtype=np.float64)
    target_sum_sq = np.asarray(target_sum_sq, dtype=np.float64)
    error_sum_sq = np.asarray(error_sum_sq, dtype=np.float64)
    total_sum_sq = target_sum_sq - np.square(target_sum) / max(int(count), 1)
    r2 = np.full_like(total_sum_sq, np.nan, dtype=np.float64)
    valid = total_sum_sq > np.finfo(np.float64).eps
    r2[valid] = 1.0 - error_sum_sq[valid] / total_sum_sq[valid]
    return r2


def forward_model(model_name, model, coorf, raw_coords, seg_matrix, order, inverse):
    if model_name == "Transolver":
        return model(coorf)
    if model_name in SEGMENTED_MODELS:
        return model((coorf, seg_matrix))
    if model_name in LINEAR_MODELS:
        return model(coorf)
    if model_name in RAW_COORD_ROPE_MODELS:
        return model(coorf[..., :3], rope_coords=raw_coords)
    if model_name in LINEAR_MODEL_SE:
        return model(coorf, order, inverse)
    if model_name in SEGMENTED_SER:
        return model(coorf, seg_matrix, order, inverse)
    raise ValueError(f"Model name {model_name} not supported")


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


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def register_mem_hooks(model):
    """Register temporary CUDA-memory hooks on leaf modules."""
    hooks = []
    layer_stats = []

    def make_hook(name):
        def hook(module, inputs, output):
            torch.cuda.synchronize()
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3

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

            if inputs and isinstance(inputs[0], torch.Tensor):
                input_shape = tuple(inputs[0].shape)
            else:
                input_shape = "-"

            layer_stats.append(
                {
                    "name": name,
                    "module": module.__class__.__name__,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "output_mb": output_mb,
                    "allocated_gb": allocated_gb,
                    "peak_gb": peak_gb,
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
    """Print one layer-level CUDA-memory report."""
    print("\n" + "=" * 100)
    print(
        f"{'layer':50s} {'module':18s} {'output_shape':22s} "
        f"{'out_MB':>8s} {'alloc_GB':>9s}"
    )
    print("-" * 100)
    for stat in layer_stats:
        print(
            f"{stat['name'][:50]:50s} {stat['module'][:18]:18s} "
            f"{str(stat['output_shape'])[:22]:22s} "
            f"{stat['output_mb']:>8.1f} {stat['allocated_gb']:>9.2f}"
        )
    print("-" * 100)

    print("\n>>> Top output-size layers:")
    for stat in sorted(layer_stats, key=lambda item: -item["output_mb"])[:top_k]:
        print(
            f"  {stat['name'][:60]:60s} out={stat['output_mb']:>8.1f}MB "
            f"shape={stat['output_shape']} alloc={stat['allocated_gb']:.2f}GB"
        )

    print("\n>>> Biggest alloc jumps (per-layer, ordered):")
    previous_allocated = 0.0
    jumps = []
    for stat in layer_stats:
        jumps.append(
            (
                stat["name"],
                stat["allocated_gb"] - previous_allocated,
                stat["allocated_gb"],
            )
        )
        previous_allocated = stat["allocated_gb"]
    for name, jump, allocated in sorted(jumps, key=lambda item: -item[1])[:top_k]:
        print(f"  {name[:60]:60s} Δ={jump:+.2f}GB  (累计 {allocated:.2f}GB)")
    print("=" * 100 + "\n")

def train(model_name, model, train_loader, val_loader,
          normalization_scalars, args, num_epochs=100, eval_freq = 10,
          save_path="models/best_model.pth", predicted_feature_name="pressure"):
    """
    Train the model with validation monitoring.

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        normalization_scalars: Normalization scalars for denormalization
        num_epochs: Number of training epochs
        save_path: Path to save the best model

    Returns:
        dict: Training history
    """
    device = select_device()
    print(f"Using device: {device}")
    amp_enabled, amp_dtype, scaler = configure_amp(device)
    if amp_enabled:
        print(f"AMP enabled with dtype={amp_dtype}")

    model = model.float().to(device)
    criterion = nn.MSELoss()

    # RETO module
    #optimizer = optim.AdamW(
    #    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    #)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer,
    #    step_size=args.scheduler_step,
    #    gamma=args.scheduler_gamma
    #)

    # PGD-NO module
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=max(1, len(train_loader) * num_epochs),
        final_div_factor=1000.,
    )

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.7, patience=20)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model_paths = checkpoint_paths(save_path)
    for model_path in model_paths.values():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_relative_error': [],
        'best_val_error': float('inf'),
        'best_val_loss': float('inf'),
        'checkpoint_paths': {
            name: str(path) for name, path in model_paths.items()
        },
    }

    print(f"Starting training for {num_epochs} epochs (model_name='{model_name}')...")
    print("Saving independent best checkpoints for val_loss and relative_error")

    memory_report_pending = device.type == "cuda"
    memory_hooks = []
    memory_stats = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Train",
            unit="batch",
            leave=False,
        )
        for batch_data in train_progress:
            if memory_report_pending:
                torch.cuda.reset_peak_memory_stats(device)
                memory_hooks, memory_stats = register_mem_hooks(model)

            # Dataloader format: normalized features, raw XYZ, segmentation, target, ID
            coorf, raw_coords, seg_matrix, target, sim_ids = batch_data
            coorf = coorf[...,:3]
            coorf, raw_coords, target = (
                coorf.to(device), raw_coords.to(device), target.to(device)
            )
            order = inverse = None
            if model_name in SERIALIZED_MODELS:
                serialization_coords = serialization_coordinates(
                    model_name, coorf, raw_coords
                )
                order, inverse = pos_to_order_inverse_index(
                    serialization_coords, device, tensor=True
                )
            if model_name in SEGMENTED_MODELS:
                seg_matrix = seg_matrix.to(device)
            if model_name in SEGMENTED_SER:
                seg_matrix = seg_matrix.to(device)

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    outputs = forward_model(
                        model_name, model, coorf, raw_coords, seg_matrix, order, inverse
                    )
                    loss = criterion(outputs, target)

            except torch.OutOfMemoryError:
                print(f"\n❌ OOM! 当前已分配 {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                if memory_report_pending:
                    print_mem_report(memory_stats)
                    remove_mem_hooks(memory_hooks)
                    memory_report_pending = False
                raise

            if memory_report_pending:
                print_mem_report(memory_stats)
                remove_mem_hooks(memory_hooks)
                memory_report_pending = False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4e}")

        train_loss /= len(train_loader)

        # Validation phase
        if epoch % eval_freq == 0:
            model.eval()
            val_loss = 0.0
            val_relative_error = 0.0
            with torch.no_grad():
                val_progress = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{num_epochs} - Val",
                    unit="batch",
                    leave=False,
                )
                for batch_data in val_progress:
                    coorf, raw_coords, seg_matrix, target, sim_ids = batch_data
                    coorf = coorf[...,:3]
                    coorf, raw_coords, target = (
                        coorf.to(device), raw_coords.to(device), target.to(device)
                    )
                    order = inverse = None
                    if model_name in SERIALIZED_MODELS:
                        serialization_coords = serialization_coordinates(
                            model_name, coorf, raw_coords
                        )
                        order, inverse = pos_to_order_inverse_index(
                            serialization_coords, device, tensor=True
                        )
                    if model_name in SEGMENTED_MODELS:
                        seg_matrix = seg_matrix.to(device)
                    if model_name in SEGMENTED_SER:
                        seg_matrix = seg_matrix.to(device)
                    with autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        outputs = forward_model(
                            model_name, model, coorf, raw_coords, seg_matrix, order, inverse
                        )
                        loss = criterion(outputs, target)
                    val_loss += loss.detach().cpu().item()

                    # Compute relative error
                    relative_error = batch_relative_error(
                        outputs, target, sim_ids, normalization_scalars,
                        predicted_feature_name,
                    )
                    val_relative_error += relative_error
                    val_progress.set_postfix(loss=f"{loss.item():.4e}")

            val_loss /= len(val_loader)
            val_relative_error /= len(val_loader)

            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_relative_error'].append(val_relative_error)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val Relative Error: {val_relative_error:.6f}")

            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                torch.save(model.state_dict(), model_paths['val_loss'])
                print(
                    "  ✅ Saved val_loss checkpoint: "
                    f"{model_paths['val_loss']} ({val_loss:.6f})"
                )

            if val_relative_error < history['best_val_error']:
                history['best_val_error'] = val_relative_error
                torch.save(model.state_dict(), model_paths['relative_error'])
                print(
                    "  ✅ Saved relative_error checkpoint: "
                    f"{model_paths['relative_error']} "
                    f"({val_relative_error:.6f})"
                )

    print("Training completed!")
    print(f"  Best validation loss: {history['best_val_loss']:.6f}")
    print(
        "  Best validation relative error: "
        f"{history['best_val_error']:.6f}"
    )
    print(f"  val_loss checkpoint: {model_paths['val_loss']}")
    print(f"  relative_error checkpoint: {model_paths['relative_error']}")
    return history

def test(model_name, model, test_loader, normalization_scalars, model_path="models/best_model.pth", predicted_feature_name="pressure"):
    """
    Test the trained model on test data.

    Args:
        model: The neural network model
        test_loader: Test data loader
        normalization_scalars: Normalization scalars for denormalization
        model_path: Path to the trained model

    Returns:
        dict: Test results
    """
    device = select_device()
    amp_enabled, amp_dtype, _ = configure_amp(device)

    # Load the best model
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")

    model = model.float().to(device)
    model.eval()

    criterion = nn.MSELoss()
    test_loss = 0.0
    test_relative_error = 0.0
    all_predictions = {}
    all_targets = {}
    r2_count = 0
    r2_target_sum = None
    r2_target_sum_sq = None
    r2_error_sum_sq = None

    with torch.no_grad():
        test_progress = tqdm(
            test_loader,
            desc="Testing",
            unit="batch",
            leave=True,
        )
        for batch_data in test_progress:
            coorf, raw_coords, seg_matrix, target, sim_ids = batch_data
            coorf = coorf[...,:3]
            coorf, raw_coords, target = (
                coorf.to(device), raw_coords.to(device), target.to(device)
            )
            order = inverse = None
            if model_name in SERIALIZED_MODELS:
                serialization_coords = serialization_coordinates(
                    model_name, coorf, raw_coords
                )
                order, inverse = pos_to_order_inverse_index(
                    serialization_coords, device, tensor=True
                )

            if model_name in SEGMENTED_MODELS:
                seg_matrix = seg_matrix.to(device)

            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = forward_model(
                    model_name, model, coorf, raw_coords, seg_matrix, order, inverse
                )
                loss = criterion(outputs, target)
            test_loss += loss.item()

            # Compute field-level relative error
            relative_error = batch_relative_error(
                outputs, target, sim_ids, normalization_scalars,
                predicted_feature_name,
            )
            test_relative_error += relative_error
            test_progress.set_postfix(loss=f"{loss.item():.4e}")

            for predicted_physical, target_physical in batch_physical_fields(
                outputs,
                target,
                sim_ids,
                normalization_scalars,
                predicted_feature_name,
            ):
                predicted_flat = predicted_physical.reshape(
                    -1, predicted_physical.shape[-1]
                )
                target_flat = target_physical.reshape(
                    -1, target_physical.shape[-1]
                )
                error_sum_sq = np.square(predicted_flat - target_flat).sum(axis=0)
                target_sum = target_flat.sum(axis=0)
                target_sum_sq = np.square(target_flat).sum(axis=0)
                if r2_target_sum is None:
                    r2_target_sum = np.zeros_like(target_sum, dtype=np.float64)
                    r2_target_sum_sq = np.zeros_like(target_sum_sq, dtype=np.float64)
                    r2_error_sum_sq = np.zeros_like(error_sum_sq, dtype=np.float64)
                r2_count += target_flat.shape[0]
                r2_target_sum += target_sum
                r2_target_sum_sq += target_sum_sq
                r2_error_sum_sq += error_sum_sq

            # Store predictions and targets for detailed analysis
            outputs_cpu = outputs.detach().float().cpu()
            targets_cpu = target.detach().cpu()
            for batch_index, sim_id in enumerate(sim_ids):
                all_predictions[sim_id] = outputs_cpu[batch_index:batch_index + 1]
                all_targets[sim_id] = targets_cpu[batch_index:batch_index + 1]

    test_loss /= len(test_loader)
    test_relative_error /= len(test_loader)
    component_r2 = r2_from_sufficient_statistics(
        r2_count,
        r2_target_sum,
        r2_target_sum_sq,
        r2_error_sum_sq,
    )
    test_r2 = float(np.mean(component_r2))

    results = {
        'test_loss': test_loss,
        'test_relative_error': test_relative_error,
        'test_r2': test_r2,
        'test_component_r2': component_r2,
        'predictions': all_predictions,
        'targets': all_targets
    }

    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Relative L2 Error: {test_relative_error:.6f}")
    print(f"  Test R2: {test_r2:.6f}")

    # Report per-component errors for vector targets such as velocity and WSS.
    if all_predictions:
        component_errors = []
        for sim_id in all_predictions:
            scalars = normalization_scalars
            if isinstance(scalars, dict) and "by_vehicle" in scalars:
                scalars = scalars["by_vehicle"][sim_id[0]]
            component_errors.append(
                compute_component_relative_errors(
                    all_predictions[sim_id].numpy(), all_targets[sim_id].numpy(),
                    scalars, predicted_feature_name,
                )
            )
        component_errors = np.mean(component_errors, axis=0)
        results["test_component_relative_errors"] = component_errors
        names = ["x", "y", "z"] if len(component_errors) == 3 else ["scalar"]
        print("  Component Relative Errors: " + ", ".join(
            f"{name}={value:.6f}" for name, value in zip(names, component_errors)
        ))
        print("  Component R2: " + ", ".join(
            f"{name}={value:.6f}" for name, value in zip(names, component_r2)
        ))

    return results
