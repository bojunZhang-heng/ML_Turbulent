import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
import numpy as np
from tqdm import tqdm
from utils.metric import compute_relative_error
from utils.metric import denormalize_pressure


SEGMENTED_MODELS = {
    "transolver_seg",
    "transolver_seg_v2",
    "SegLinearNO",
    "SegLinearNO_v2",
}


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

def train(model_name, model, train_loader, val_loader, normalization_scalars,
          num_epochs=100, learning_rate=0.0001, eval_freq = 10,
          save_path="models/best_model.pth", predicted_feature_name="pressure"):
    """
    Train the model with validation monitoring.

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        normalization_scalars: Normalization scalars for denormalization
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Path to save the best model

    Returns:
        dict: Training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    amp_enabled, amp_dtype, scaler = configure_amp(device)
    if amp_enabled:
        print(f"AMP enabled with dtype={amp_dtype}")

    model = model.float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=(len(train_loader) // batch_size + 1) * num_epochs,
        final_div_factor=1000.,
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.7, patience=20)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_relative_error': [],
        'best_val_error': float('inf')
    }

    print(f"Starting training for {num_epochs} epochs (model_name='{model_name}', assuming 'transolver' and coorf-only input)...")

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

            # New dataloader format: (coorf, seg_matrix, p, sim_id)
            coorf, seg_matrix, target, sim_ids = batch_data
            coorf, target = coorf.to(device), target.to(device)
            if model_name in SEGMENTED_MODELS:
                seg_matrix = seg_matrix.to(device)

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    if model_name == 'transolver':
                        outputs = model(coorf)
                    elif model_name in SEGMENTED_MODELS:
                        outputs = model((coorf, seg_matrix))
                    elif model_name in ('LinearNO', 'RopeLinearNO'):
                        outputs = model(coorf)
                    else:
                        raise ValueError(f"Model name {model_name} not supported")
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
                    # New dataloader format: (coorf, seg_matrix, p, sim_id)
                    coorf, seg_matrix, target, sim_ids = batch_data
                    coorf, target = coorf.to(device), target.to(device)
                    if model_name in SEGMENTED_MODELS:
                        seg_matrix = seg_matrix.to(device)
                    with autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        if model_name == 'transolver':
                            outputs = model(coorf)
                        elif model_name in SEGMENTED_MODELS:
                            outputs = model((coorf, seg_matrix))
                        elif model_name in ('LinearNO', 'RopeLinearNO'):
                            outputs = model(coorf)
                        else:
                            raise ValueError(f"Model name {model_name} not supported")
                        loss = criterion(outputs, target)
                    val_loss += loss.detach().cpu().item()

                    # Compute relative error
                    relative_error = compute_relative_error(
                        outputs.detach().float().cpu().numpy(),
                        target.detach().cpu().numpy(), normalization_scalars)
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

            # Save best model based on validation relative error
            if val_relative_error < history['best_val_error']:
                history['best_val_error'] = val_relative_error
                torch.save(model.state_dict(), save_path)
                print(f"  ✅ New best model saved! (Relative Error: {val_relative_error:.6f})")

    print(f"Training completed! Best validation relative error: {history['best_val_error']:.6f}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype, _ = configure_amp(device)

    # Load the best model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
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

    with torch.no_grad():
        test_progress = tqdm(
            test_loader,
            desc="Testing",
            unit="batch",
            leave=True,
        )
        for batch_data in test_progress:
            # New dataloader format: (coorf, seg_matrix, p, sim_id)
            coorf, seg_matrix, target, sim_ids = batch_data
            coorf, target = coorf.to(device), target.to(device)
            if model_name in SEGMENTED_MODELS:
                seg_matrix = seg_matrix.to(device)

            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                if model_name == 'transolver':
                    outputs = model(coorf)
                elif model_name in SEGMENTED_MODELS:
                    outputs = model((coorf, seg_matrix))
                elif model_name in ('LinearNO', 'RopeLinearNO'):
                    outputs = model(coorf)
                else:
                    raise ValueError(f"Model name {model_name} not supported")
                loss = criterion(outputs, target)
            test_loss += loss.item()

            # Compute field-level relative error
            relative_error = compute_relative_error(
                outputs.detach().float().cpu().numpy(),
                target.detach().cpu().numpy(), normalization_scalars)
            test_relative_error += relative_error
            test_progress.set_postfix(loss=f"{loss.item():.4e}")

            # Store predictions and targets for detailed analysis
            SIM_ID = sim_ids[0]
            all_predictions[SIM_ID] = outputs.detach().float().cpu()
            all_targets[SIM_ID] = target.detach().cpu()

    test_loss /= len(test_loader)
    test_relative_error /= len(test_loader)

    results = {
        'test_loss': test_loss,
        'test_relative_error': test_relative_error,
        'predictions': all_predictions,
        'targets': all_targets
    }

    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Relative Error: {test_relative_error:.6f}")

    return results
