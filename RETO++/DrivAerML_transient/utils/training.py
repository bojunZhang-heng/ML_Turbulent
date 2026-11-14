"""Training, validation and autoregressive rollout for transient fields."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dataloader import TASKS, TransientSequence


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1.0e-12)
    return (numerator / denominator).mean()


def amp_context(device: torch.device, amp_type: str):
    if device.type != "cuda" or amp_type == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_type == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        batch["coordinates"],
        batch["current_state"],
        batch["time"],
        batch["delta_t"],
    )


def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_type: str,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    value_count = 0
    relative_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with amp_context(device, amp_type):
                prediction = forward_batch(model, batch)
            difference = prediction.float() - batch["target_state"].float()
            squared_error += difference.square().sum().item()
            value_count += difference.numel()
            batch_size = prediction.shape[0]
            relative_sum += relative_l2(
                prediction.float(), batch["target_state"].float()
            ).item() * batch_size
            sample_count += batch_size
    return {
        "mse": squared_error / max(value_count, 1),
        "relative_l2": relative_sum / max(sample_count, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_relative_l2: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_relative_l2": best_relative_l2,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: Path,
    resume: bool = False,
) -> list[dict[str, float]]:
    training = config["training"]
    epochs = int(training["epochs"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    relative_weight = float(training.get("relative_loss_weight", 0.0))
    gradient_clip = float(training.get("gradient_clip", 0.0))
    amp_type = str(training.get("amp_type", "none")).lower()
    if amp_type not in {"none", "fp16", "bf16"}:
        raise ValueError("training.amp_type must be none, fp16 or bf16")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(training.get("min_lr", 1.0e-6))
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_type == "fp16")
    )
    start_epoch = 1
    best = math.inf
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, model, device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_relative_l2"])

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_type):
                prediction = forward_batch(model, batch)
                mse = nn.functional.mse_loss(prediction, batch["target_state"])
                rel = relative_l2(prediction, batch["target_state"])
                loss = mse + relative_weight * rel
            scaler.scale(loss).backward()
            if gradient_clip > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().float().item()
            batches += 1
        scheduler.step()

        validation = evaluate_loader(model, val_loader, device, amp_type)
        record = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": loss_sum / max(batches, 1),
            "val_mse": validation["mse"],
            "val_relative_l2": validation["relative_l2"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:4d}/{epochs}: train={record['train_loss']:.6e}, "
            f"val_mse={record['val_mse']:.6e}, "
            f"val_relL2={record['val_relative_l2']:.6e}, "
            f"lr={record['learning_rate']:.3e}"
        )
        if validation["relative_l2"] < best:
            best = validation["relative_l2"]
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch, best, config
            )
            print(f"  saved best checkpoint: {checkpoint_path}")
    return history


def denormalization_values(
    sequence: TransientSequence, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    _, field_key, _ = TASKS[sequence.task]
    prefix = field_key.removeprefix("y_")
    mean_key = f"y_{prefix}_mean"
    std_key = f"y_{prefix}_std"
    if mean_key not in sequence.scalars or std_key not in sequence.scalars:
        raise KeyError(f"Missing {mean_key}/{std_key} in normalization_scalars.pkl")
    mean = torch.as_tensor(sequence.scalars[mean_key], dtype=torch.float32, device=device)
    std = torch.as_tensor(sequence.scalars[std_key], dtype=torch.float32, device=device)
    return mean, std


def rollout(
    model: nn.Module,
    sequence: TransientSequence,
    pair_starts: list[int],
    context_steps: int,
    device: torch.device,
    amp_type: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not pair_starts:
        raise ValueError("Rollout requires at least one temporal pair")
    coordinates = sequence.coordinates.unsqueeze(0).to(device)
    predictions, targets, sample_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        segment_start = 0
        while segment_start < len(pair_starts):
            segment_end = segment_start + 1
            while (
                segment_end < len(pair_starts)
                and pair_starts[segment_end] == pair_starts[segment_end - 1] + 1
            ):
                segment_end += 1
            segment = pair_starts[segment_start:segment_end]
            context = [
                sequence.field(index).to(device)
                for index in range(segment[0], segment[0] + context_steps)
            ]
            for start in segment:
                target_index = start + context_steps
                current_time = sequence.frames[target_index - 1].time_value
                target_time = sequence.frames[target_index].time_value
                current = torch.cat(context[-context_steps:], dim=-1).unsqueeze(0)
                time = torch.tensor(
                    [sequence.normalized_time(current_time)], device=device
                )
                delta_t = torch.tensor(
                    [sequence.normalized_delta(target_time - current_time)], device=device
                )
                with amp_context(device, amp_type):
                    prediction = model(coordinates, current, time, delta_t).float()
                prediction_cpu = prediction.squeeze(0).cpu()
                predictions.append(prediction_cpu)
                targets.append(sequence.field(target_index))
                sample_ids.append(sequence.frames[target_index].sample_id)
                context.append(prediction.squeeze(0))
            segment_start = segment_end
    return torch.stack(predictions), torch.stack(targets), sample_ids


def test_rollout(
    model: nn.Module,
    sequence: TransientSequence,
    test_pair_starts: list[int],
    context_steps: int,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, float]:
    load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    amp_type = str(config["training"].get("amp_type", "none")).lower()
    prediction, target, sample_ids = rollout(
        model, sequence, test_pair_starts, context_steps, device, amp_type
    )
    difference = prediction - target
    normalized_mse = difference.square().mean().item()
    normalized_rel = (
        torch.linalg.vector_norm(difference.flatten(1), dim=1)
        / torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1.0e-12)
    ).mean().item()
    mean, std = denormalization_values(sequence, torch.device("cpu"))
    prediction_physical = prediction * std + mean
    target_physical = target * std + mean
    physical_difference = prediction_physical - target_physical
    metrics = {
        "normalized_mse": normalized_mse,
        "normalized_relative_l2": normalized_rel,
        "physical_mae": physical_difference.abs().mean().item(),
        "physical_rmse": physical_difference.square().mean().sqrt().item(),
        "physical_relative_l2": (
            torch.linalg.vector_norm(physical_difference.flatten(1), dim=1)
            / torch.linalg.vector_norm(target_physical.flatten(1), dim=1).clamp_min(1.0e-12)
        ).mean().item(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    if bool(config.get("evaluation", {}).get("save_predictions", True)):
        np.savez_compressed(
            output_dir / "rollout_predictions.npz",
            prediction_normalized=prediction.numpy(),
            target_normalized=target.numpy(),
            prediction_physical=prediction_physical.numpy(),
            target_physical=target_physical.numpy(),
            sample_ids=np.asarray(sample_ids),
        )
    print("Autoregressive test rollout:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.8e}")
    return metrics


def inspect_model(
    model: nn.Module,
    sequence: TransientSequence,
    context_steps: int,
    device: torch.device,
) -> None:
    if len(sequence.frames) <= context_steps:
        raise ValueError("Not enough frames for one forward pass")
    context = torch.cat(
        [sequence.field(index) for index in range(context_steps)], dim=-1
    ).unsqueeze(0).to(device)
    coordinates = sequence.coordinates.unsqueeze(0).to(device)
    current_time = sequence.frames[context_steps - 1].time_value
    target_time = sequence.frames[context_steps].time_value
    time = torch.tensor([sequence.normalized_time(current_time)], device=device)
    delta_t = torch.tensor(
        [sequence.normalized_delta(target_time - current_time)], device=device
    )
    model.to(device).eval()
    with torch.inference_mode():
        output = model(coordinates, context, time, delta_t)
    print(f"Inspect forward input : {tuple(context.shape)}")
    print(f"Inspect coordinates   : {tuple(coordinates.shape)}")
    print(f"Inspect forward output: {tuple(output.shape)}")
    print(f"Output finite         : {bool(torch.isfinite(output).all())}")
