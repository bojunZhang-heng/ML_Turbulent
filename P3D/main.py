"""CPU debugging script for one-step prediction with P3D-S.

The synthetic dataset has shape ``(groups, batch, points, time, channels)``.
P3D itself expects a dense 3D field with shape
``(batch, channels, depth, height, width)``. This script stores fixed-order
points in a dense cube and uses a mask for padded points.

1. 使用第10个时间步，预测第11个
/opt/anaconda3/envs/MLTG/bin/python main.py --time-index 10

2. 使用全部时间步，预测下一步 
/opt/anaconda3/envs/MLTG/bin/python main.py --all-time-steps

"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from p3d_surrogate import P3D_S


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug P3D-S next-time-step prediction with random point data."
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument(
        "--num-points",
        type=int,
        default=0,
        help="Number of points. 0 means size**3; points are padded to the cube.",
    )
    parser.add_argument("--time-steps", type=int, default=50)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument(
        "--all-time-steps",
        action="store_true",
        help="Train on all t -> t+1 pairs instead of only --time-index.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pair-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mode", choices=("inference", "train"), default="train")
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--shift", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "channels": args.channels,
        "size": args.size,
        "time-steps": args.time_steps,
        "num-groups": args.num_groups,
        "batch-size": args.batch_size,
        "epochs": args.epochs,
        "pair-batch-size": args.pair_batch_size,
        "threads": args.threads,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be greater than zero, got {value}")

    if args.size < 16 or args.size % 4 != 0:
        raise ValueError(
            "--size must be at least 16 and divisible by 4 for P3D-S, "
            f"got {args.size}"
        )

    if args.num_points < 0 or args.num_points > args.size**3:
        raise ValueError(
            f"--num-points must be between 1 and size**3 ({args.size**3}), "
            f"got {args.num_points}"
        )

    if args.time_steps < 2:
        raise ValueError("--time-steps must be at least 2 for t -> t+1 prediction")

    if not 0 <= args.time_index < args.time_steps - 1:
        raise ValueError(
            f"--time-index must be in [0, {args.time_steps - 2}], "
            f"got {args.time_index}"
        )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def generate_sequences(
    num_groups: int,
    batch_size: int,
    num_points: int,
    time_steps: int,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate temporally correlated data with shape (G, B, N, Nt, C)."""
    initial_state = torch.randn(
        num_groups,
        batch_size,
        num_points,
        channels,
        device=device,
    )
    sequence = [initial_state]
    for _ in range(time_steps - 1):
        next_state = 0.98 * sequence[-1] + 0.02 * torch.randn_like(sequence[-1])
        sequence.append(next_state)
    return torch.stack(sequence, dim=3)


def point_cloud_to_grid(
    point_field: torch.Tensor,
    size: int,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map (B, N, C) fixed-order points to (B, C, size, size, size)."""
    batch_size, _, channels = point_field.shape
    grid_point_count = size**3
    grid = point_field.new_zeros(batch_size, channels, grid_point_count)
    valid_mask = point_field.new_zeros(batch_size, 1, grid_point_count)

    grid[:, :, :num_points] = point_field.transpose(1, 2)
    valid_mask[:, :, :num_points] = 1.0
    grid = grid.reshape(batch_size, channels, size, size, size)
    valid_mask = valid_mask.reshape(batch_size, 1, size, size, size)
    return grid, valid_mask


def grid_to_point_cloud(
    grid_field: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Map (B, C, size, size, size) back to (B, N, C)."""
    batch_size, channels, _, _, _ = grid_field.shape
    point_field = grid_field.reshape(batch_size, channels, -1)
    return point_field[:, :, :num_points].transpose(1, 2).contiguous()


def build_temporal_pairs(
    sequences: torch.Tensor,
    size: int,
    num_points: int,
    time_index: int,
    all_time_steps: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build current and next fields for the requested time-step pairs."""
    num_groups, batch_size, _, time_steps, _ = sequences.shape
    time_indices = range(time_steps - 1) if all_time_steps else (time_index,)

    current_fields = []
    next_fields = []
    masks = []
    normalized_times = []
    for current_time in time_indices:
        current = sequences[:, :, :, current_time, :].reshape(
            num_groups * batch_size, -1, sequences.shape[-1]
        )
        next_state = sequences[:, :, :, current_time + 1, :].reshape(
            num_groups * batch_size, -1, sequences.shape[-1]
        )
        current_grid, valid_mask = point_cloud_to_grid(
            current, size, num_points
        )
        next_grid, _ = point_cloud_to_grid(next_state, size, num_points)
        current_fields.append(current_grid)
        next_fields.append(next_grid)
        masks.append(valid_mask)
        normalized_times.extend(
            [current_time / (time_steps - 1)] * (num_groups * batch_size)
        )

    return (
        torch.cat(current_fields, dim=0),
        torch.cat(next_fields, dim=0),
        torch.cat(masks, dim=0),
        torch.tensor(
            normalized_times,
            dtype=sequences.dtype,
            device=sequences.device,
        ),
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction - target).square() * valid_mask
    return squared_error.sum() / (valid_mask.sum() * prediction.shape[1])


def predict_one_step(
    model: torch.nn.Module,
    current_field: torch.Tensor,
    current_time: torch.Tensor,
    class_labels: torch.Tensor,
    pde_parameters: torch.Tensor,
) -> torch.Tensor:
    return model(
        current_field,
        timestep=current_time,
        class_labels=class_labels,
        pde_parameters=pde_parameters,
    ).sample


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    num_points = args.num_points or args.size**3
    periodic = [True, True, True] if args.periodic else False
    model = P3D_S(
        channel_size=args.channels,
        channel_size_out=args.channels,
        drop_class_labels=True,
        periodic=periodic,
        shift=args.shift,
    ).to(device)

    sequences = generate_sequences(
        num_groups=args.num_groups,
        batch_size=args.batch_size,
        num_points=num_points,
        time_steps=args.time_steps,
        channels=args.channels,
        device=device,
    )
    current_fields, next_fields, valid_masks, normalized_times = build_temporal_pairs(
        sequences,
        size=args.size,
        num_points=num_points,
        time_index=args.time_index,
        all_time_steps=args.all_time_steps,
    )

    pair_count = current_fields.shape[0]
    print(f"device: {device}")
    print("model: P3D_S")
    print(f"parameters: {parameter_count(model):,}")
    print(f"synthetic dataset shape: {tuple(sequences.shape)}")
    print(f"point-cloud pair count: {pair_count}")
    print(f"current field shape: {tuple(current_fields.shape)}")
    print(f"next field shape: {tuple(next_fields.shape)}")
    print(f"valid points per sample: {num_points}/{args.size**3}")

    if args.mode == "inference":
        model.eval()
        sample_count = min(args.pair_batch_size, pair_count)
        with torch.inference_mode():
            prediction = predict_one_step(
                model,
                current_fields[:sample_count],
                normalized_times[:sample_count],
                torch.zeros(sample_count, dtype=torch.long, device=device),
                torch.zeros(sample_count, device=device),
            )
        point_prediction = grid_to_point_cloud(prediction, num_points)
        print(f"prediction shape: {tuple(prediction.shape)}")
        print(f"point prediction shape: {tuple(point_prediction.shape)}")
        print(
            "masked one-step MSE: "
            f"{masked_mse(prediction, next_fields[:sample_count], valid_masks[:sample_count]).item():.6e}"
        )
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        permutation = torch.randperm(pair_count, device=device)
        epoch_loss = 0.0
        gradient_norm = torch.tensor(0.0, device=device)
        for batch_start in range(0, pair_count, args.pair_batch_size):
            indices = permutation[batch_start : batch_start + args.pair_batch_size]
            current_batch = current_fields[indices]
            target_batch = next_fields[indices]
            mask_batch = valid_masks[indices]
            time_batch = normalized_times[indices]
            labels_batch = torch.zeros(
                indices.shape[0], dtype=torch.long, device=device
            )
            parameters_batch = torch.zeros(indices.shape[0], device=device)

            optimizer.zero_grad(set_to_none=True)
            prediction = predict_one_step(
                model,
                current_batch,
                time_batch,
                labels_batch,
                parameters_batch,
            )
            loss = masked_mse(prediction, target_batch, mask_batch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")
            )
            optimizer.step()
            epoch_loss += loss.item() * indices.shape[0]

        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"masked one-step MSE = {epoch_loss / pair_count:.6e}"
        )
        print(f"last gradient norm: {gradient_norm.item():.6e}")

    model.eval()
    with torch.inference_mode():
        first_prediction = predict_one_step(
            model,
            current_fields[:1],
            normalized_times[:1],
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, device=device),
        )
    point_prediction = grid_to_point_cloud(first_prediction, num_points)
    print(f"prediction shape: {tuple(first_prediction.shape)}")
    print(f"point prediction shape: {tuple(point_prediction.shape)}")
    print("next-step training: ok")
    print("backward pass: ok")
    print("optimizer step: ok")


if __name__ == "__main__":
    main()
