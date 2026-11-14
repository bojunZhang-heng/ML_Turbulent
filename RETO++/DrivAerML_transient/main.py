#!/usr/bin/env python3
"""Train a next-frame surrogate on transient DrivAerML point fields."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from dataloader import TASKS, TransientSequence, create_data_loaders
from models import TemporalPointOperator
from utils import inspect_model, test_rollout, train_model


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=None, help="YAML configuration")
    parser.add_argument("--config", dest="config_option", default=None)
    parser.add_argument("--phase", choices=("inspect", "train", "resume", "test"))
    parser.add_argument("--device", default=None, help="auto, cpu, cuda or mps")
    args = parser.parse_args()
    selected = args.config_option or args.config or "config_surface_pressure.yml"
    args.config_path = Path(selected)
    if not args.config_path.is_absolute():
        args.config_path = (ROOT / args.config_path).resolve()
    return args


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    for section in ("experiment", "data", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    cli = parse_args()
    config = load_config(cli.config_path)
    phase = cli.phase or str(config["experiment"].get("phase", "train"))
    device = resolve_device(cli.device or str(config["experiment"].get("device", "auto")))
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)

    print("=" * 80)
    print("DrivAerML transient next-frame training")
    print(f"Configuration: {cli.config_path}")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True).rstrip())
    print(f"Resolved device: {device}")
    print("=" * 80)

    data = config["data"]
    task = str(data["task"])
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose one of {sorted(TASKS)}")
    data_path = Path(data["path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()
    sequence = TransientSequence(
        data_path,
        task=task,
        start_sample_id=int(data.get("start_sample_id", 10)),
        end_sample_id=(
            int(data["end_sample_id"])
            if data.get("end_sample_id") is not None
            else None
        ),
        cache_frames=bool(data.get("cache_frames", True)),
    )
    print("Sequence:")
    print(json.dumps(sequence.describe(), indent=2))

    context_steps = int(data.get("context_steps", 1))
    model_config = config["model"]
    model = TemporalPointOperator(
        state_channels=sequence.channels,
        context_steps=context_steps,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        layers=int(model_config.get("layers", 6)),
        heads=int(model_config.get("heads", 8)),
        mlp_ratio=float(model_config.get("mlp_ratio", 2.0)),
        dropout=float(model_config.get("dropout", 0.0)),
        position_frequencies=int(model_config.get("position_frequencies", 6)),
        time_frequencies=int(model_config.get("time_frequencies", 8)),
        checkpoint_blocks=bool(model_config.get("checkpoint_blocks", True)),
        residual_prediction=bool(model_config.get("residual_prediction", True)),
    )
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    if phase == "inspect":
        inspect_model(model, sequence, context_steps, device)
        return 0

    train_loader, val_loader, _, splits = create_data_loaders(
        sequence=sequence,
        context_steps=context_steps,
        batch_size=int(data.get("batch_size", 1)),
        train_ratio=float(data.get("train_ratio", 0.7)),
        val_ratio=float(data.get("val_ratio", 0.15)),
        test_ratio=float(data.get("test_ratio", 0.15)),
        shuffle=bool(data.get("shuffle", True)),
        num_workers=int(data.get("num_workers", 0)),
        expected_delta_t=float(data.get("expected_delta_t", 0.005)),
        delta_t_tolerance=float(data.get("delta_t_tolerance", 1.0e-8)),
    )
    print(f"Chronological pair split: {splits}")

    experiment = config["experiment"]
    output_dir = Path(experiment.get("output_dir", "experiments")).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    name = str(experiment.get("name", f"temporal_{task}"))
    run_dir = output_dir / name
    checkpoint_path = run_dir / "best_model.pth"

    if phase in {"train", "resume"}:
        history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            config,
            checkpoint_path,
            resume=(phase == "resume"),
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
    elif phase != "test":
        raise ValueError(f"Unsupported phase: {phase}")

    test_rollout(
        model=model,
        sequence=sequence,
        test_pair_starts=splits["test"],
        context_steps=context_steps,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=run_dir / "test",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
