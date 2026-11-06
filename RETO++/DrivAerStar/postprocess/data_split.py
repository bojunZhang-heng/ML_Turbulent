"""Reconstruct exactly the test split encoded in a training YAML."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

from dataloader import discover_drivaerstar_samples, discover_multi_vehicle_samples

CONFIG_SECTIONS = ("experiment", "model", "attention", "position_encoding",
                   "serialization", "checkpoint", "training", "data")


def load_config(path: Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    flat = {key: value for key, value in raw.items() if not isinstance(value, dict)}
    for section in CONFIG_SECTIONS:
        values = raw.get(section, {}) or {}
        if not isinstance(values, dict):
            raise TypeError(f"YAML section {section!r} must be a mapping")
        flat.update(values)
    return raw, flat


def resolve_test_data(raw_cfg: dict, cfg: dict, override: Path | None = None):
    data_cfg = raw_cfg.get("data", {}) or {}
    vehicle_cfg = data_cfg.get("vehicle_datasets")
    if vehicle_cfg:
        roots, _, _, test_ids, _ = discover_multi_vehicle_samples(
            vehicle_configs=vehicle_cfg,
            data_root=str(override) if override else cfg.get("dataset_root") or None,
            selection_mode=str(cfg.get("selection_mode", "random")),
            selection_seed=int(cfg.get("selection_seed", 42)),
            train_ratio=float(cfg.get("train_ratio", 0.6)),
            val_ratio=float(cfg.get("val_ratio", 0.3)),
            test_ratio=float(cfg.get("test_ratio", 0.1)),
        )
        return roots, test_ids

    path_value = override if override is not None else cfg.get("data_path")
    if not path_value:
        raise ValueError("Set data.data_path in YAML or pass --data-path")
    data_path = Path(path_value)
    all_ids, _ = discover_drivaerstar_samples(
        data_path, str(cfg.get("predicted_feature_name", "surface_pressure"))
    )
    selected = list(all_ids)
    mode = str(cfg.get("selection_mode", "random"))
    if mode == "random":
        random.Random(int(cfg.get("selection_seed", 42))).shuffle(selected)
    elif mode != "first":
        raise ValueError("selection_mode must be 'first' or 'random'")
    ratios = np.asarray([cfg.get("train_ratio", 0.6), cfg.get("val_ratio", 0.3), cfg.get("test_ratio", 0.1)], float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train/val/test ratios must be non-negative and sum to one")
    train_count = int(len(selected) * ratios[0])
    val_count = int(len(selected) * ratios[1])
    return str(data_path), selected[train_count + val_count:]


def display_case_id(sim_id):
    return "/".join(map(str, sim_id)) if isinstance(sim_id, tuple) else str(sim_id)
