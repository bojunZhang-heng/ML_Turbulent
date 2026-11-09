"""Reconstruct exactly the test split encoded in a training YAML."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

CONFIG_SECTIONS = ("experiment", "model", "attention", "position_encoding",
                   "serialization", "checkpoint", "training", "data")


TASK_FIELD_ALIASES = {
    "surface_pressure": (("x_surface", "coor", "features_6d"), ("y_spressure", "pressure")),
    "pressure": (("x_surface", "coor", "features_6d"), ("y_spressure", "pressure")),
    "surface_wss": (("x_surface", "coor", "features_6d"), ("y_wss", "wss")),
    "wss": (("x_surface", "coor", "features_6d"), ("y_wss", "wss")),
    "volume_velocity": (("x_volume",), ("y_velocity", "velocity")),
    "velocity": (("x_volume",), ("y_velocity", "velocity")),
    "p_and_v": (("input", "x"), ("output", "y")),
    "combined": (("input", "x"), ("output", "y")),
}


def load_config(path: Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tab_lines = [
        index for index, line in enumerate(text.splitlines(), start=1)
        if "\t" in line
    ]
    if tab_lines:
        raise ValueError(
            f"YAML indentation cannot contain Tab characters: {path}; "
            f"lines={tab_lines}. Replace each Tab with two spaces."
        )
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        lines = text.splitlines()
        if mark is None:
            context = "<line information unavailable>"
        else:
            error_line = mark.line + 1
            start = max(1, error_line - 3)
            end = min(len(lines), error_line + 3)
            context = "\n".join(
                f"{line_number:>5}: {lines[line_number - 1]!r}"
                for line_number in range(start, end + 1)
            )
        raise ValueError(
            f"Invalid YAML syntax in {path}. YAML hierarchy must use spaces "
            f"consistently (this project uses two spaces per level).\n"
            f"Context around the parser error:\n{context}\n"
            f"Original parser message: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping in {path}")
    flat = {key: value for key, value in raw.items() if not isinstance(value, dict)}
    for section in CONFIG_SECTIONS:
        values = raw.get(section, {}) or {}
        if not isinstance(values, dict):
            raise TypeError(f"YAML section {section!r} must be a mapping")
        flat.update(values)
    flat["_config_dir"] = str(path.parent.resolve())
    return raw, flat


def _numeric_sample_ids(data_path: Path, task: str) -> tuple[list[int], list[int]]:
    """Discover numbered PKLs without tying postprocess to one dataset loader."""
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_path}")
    paths = {
        int(path.stem): path
        for path in data_path.glob("*.pkl")
        if path.stem.isdigit()
    }
    if not paths:
        raise FileNotFoundError(f"No numbered PKL samples found in {data_path}")

    aliases = TASK_FIELD_ALIASES.get(str(task).strip().lower())
    if aliases is None:
        raise ValueError(f"Unsupported task for split discovery: {task!r}")
    input_keys, target_keys = aliases
    valid, skipped = [], []
    scalar_path = data_path / "normalization_scalars.pkl"
    manifest_ids = None
    if scalar_path.is_file():
        try:
            import pickle

            with scalar_path.open("rb") as handle:
                scalars = pickle.load(handle)
            manifest = scalars.get("samples") if isinstance(scalars, dict) else None
            if manifest:
                manifest_ids = {
                    int(item["sample_id"])
                    for item in manifest
                    if isinstance(item, dict) and str(item.get("sample_id", "")).isdigit()
                }
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, TypeError, ValueError):
            manifest_ids = None
    for sample_id, path in sorted(paths.items()):
        if manifest_ids is not None and sample_id not in manifest_ids:
            skipped.append(sample_id)
            continue
        try:
            import pickle

            with path.open("rb") as handle:
                sample = pickle.load(handle)
            has_input = any(key in sample for key in input_keys)
            has_target = any(key in sample for key in target_keys)
            (valid if has_input and has_target else skipped).append(sample_id)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, TypeError):
            skipped.append(sample_id)
    if not valid:
        raise FileNotFoundError(
            f"No valid PKL samples for task {task!r} in {data_path}"
        )
    return valid, skipped


def resolve_test_data(raw_cfg: dict, cfg: dict, override: Path | None = None):
    data_cfg = raw_cfg.get("data", {}) or {}
    vehicle_cfg = data_cfg.get("vehicle_datasets")
    if vehicle_cfg:
        from dataloader import discover_multi_vehicle_samples

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
    data_path = Path(path_value).expanduser()
    if not data_path.is_absolute():
        data_path = Path(cfg.get("_config_dir", Path.cwd())) / data_path
    data_path = data_path.resolve()
    all_ids, _ = _numeric_sample_ids(
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
