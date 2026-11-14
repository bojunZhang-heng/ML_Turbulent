"""Temporal data loading for split-geometry DrivAerML PKL sequences."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TASKS = {
    "surface_pressure": ("x_surface", "y_spressure", 1),
    "volume_velocity": ("x_volume", "y_velocity", 3),
}


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load PKLs written by NumPy versions using ``numpy._core``."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core":
            module = "numpy.core"
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return NumpyCompatibleUnpickler(stream).load()


@dataclass(frozen=True)
class FrameInfo:
    sample_id: int
    path: Path
    time_value: float
    time_name: str


class TransientSequence:
    """One fixed mesh and its chronologically ordered transient fields."""

    def __init__(
        self,
        root: str | Path,
        task: str,
        start_sample_id: int = 10,
        end_sample_id: int | None = None,
        cache_frames: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Transient data directory not found: {self.root}")
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; choose one of {sorted(TASKS)}")
        self.task = task
        self.coord_key, self.field_key, self.channels = TASKS[task]
        self.cache_frames = bool(cache_frames)
        self._cache: dict[int, torch.Tensor] = {}

        geometry_path = self.root / "geometry.pkl"
        scalars_path = self.root / "normalization_scalars.pkl"
        if not geometry_path.is_file() or not scalars_path.is_file():
            raise FileNotFoundError(
                f"{self.root} must contain geometry.pkl and normalization_scalars.pkl"
            )
        geometry = load_pickle(geometry_path)
        self.scalars = load_pickle(scalars_path)
        if self.coord_key not in geometry:
            raise KeyError(f"{geometry_path} does not contain {self.coord_key}")
        coordinates = np.asarray(geometry[self.coord_key], dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(f"{self.coord_key} must have shape (N,3), got {coordinates.shape}")
        if not np.isfinite(coordinates).all():
            raise ValueError(f"{self.coord_key} contains NaN/Inf")
        self.coordinates = torch.from_numpy(coordinates.copy())

        candidates = sorted(
            (
                path
                for path in self.root.glob("*.pkl")
                if path.stem.isdigit()
                and int(path.stem) >= start_sample_id
                and (end_sample_id is None or int(path.stem) <= end_sample_id)
            ),
            key=lambda path: int(path.stem),
        )
        if not candidates:
            raise FileNotFoundError(
                f"No numbered PKLs in [{start_sample_id}, {end_sample_id}] under {self.root}"
            )

        frames: list[FrameInfo] = []
        for path in candidates:
            sample = load_pickle(path)
            if self.field_key not in sample:
                raise KeyError(f"{path} does not contain {self.field_key}")
            field = self._validate_field(sample[self.field_key], path)
            time_value = float(sample.get("time_value", sample.get("time_name")))
            frames.append(
                FrameInfo(
                    sample_id=int(path.stem),
                    path=path,
                    time_value=time_value,
                    time_name=str(sample.get("time_name", time_value)),
                )
            )
            if self.cache_frames:
                self._cache[len(frames) - 1] = field

        frames.sort(key=lambda item: item.time_value)
        if any(b.time_value <= a.time_value for a, b in zip(frames, frames[1:])):
            raise ValueError("Selected time values must be strictly increasing")
        self.frames = frames
        self.times = torch.tensor([item.time_value for item in frames], dtype=torch.float32)
        self.time_origin = float(self.times[0])
        span = float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0
        self.time_scale = span if span > 0.0 else 1.0

        # Sorting can change cache indices if filenames are not time ordered.
        if self.cache_frames:
            self._cache.clear()
            for index, frame in enumerate(self.frames):
                self._cache[index] = self._load_field(frame.path)

    def _validate_field(self, values: Any, path: Path) -> torch.Tensor:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        expected = (len(self.coordinates), self.channels)
        if tuple(array.shape) != expected:
            raise ValueError(f"{path}: {self.field_key} shape {array.shape} != {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: {self.field_key} contains NaN/Inf")
        return torch.from_numpy(array.copy())

    def _load_field(self, path: Path) -> torch.Tensor:
        sample = load_pickle(path)
        return self._validate_field(sample[self.field_key], path)

    def field(self, frame_index: int) -> torch.Tensor:
        if frame_index not in self._cache:
            self._cache[frame_index] = self._load_field(self.frames[frame_index].path)
            if not self.cache_frames:
                result = self._cache.pop(frame_index)
                return result
        return self._cache[frame_index]

    def normalized_time(self, value: float) -> float:
        return (float(value) - self.time_origin) / self.time_scale

    def normalized_delta(self, value: float) -> float:
        return float(value) / self.time_scale

    def describe(self) -> dict[str, Any]:
        deltas = np.diff([frame.time_value for frame in self.frames])
        return {
            "root": str(self.root),
            "task": self.task,
            "coordinates": tuple(self.coordinates.shape),
            "frames": len(self.frames),
            "first_sample_id": self.frames[0].sample_id,
            "last_sample_id": self.frames[-1].sample_id,
            "first_time": self.frames[0].time_value,
            "last_time": self.frames[-1].time_value,
            "delta_t_min": float(deltas.min()) if len(deltas) else None,
            "delta_t_max": float(deltas.max()) if len(deltas) else None,
        }

    def valid_pair_starts(
        self,
        context_steps: int,
        expected_delta_t: float,
        tolerance: float,
    ) -> tuple[list[int], list[dict[str, float | int]]]:
        """Return windows whose every adjacent frame has the configured spacing."""
        valid: list[int] = []
        gaps: list[dict[str, float | int]] = []
        for start in range(len(self.frames) - context_steps):
            window = self.frames[start : start + context_steps + 1]
            deltas = [
                right.time_value - left.time_value
                for left, right in zip(window, window[1:])
            ]
            if all(abs(value - expected_delta_t) <= tolerance for value in deltas):
                valid.append(start)
            else:
                gaps.append(
                    {
                        "start_sample_id": window[0].sample_id,
                        "target_sample_id": window[-1].sample_id,
                        "observed_delta_t": float(max(deltas)),
                    }
                )
        return valid, gaps


class TemporalPairDataset(Dataset):
    """Create context-window -> next-frame samples from one trajectory."""

    def __init__(
        self,
        sequence: TransientSequence,
        pair_starts: list[int],
        context_steps: int,
    ) -> None:
        self.sequence = sequence
        self.pair_starts = list(pair_starts)
        self.context_steps = int(context_steps)

    def __len__(self) -> int:
        return len(self.pair_starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = self.pair_starts[index]
        target_index = start + self.context_steps
        context = torch.cat(
            [self.sequence.field(i) for i in range(start, target_index)], dim=-1
        )
        current_time = self.sequence.frames[target_index - 1].time_value
        target_time = self.sequence.frames[target_index].time_value
        return {
            "coordinates": self.sequence.coordinates,
            "current_state": context,
            "target_state": self.sequence.field(target_index),
            "time": torch.tensor(
                self.sequence.normalized_time(current_time), dtype=torch.float32
            ),
            "delta_t": torch.tensor(
                self.sequence.normalized_delta(target_time - current_time),
                dtype=torch.float32,
            ),
            "start_index": start,
            "target_index": target_index,
            "target_sample_id": self.sequence.frames[target_index].sample_id,
        }


def chronological_split(
    pair_starts: list[int], train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[list[int], list[int], list[int]]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train/val/test ratios must be positive and sum to 1")
    pair_count = len(pair_starts)
    if pair_count < 3:
        raise ValueError(
            f"At least 3 temporal pairs are required for independent train/val/test "
            f"sets, but only {pair_count} are available"
        )
    remaining = pair_count - 3
    raw = remaining * ratios
    extra = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - extra))[: remaining - int(extra.sum())]:
        extra[index] += 1
    counts = extra + 1
    train_end = int(counts[0])
    val_end = train_end + int(counts[1])
    return (
        pair_starts[:train_end],
        pair_starts[train_end:val_end],
        pair_starts[val_end:],
    )


def create_data_loaders(
    sequence: TransientSequence,
    context_steps: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool,
    num_workers: int,
    expected_delta_t: float,
    delta_t_tolerance: float,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    if context_steps < 1:
        raise ValueError("context_steps must be positive")
    valid_starts, gaps = sequence.valid_pair_starts(
        context_steps, expected_delta_t, delta_t_tolerance
    )
    train_ids, val_ids, test_ids = chronological_split(
        valid_starts, train_ratio, val_ratio, test_ratio
    )
    common = {"batch_size": batch_size, "num_workers": num_workers}
    train_loader = DataLoader(
        TemporalPairDataset(sequence, train_ids, context_steps),
        shuffle=shuffle,
        **common,
    )
    val_loader = DataLoader(
        TemporalPairDataset(sequence, val_ids, context_steps),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        TemporalPairDataset(sequence, test_ids, context_steps),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader, {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "skipped_noncontiguous": gaps,
    }
