import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
from collections.abc import Sequence
from scipy.sparse import csr_matrix


TASK_ALIASES = {
    "p_and_v": "combined",
    "combined": "combined",
    "output": "combined",
    "seven_to_four": "combined",
    "pressure": "surface_pressure",
    "surface_pressure": "surface_pressure",
    "spressure": "surface_pressure",
    "velocity": "volume_velocity",
    "volume_velocity": "volume_velocity",
}


def canonical_task_name(predicted_feature_name):
    """Map legacy configuration names to one of the three supported tasks."""
    key = str(predicted_feature_name).strip().lower()
    try:
        return TASK_ALIASES[key]
    except KeyError as error:
        supported = ", ".join(sorted(TASK_ALIASES))
        raise ValueError(
            f"Unsupported predicted_feature_name={predicted_feature_name!r}; "
            f"supported names: {supported}"
        ) from error


def task_shapes(sample, predicted_feature_name):
    """Return the input and target shapes selected from one sample."""
    return (
        tuple(_get_input(sample, predicted_feature_name).shape),
        tuple(_get_output(sample, predicted_feature_name).shape),
    )


def custom_collate_fn(batch):
    """
    Custom collate function for batch processing.
    Expects 6-tuples from `PerFileVTKDataset`:
        (model_input, seg_matrix, target, surface_mask, sim_id, metadata)
    """
    first = batch[0]
    n_fields = len(first)

    if n_fields != 6:
        raise ValueError(
            f"Unexpected number of fields in batch elements: {n_fields}. "
            "Expected 6-tuple: (model_input, seg_matrix, target, "
            "surface_mask, sim_id, metadata)."
        )

    input_list, seg_matrix_list, target_list, surface_mask_list, sim_ids, metadata = zip(*batch)

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

    shapes = [
        (tuple(x.shape), tuple(m.shape), tuple(y.shape), tuple(mask.shape))
        for x, m, y, mask, _, _ in batch
    ]
    if len(set(shapes)) != 1:
        raise ValueError(
            "Samples with different mesh shapes cannot share a batch. "
            f"Use batch_size=1 or implement padding; received shapes={shapes}."
        )

    input_batch = torch.stack([_to_tensor(x) for x in input_list], dim=0)
    seg_matrix_batch = torch.stack([_to_tensor(x) for x in seg_matrix_list], dim=0)
    target_batch = torch.stack([_to_tensor(x) for x in target_list], dim=0)
    surface_mask_batch = torch.stack(
        [torch.as_tensor(mask, dtype=torch.bool) for mask in surface_mask_list], dim=0
    )

    return (
        input_batch,
        seg_matrix_batch,
        target_batch,
        surface_mask_batch,
        list(sim_ids),
        list(metadata),
    )


class _NumpyCoreAliasUnpickler(pickle.Unpickler):
    """
    Unpickler that remaps modules from environments that saved pickles with
    old/private NumPy package paths (e.g., 'numpy._core') to current paths
    (e.g., 'numpy.core'). This helps load pickles across NumPy versions.
    """

    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        # Add other common aliases here if encountered in the future
        # "numpy._multiarray_umath": "numpy.core._multiarray_umath",
    }

    def find_class(self, module, name):
        remapped_module = self.MODULE_ALIASES.get(module, module)
        return super().find_class(remapped_module, name)


class VTKDataset(Dataset):
    """In-memory dataset exposing model input, segmentation and CFD targets."""

    def __init__(self, data_dict, indices, predicted_feature_name="pressure"):
        self.data = []
        self.indices = indices
        self.sim_ids = []  # Track simulation IDs

        # Extract data for specified indices
        for idx in indices:
            if idx in data_dict and idx != "normalization_scalars":
                sample = data_dict[idx]
                model_input = _get_input(sample, predicted_feature_name)

                seg_matrix = _get_seg_matrix(sample, len(model_input), predicted_feature_name)
                target = _get_output(sample, predicted_feature_name)
                surface_mask = _get_surface_mask(
                    sample, len(target), predicted_feature_name
                )
                metadata = _get_metadata(
                    sample, idx, predicted_feature_name=predicted_feature_name
                )
                self.data.append((model_input, seg_matrix, target, surface_mask, metadata))
                self.sim_ids.append(idx)  # Store the simulation ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_input, seg_matrix, target, surface_mask, metadata = self.data[idx]
        sim_id = self.sim_ids[idx]

        return model_input, seg_matrix, target, surface_mask, sim_id, metadata


def _get_combined_input(sample):
    if "input" in sample:
        features = np.asarray(sample["input"], dtype=np.float32)
    elif "x" in sample:
        features = np.asarray(sample["x"], dtype=np.float32)
    elif "features_6d" in sample:
        features = np.asarray(sample["features_6d"], dtype=np.float32)
    elif "coor" in sample and "normals" in sample:
        features = np.concatenate(
            [np.asarray(sample["coor"]), np.asarray(sample["normals"])], axis=1
        ).astype(np.float32, copy=False)
    else:
        raise KeyError("Sample must contain input/x, features_6d, or coor and normals")
    if features.ndim != 2 or features.shape[1] < 3:
        raise ValueError(f"Expected input shape (N, C>=3), got {features.shape}")
    return features


def _get_input(sample, predicted_feature_name="p_and_v"):
    """Select normalized input features for the configured physical task."""
    task = canonical_task_name(predicted_feature_name)
    if task == "surface_pressure" and "x_surface" in sample:
        features = np.asarray(sample["x_surface"], dtype=np.float32)
    elif task == "volume_velocity" and "x_volume" in sample:
        features = np.asarray(sample["x_volume"], dtype=np.float32)
    else:
        features = _get_combined_input(sample)
        if task != "combined":
            mask = _get_combined_surface_mask(sample, len(features))
            features = features[mask if task == "surface_pressure" else ~mask, :3]
    if features.ndim != 2:
        raise ValueError(f"Expected two-dimensional model input, got {features.shape}")
    expected_channels = 7 if task == "combined" else 3
    if features.shape[1] != expected_channels:
        raise ValueError(
            f"Task {task!r} expects input shape (N,{expected_channels}), "
            f"got {features.shape}"
        )
    return torch.as_tensor(features, dtype=torch.float32)


def _get_combined_output(sample):
    if "output" in sample:
        target = np.asarray(sample["output"], dtype=np.float32)
    elif "y" in sample:
        target = np.asarray(sample["y"], dtype=np.float32)
    elif "pressure" in sample:
        pressure = np.asarray(sample["pressure"], dtype=np.float32)
        target = pressure[:, None] if pressure.ndim == 1 else pressure
    else:
        raise KeyError("Sample must contain output/y or pressure")
    if target.ndim != 2:
        raise ValueError(f"Expected target shape (N,C), got {target.shape}")
    return target


def _get_output(sample, predicted_feature_name="p_and_v"):
    """Select normalized pressure, velocity, or combined targets."""
    task = canonical_task_name(predicted_feature_name)
    if task == "surface_pressure" and "y_spressure" in sample:
        target = np.asarray(sample["y_spressure"], dtype=np.float32)
    elif task == "volume_velocity" and "y_velocity" in sample:
        target = np.asarray(sample["y_velocity"], dtype=np.float32)
    else:
        combined = _get_combined_output(sample)
        if task == "combined":
            target = combined
        else:
            mask = _get_combined_surface_mask(sample, len(combined))
            if task == "surface_pressure":
                target = combined[mask, -1:]
            else:
                target = combined[~mask, :3]
    expected_channels = {"combined": 4, "surface_pressure": 1, "volume_velocity": 3}[task]
    if target.ndim != 2 or target.shape[1] != expected_channels:
        raise ValueError(
            f"Task {task!r} expects target shape (N,{expected_channels}), got {target.shape}"
        )
    return torch.as_tensor(target, dtype=torch.float32)


def _get_combined_surface_mask(sample, num_nodes):
    mask = sample.get("surface_mask", sample.get("surf"))
    if mask is None:
        if "pressure" in sample:
            mask = np.ones(num_nodes, dtype=bool)
        else:
            raise KeyError("Four-channel CFD samples must contain surface_mask or surf")
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != num_nodes:
        raise ValueError(f"surface_mask length {len(mask)} != number of nodes {num_nodes}")
    return mask


def _get_surface_mask(sample, num_nodes, predicted_feature_name="p_and_v"):
    task = canonical_task_name(predicted_feature_name)
    if task == "surface_pressure":
        mask = np.ones(num_nodes, dtype=bool)
    elif task == "volume_velocity":
        mask = np.zeros(num_nodes, dtype=bool)
    else:
        mask = _get_combined_surface_mask(sample, num_nodes)
    return torch.as_tensor(mask, dtype=torch.bool)


def _get_seg_matrix(sample, num_nodes, predicted_feature_name="p_and_v"):
    if "seg_matrix" not in sample:
        return torch.full((1, num_nodes), 1.0 / num_nodes, dtype=torch.float32)
    matrix = sample["seg_matrix"]
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"seg_matrix must be two-dimensional, got {matrix.shape}")
    if matrix.shape[1] != num_nodes:
        task = canonical_task_name(predicted_feature_name)
        if task == "combined":
            raise ValueError(
                f"seg_matrix has {matrix.shape[1]} nodes but input has {num_nodes}"
            )
        source_mask = _get_combined_surface_mask(sample, matrix.shape[1])
        matrix = matrix[:, source_mask if task == "surface_pressure" else ~source_mask]
        row_sum = matrix.sum(axis=1, keepdims=True)
        nonempty = row_sum[:, 0] > 0.0
        matrix = matrix[nonempty]
        matrix /= matrix.sum(axis=1, keepdims=True)
    return torch.as_tensor(matrix, dtype=torch.float32)


def _get_metadata(
    sample,
    sim_id,
    data_root=None,
    vtk_data_path=None,
    predicted_feature_name="p_and_v",
):
    source = sample.get("source", {})
    if vtk_data_path is not None:
        pressure_path = Path(vtk_data_path) / f"{sim_id}_pressure.vtk"
        if not pressure_path.is_file():
            raise FileNotFoundError(
                f"Pressure VTK for sample {sim_id} not found: {pressure_path}"
            )
    else:
        pressure_path = source.get("pressure_ref")
        if data_root is not None and (
            not pressure_path or not Path(pressure_path).exists()
        ):
            pressure_path = Path(data_root).parent / "raw_data" / f"{sim_id}_pressure.vtk"

    pressure_path = str(pressure_path) if pressure_path else None
    return {
        "sample_id": sample.get("sample_id", str(sim_id)),
        "pressure_path": pressure_path,
        "has_seg_matrix": "seg_matrix" in sample,
        "task": canonical_task_name(predicted_feature_name),
    }


class PerFileVTKDataset(Dataset):
    """
    Dataset variant that loads **one simulation per file** on demand.

    Expected layout (can be adjusted if your naming is different):
        data_root/
            1.pkl
            2.pkl
            ...

    Each per-index pickle is expected to contain the same fields that were
    previously stored in `data_dict[idx]`.
    """

    def __init__(
        self,
        data_root,
        indices,
        predicted_feature_name="pressure",
        vtk_data_path=None,
    ):
        if isinstance(data_root, (str, os.PathLike)):
            data_root = [data_root]
        if not isinstance(data_root, Sequence) or not data_root:
            raise TypeError("data_root must be a directory or a non-empty sequence of directories")
        self.data_roots = [Path(root) for root in data_root]
        self.indices = indices
        self.predicted_feature_name = predicted_feature_name
        self.vtk_data_path = Path(vtk_data_path) if vtk_data_path else None
        if self.vtk_data_path is not None and not self.vtk_data_path.is_dir():
            raise FileNotFoundError(f"VTK data directory not found: {self.vtk_data_path}")

    def _sample_path(self, sim_id):
        # If your naming is different (e.g. f"sample_{sim_id}.pkl"),
        # we can tweak this in the next step.
        for data_root in self.data_roots:
            candidate = data_root / f"{sim_id}.pkl"
            if candidate.exists():
                return candidate
        roots = ", ".join(str(root) for root in self.data_roots)
        raise FileNotFoundError(f"Sample {sim_id}.pkl not found in: {roots}")

    def _load_sample(self, sim_id):
        file_path = self._sample_path(sim_id)
        with open(file_path, "rb") as f:
            sample = pickle.load(f)
        return sample

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map from dataset index to simulation id
        sim_id = self.indices[idx]
        sample = self._load_sample(sim_id)
        model_input = _get_input(sample, self.predicted_feature_name)

        seg_matrix = _get_seg_matrix(
            sample, len(model_input), self.predicted_feature_name
        )
        target = _get_output(sample, self.predicted_feature_name)
        surface_mask = _get_surface_mask(
            sample, len(target), self.predicted_feature_name
        )
        metadata = _get_metadata(
            sample,
            sim_id,
            self._sample_path(sim_id).parent,
            vtk_data_path=self.vtk_data_path,
            predicted_feature_name=self.predicted_feature_name,
        )

        return model_input, seg_matrix, target, surface_mask, sim_id, metadata



def create_data_loaders(
    data_source,
    batch_size=1,
    train_index=[66],
    val_index=[66],
    test_index=[66],
    shuffle=True,
    predicted_feature_name="pressure",
    vtk_data_path=None,
):
    """
    Create train, validation, and test data loaders based on specific indices.

    There are two modes:
      1. Legacy in-memory mode: `data_source` is a dict like the old `data_dict`.
      2. Per-file lazy mode:  `data_source` is a directory path. Each sample
         with index `i` is loaded from a file like `<data_source>/<i>.pkl`.

    Args:
        data_source (dict | str): Either the in-memory data_dict, or the
            directory containing one file per sample.
        batch_size (int): Batch size for data loaders
        train_index (list): List of indices for training data
        val_index (list): List of indices for validation data
        test_index (list): List of indices for testing data
        shuffle (bool): Whether to shuffle training data
        predicted_feature_name (str): Name of the feature to predict
            ("pressure", "x_force", "y_force", "z_force")
        vtk_data_path (str | None): Directory containing original pressure VTK
            files named <simulation_id>_pressure.vtk.

    Returns:
        tuple: (train_loader, val_loader, test_loader, normalization_scalars)
            In per-file mode, `normalization_scalars` is returned as None
            (we can wire this up in a later step if needed).
    """

    is_dict_mode = isinstance(data_source, dict)
    is_path_mode = isinstance(data_source, (str, os.PathLike, Sequence)) and not isinstance(data_source, dict)

    if not (is_dict_mode or is_path_mode):
        raise TypeError(
            f"data_source must be either a dict or a str (path), got {type(data_source)}"
        )

    # Use provided indices
    train_indices = train_index
    val_indices = val_index
    test_indices = test_index

    if is_dict_mode:
        data_dict = data_source
        # Get simulation indices (excluding normalization_scalars)
        sim_indices = [idx for idx in data_dict.keys() if idx != "normalization_scalars"]
        sim_indices.sort()
        print(f"Available simulation indices: {sim_indices}")

        normalization_scalars = data_dict.get("normalization_scalars", None)

        # Create datasets with predicted_feature_name (legacy in-memory)
        train_dataset = VTKDataset(
            data_dict, train_indices, predicted_feature_name=predicted_feature_name
        )
        val_dataset = VTKDataset(
            data_dict, val_indices, predicted_feature_name=predicted_feature_name
        )
        test_dataset = VTKDataset(
            data_dict, test_indices, predicted_feature_name=predicted_feature_name
        )

    else:
        data_root = data_source
        print(f"Per-file data root(s): {data_root}")
        # We don't know normalization_scalars location yet in per-file mode.
        normalization_scalars = None

        # Create datasets that load one file per sample on demand
        train_dataset = PerFileVTKDataset(
            data_root,
            train_indices,
            predicted_feature_name=predicted_feature_name,
            vtk_data_path=vtk_data_path,
        )
        val_dataset = PerFileVTKDataset(
            data_root,
            val_indices,
            predicted_feature_name=predicted_feature_name,
            vtk_data_path=vtk_data_path,
        )
        test_dataset = PerFileVTKDataset(
            data_root,
            test_indices,
            predicted_feature_name=predicted_feature_name,
            vtk_data_path=vtk_data_path,
        )

    print("Data split:")
    print(f"  - Training: {len(train_indices)} simulations (indices: {train_indices})")
    print(f"  - Validation: {len(val_indices)} simulations (indices: {val_indices})")
    print(f"  - Testing: {len(test_indices)} simulations (indices: {test_indices})")
    print(f"  - Predicting: {predicted_feature_name}")

    # Create data loaders with custom collate function for sparse tensors
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader, normalization_scalars


if __name__ == "__main__":

    DATA_PATH = "/taiga/illinois/eng/cee/meidani/Vincent/FC4NO/driver_plus/PressureVTK/processed/"
    with open(DATA_PATH + "normalization_scalars.pkl", "rb") as f:
        normalization_scalars = pickle.load(f)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        DATA_PATH, batch_size=1
    )

    print(f"\nDataLoader test:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # print the first batch
    for batch in train_loader:
        coorf, seg_matrix, target, surface_mask, sim_id, metadata = batch
        print(coorf.shape)
        print(seg_matrix.shape)
        print(target.shape)
        print(surface_mask.shape)
        print(sim_id)
        print(metadata)
        break
