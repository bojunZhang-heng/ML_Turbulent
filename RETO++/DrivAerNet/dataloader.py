import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from pathlib import Path
from collections.abc import Sequence
from scipy.sparse import csr_matrix


def custom_collate_fn(batch):
    """
    Custom collate function for batch processing.
    Expects 5-tuples from `PerFileVTKDataset`:
        (coorf, raw_coords, seg_matrix, p, sim_id)
    """
    first = batch[0]
    n_fields = len(first)

    if n_fields != 5:
        raise ValueError(
            f"Unexpected number of fields in batch elements: {n_fields}. "
            "Expected 5-tuple: (coorf, raw_coords, seg_matrix, p, sim_id)."
        )

    coorf_list, raw_coords_list, seg_matrix_list, p_list, sim_id = zip(*batch)

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)

    shapes = [
        (tuple(x.shape), tuple(raw.shape), tuple(m.shape), tuple(y.shape))
        for x, raw, m, y, _ in batch
    ]
    if len(set(shapes)) != 1:
        raise ValueError(
            "Samples with different mesh shapes cannot share a batch. "
            f"Use batch_size=1 or implement padding; received shapes={shapes}."
        )

    coorf_batch = torch.stack([_to_tensor(x) for x in coorf_list], dim=0)
    raw_coords_batch = torch.stack([_to_tensor(x) for x in raw_coords_list], dim=0)
    seg_matrix_batch = torch.stack([_to_tensor(x) for x in seg_matrix_list], dim=0)
    p_batch = torch.stack([_to_tensor(x) for x in p_list], dim=0)
    sim_id_batch = list(sim_id)

    return coorf_batch, raw_coords_batch, seg_matrix_batch, p_batch, sim_id_batch


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
    """
    In-memory dataset exposing only input, geometry partition and pressure.
    """

    def __init__(
        self,
        data_dict,
        indices,
        predicted_feature_name="pressure",
        coordinate_only=False,
    ):
        self.data = []
        self.indices = indices
        self.sim_ids = []  # Track simulation IDs

        # Extract data for specified indices
        for idx in indices:
            if idx in data_dict and idx != "normalization_scalars":
                sample = data_dict[idx]
                model_input = _get_input(sample, coordinate_only=coordinate_only)

                if "seg_matrix" in sample:
                    # Check if seg_matrix is already a sparse matrix
                    if isinstance(data_dict[idx]["seg_matrix"], csr_matrix):
                        # Convert sparse matrix to dense tensor
                        seg_matrix = torch.as_tensor(sample["seg_matrix"].toarray(), dtype=torch.float32)
                    else:
                        # Convert dense matrix to tensor
                        seg_matrix = torch.as_tensor(sample["seg_matrix"], dtype=torch.float32)
                else:
                    # Create empty tensor if no seg_matrix
                    seg_matrix = torch.zeros((1, model_input.shape[0]), dtype=torch.float32)

                raw_coords = _get_raw_coords(sample, data_dict.get("normalization_scalars"))
                target = _get_pressure(sample)
                self.data.append((model_input, raw_coords, seg_matrix, target))
                self.sim_ids.append(idx)  # Store the simulation ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model_input, raw_coords, seg_matrix, target = self.data[idx]
        sim_id = self.sim_ids[idx]

        return model_input, raw_coords, seg_matrix, target, sim_id


def _get_input(sample, coordinate_only=False):
    """Return normalized XYZ, optionally followed by the three normals."""
    if coordinate_only:
        if "coor" in sample:
            features = np.asarray(sample["coor"], dtype=np.float32)
        elif "features_6d" in sample:
            features = np.asarray(sample["features_6d"], dtype=np.float32)[..., :3]
        else:
            raise KeyError("Sample must contain coor or features_6d")
        if features.ndim != 2 or features.shape[1] != 3:
            raise ValueError(f"Expected coordinate input shape (N, 3), got {features.shape}")
        return torch.as_tensor(features, dtype=torch.float32)

    if "features_6d" in sample:
        features = np.asarray(sample["features_6d"], dtype=np.float32)
    elif "coor" in sample and "normals" in sample:
        features = np.concatenate(
            [np.asarray(sample["coor"]), np.asarray(sample["normals"])], axis=1
        ).astype(np.float32, copy=False)
    else:
        raise KeyError("Sample must contain features_6d or both coor and normals")
    if features.ndim != 2 or features.shape[1] != 6:
        raise ValueError(f"Expected input shape (N, 6), got {features.shape}")
    return torch.as_tensor(features, dtype=torch.float32)


def _get_pressure(sample):
    if "pressure" not in sample:
        raise KeyError("Sample must contain the pressure field")
    pressure = np.asarray(sample["pressure"], dtype=np.float32)
    if pressure.ndim == 1:
        pressure = pressure[:, None]
    if pressure.ndim != 2 or pressure.shape[1] != 1:
        raise ValueError(f"Expected pressure shape (N, 1), got {pressure.shape}")
    return torch.as_tensor(pressure, dtype=torch.float32)


def _get_raw_coords(sample, normalization_scalars=None):
    """Return unnormalized XYZ coordinates for relative-position encoding."""
    if "coor_raw" in sample:
        raw_coords = np.asarray(sample["coor_raw"], dtype=np.float32)
    elif "coor" in sample and normalization_scalars is not None:
        coords = np.asarray(sample["coor"], dtype=np.float32)
        normalization = sample.get("normalization", {})
        coordinates_are_normalized = normalization.get("coor") == "min_max"
        if not normalization:
            coordinates_are_normalized = bool(
                np.all(coords >= -1e-5) and np.all(coords <= 1.0 + 1e-5)
            )
        if coordinates_are_normalized:
            coords_min = np.asarray(
                normalization_scalars["coords_min"], dtype=np.float32
            )
            coords_range = np.asarray(
                normalization_scalars["coords_range"], dtype=np.float32
            )
            raw_coords = coords * coords_range + coords_min
        else:
            raw_coords = coords
    elif "coor" in sample:
        raw_coords = np.asarray(sample["coor"], dtype=np.float32)
    else:
        raise KeyError(
            "Sample must contain coor_raw or coor for relative-position encoding."
        )
    if raw_coords.ndim != 2 or raw_coords.shape[1] != 3:
        raise ValueError(f"Expected raw coordinate shape (N, 3), got {raw_coords.shape}")
    return torch.as_tensor(raw_coords, dtype=torch.float32)


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
        coordinate_only=False,
    ):
        if isinstance(data_root, (str, os.PathLike)):
            data_root = [data_root]
        if not isinstance(data_root, Sequence) or not data_root:
            raise TypeError("data_root must be a directory or a non-empty sequence of directories")
        self.data_roots = [Path(root) for root in data_root]
        self.indices = indices
        self.coordinate_only = coordinate_only
        self.normalization_scalars = self._load_normalization_scalars()

    def _load_normalization_scalars(self):
        for data_root in self.data_roots:
            scalar_path = data_root / "normalization_scalars.pkl"
            if scalar_path.exists():
                with open(scalar_path, "rb") as handle:
                    return pickle.load(handle)
        return None

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
        coorf = _get_input(sample, coordinate_only=self.coordinate_only)
        raw_coords = _get_raw_coords(sample, self.normalization_scalars)

        # Ensure seg_matrix is dense (mirror behavior of VTKDataset)
        seg_matrix = sample.get("seg_matrix")
        if seg_matrix is None:
            seg_matrix = np.zeros((1, coorf.shape[0]), dtype=np.float32)
        if isinstance(seg_matrix, csr_matrix):
            seg_matrix = seg_matrix.toarray()
        seg_matrix = torch.as_tensor(seg_matrix, dtype=torch.float32)
        p = _get_pressure(sample)

        return coorf, raw_coords, seg_matrix, p, sim_id



def create_data_loaders(
    data_source,
    batch_size=1,
    train_index=[66],
    val_index=[66],
    test_index=[66],
    shuffle=True,
    predicted_feature_name="pressure",
    coordinate_only=False,
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
        coordinate_only (bool): Return normalized XYZ only instead of
            normalized XYZ concatenated with normals.

    Returns:
        tuple: (train_loader, val_loader, test_loader, normalization_scalars)
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
            data_dict,
            train_indices,
            predicted_feature_name=predicted_feature_name,
            coordinate_only=coordinate_only,
        )
        val_dataset = VTKDataset(
            data_dict,
            val_indices,
            predicted_feature_name=predicted_feature_name,
            coordinate_only=coordinate_only,
        )
        test_dataset = VTKDataset(
            data_dict,
            test_indices,
            predicted_feature_name=predicted_feature_name,
            coordinate_only=coordinate_only,
        )

    else:
        data_root = data_source
        print(f"Per-file data root(s): {data_root}")
        # Create datasets that load one file per sample on demand
        train_dataset = PerFileVTKDataset(
            data_root,
            train_indices,
            predicted_feature_name=predicted_feature_name,
            coordinate_only=coordinate_only,
        )
        val_dataset = PerFileVTKDataset(
            data_root,
            val_indices,
            predicted_feature_name=predicted_feature_name,
            coordinate_only=coordinate_only,
        )
        test_dataset = PerFileVTKDataset(
            data_root,
            test_indices,
            predicted_feature_name=predicted_feature_name,
            coordinate_only=coordinate_only,
        )
        normalization_scalars = train_dataset.normalization_scalars

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
        coorf, raw_coords, seg_matrix, p, sim_id = batch
        print(coorf.shape)
        print(raw_coords.shape)
        print(seg_matrix.shape)
        print(p.shape)
        print(sim_id)
        break
