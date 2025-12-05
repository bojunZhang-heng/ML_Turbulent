import gc
import torch
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from dataset.serialize import Point


@dataclass
class DrivAerMLStats:
    raw_pos_min: tuple[float] = (-40.0,)
    raw_pos_max: tuple[float] = (80.0,)
    surface_pressure_mean: tuple[float] = (-2.29772e02,)
    surface_pressure_std: tuple[float] = (2.69345e02,)
    surface_wallshearstress_mean: tuple[float, float, float] = (-1.20054e00, 1.49358e-03, -7.20107e-02)
    surface_wallshearstress_std: tuple[float, float, float] = (2.07670e00, 1.35628e00, 1.11426e00)
    volume_totalpcoeff_mean: tuple[float] = (1.71387e-01,)
    volume_totalpcoeff_std: tuple[float] = (5.00826e-01,)
    volume_velocity_mean: tuple[float, float, float] = (1.67909e01, -3.82238e-02, 4.07968e-01)
    volume_velocity_std: tuple[float, float, float] = (1.64115e01, 8.63614e00, 6.64996e00)
    volume_vorticity_logscale_mean: tuple[float, float, float] = (-1.47814e-02, 7.87642e-01, 2.81023e-03)
    volume_vorticity_logscale_std: tuple[float, float, float] = (5.45681e00, 5.77081e00, 5.46175e00)


class DrivAerMLDefaultSplitIDs:
    # fmt: off
#    train = {
#        110, 111, 112, 113, 114, 115, 117, 118, 119,
#        130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
#        140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
#        150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
#        160, 161, 162, 163, 164, 165, 166, 168, 169,
#        170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
#        180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
#        230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
#        240, 241, 242, 243, 244, 245, 246, 247, 249,
#        190, 191, 192, 193, 194, 195, 196, 197,
#    }
#
#    val = {
#        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#        20, 21, 22, 23, 24, 25,
#        120, 121, 122, 123, 125, 126, 127, 129,
#        200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
#        210, 212, 213, 214, 215, 216, 217, 219,
#    }
#
#    test = {
#        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
#        220, 222, 223, 224, 225, 226, 227, 228, 229,
#    }
    train = {
        115, 141, 144, 150, 154, 162, 176, 183, 232, 242,
        111, 119, 122, 129, 135, 140, 146, 151, 155, 159,
        164, 169, 177, 184, 189, 230, 233, 235, 240, 244,
        110, 113, 118, 123, 130, 134, 137, 142, 147, 152,
        156, 160, 165, 170, 172, 178, 180, 185, 188, 231,
        234, 236, 238, 241, 244, 246, 249, 112, 117, 121,
        125, 126, 131, 133, 136, 138, 143, 145, 148, 149,
        153, 157, 158, 161, 163, 166, 168, 171, 173, 174,
        175, 179, 181, 182, 186, 187, 237, 239, 245, 247,
        114, 120, 127, 132, 139,
    }

    val = {
        10, 13, 16, 19, 22, 25, 190, 193, 196, 200,
        203, 206, 209, 212, 215, 217,
    }

    test = {
        11, 14, 17, 20, 23, 191, 194, 197, 201, 204,
        207, 210, 213, 216, 219,
    }

    tutorial = {11}
    # The following design IDs are not available in the dataset. They are held back by the authors for testing purposes.
    hidden_val = {167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473}
    # fmt: on


class DrivAerMLDataset(Dataset):
    """Dataset implementation for DrivAerML that supports both local and AISTORE storage.

    Args:
        split: Which split to use.
        root: path to the processed dataset, e.g. .../drivaerml_processed/subsampled_10x.
    """

    def __init__(self, root: str, split: str):
        super().__init__()
        self.root = Path(root).expanduser()
        if split == "train":
            design_ids = DrivAerMLDefaultSplitIDs.train
        elif split == "val":
            design_ids = DrivAerMLDefaultSplitIDs.val
        elif split == "test":
            design_ids = DrivAerMLDefaultSplitIDs.test
        elif split == "tutorial":
            design_ids = DrivAerMLDefaultSplitIDs.tutorial
        else:
            raise NotImplementedError
        # convert sets to list
        self.design_ids = sorted(design_ids)

    def __len__(self):
        return len(self.design_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取单个样本的所有数据 - 只包含需要的字段"""
        sample = {
            'index': idx,  # 用于随机种子
            'surface_position_vtp': self.getitem_surface_position_vtp(idx),
            'surface_pressure': self.getitem_surface_pressure(idx),
            'surface_wallshearstress': self.getitem_surface_wallshearstress(idx),
            'volume_position': self.getitem_volume_position(idx),
            'volume_totalpcoeff': self.getitem_volume_totalpcoeff(idx),
            'volume_velocity': self.getitem_volume_velocity(idx),
        }
        return sample

    @staticmethod
    def get_normalization_stats():
        return DrivAerMLStats()

    def _load(self, idx: int, filename_base: str) -> torch.Tensor:
        """
        仅保留预抽样优先加载 + 回退到完整文件的逻辑（已移除所有随机采样代码）。

        行为：
          - 优先尝试 {filename_base}_v2.npy / {filename_base}_v2.pt（若存在直接返回）
          - 否则尝试 {filename_base}.npy（一次性读入并返回）
          - 否则尝试 {filename_base}.pt（torch.load 并返回）
          - 三者都不存在则抛出 FileNotFoundError

        注意：返回的是 CPU tensor（caller 若需 GPU 请自行 .to(device)）。
        """
        design_id = self.design_ids[idx]
        design_uri = Path(self.root) / f"run_{design_id}"
        assert design_uri.exists(), f"{design_uri.as_posix()} does not exist"

        # 先查找预抽样文件（优先）
#        v2_np_path = design_uri / f"{filename_base}_v2.npy"
#        v2_pt_path = design_uri / f"{filename_base}_v2.pt"
        np_path = design_uri / f"{filename_base}.npy"
        pt_path = design_uri / f"{filename_base}.pt"

        # 如果存在预抽样的 .npy，直接读入并返回
#        if v2_np_path.exists():
#            arr = np.load(v2_np_path, mmap_mode=None)   # 预抽样文件通常较小，直接读入
#            return torch.from_numpy(np.array(arr)).contiguous().clone()

        # 如果存在预抽样的 .pt，直接加载并返回
#        if v2_pt_path.exists():
#            tensor = torch.load(v2_pt_path, map_location="cpu")
#            return tensor.contiguous().clone()

        # 回退到完整的 .npy（一次性载入）
        if np_path.exists():
            arr = np.load(np_path, mmap_mode=None)
            return torch.from_numpy(np.array(arr)).contiguous().clone()

        # 回退到完整的 .pt
        if pt_path.exists():
            tensor = torch.load(pt_path, map_location="cpu")
            return tensor.contiguous().clone()

        raise FileNotFoundError(f"Neither {np_path} nor {pt_path} (nor _v2 versions) found for run_{design_id}")

#    def _load(self, idx: int, filename: str) -> torch.Tensor:
#        design_uri = self.root / f"run_{self.design_ids[idx]}"
#        assert design_uri.exists(), f"{design_uri.as_posix()} does not exist"
#        return torch.load(design_uri / filename, weights_only=True)

    def getitem_surface_position_vtp(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions from the CFD mesh (num_surface_points, 3)"""
        #return self._load(idx=idx, filename="surface_position_vtp.npy")
        return self._load(idx=idx, filename_base="surface_position_vtp")

    def getitem_surface_position_stl(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions from the STL file (num_surface_points, 3)"""
        return self._load(idx=idx, filename="surface_position_stl_resampled100k.npy")

    def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves surface pressures (num_surface_points, 1)"""
        return self._load(idx=idx, filename_base="surface_pressure").unsqueeze(1)
        #return self._load(idx=idx, filename="surface_pressure.npy").unsqueeze(1)

    def getitem_surface_wallshearstress(self, idx: int) -> torch.Tensor:
        """Retrieves surface wallshearstress (num_surface_points, 3)"""
        return self._load(idx=idx, filename_base="surface_wallshearstress")
        #return self._load(idx=idx, filename="surface_wallshearstress.npy")

    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (num_volume_points, 3)"""
        #return self._load(idx=idx, filename="volume_cell_position.npy")
        return self._load(idx=idx, filename_base="volume_cell_position")

    def getitem_volume_totalpcoeff(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressures (num_volume_points, 1)"""
       # return self._load(idx=idx, filename="volume_cell_totalpcoeff.npy").unsqueeze(1)
        return self._load(idx=idx, filename_base="volume_cell_totalpcoeff").unsqueeze(1)

    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (num_volume_points, 3)"""
       # return self._load(idx=idx, filename="volume_cell_velocity.npy")
        return self._load(idx=idx, filename_base="volume_cell_velocity")

#    def getitem_volume_vorticity(self, idx: int) -> torch.Tensor:
#        """Retrieves volume vorticity (num_volume_points, 3)"""
#        return self._load(idx=idx, filename="volume_cell_vorticity.npy")
