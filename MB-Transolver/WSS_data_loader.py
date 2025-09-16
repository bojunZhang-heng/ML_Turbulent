# data_loader.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Data loading utilities for the DrivAerNet++ dataset.

This module provides functionality for loading and preprocessing point cloud data
with wall shear stress field information from the DrivAerNet++ dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torch.distributed as dist
import pyvista as pv
import logging
from colorama import Fore, Style

class SurfaceWSSDataset(Dataset):
    """
    Dataset class for loading and preprocessing surface wall shear stress data from DrivAerNet++ VTK files.

    This dataset handles loading surface meshes with wall shear stress field data,
    sampling points, and caching processed data for faster loading.
    """

    def __init__(self, root_dir: str, num_points: int, preprocess=False, cache_dir=None):
        """
        Initializes the SurfaceWSSDataset instance.

        Args:
            root_dir: Directory containing the VTK files for the car surface meshes.
            num_points: Fixed number of points to sample from each 3D model.
            preprocess: Flag to indicate if preprocessing should occur or not.
            cache_dir: Directory where the preprocessed files (NPZ) are stored.
        """
        self.root_dir = root_dir
        self.vtk_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.vtk')]
        self.num_points = num_points
        self.preprocess = preprocess
        self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, "processed_data")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.vtk_files)

    def _get_cache_path(self, vtk_file_path):
        """Get the corresponding .npz file path for a given .vtk file."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', '.npz')
        return os.path.join(self.cache_dir, base_name)

    def _save_to_cache(self, cache_path, point_cloud, WallShearStress):
        """Save preprocessed point cloud and wall shear stress data into an npz file."""
        np.savez_compressed(cache_path, points=point_cloud.points, WallShearStress=WallShearStress)

    def _load_from_cache(self, cache_path):
        """Load preprocessed point cloud and wall shear stress data from an npz file."""
        data = np.load(cache_path)
        point_cloud = pv.PolyData(data['points'])
        WallShearStress = data['WallShearStress']
        return point_cloud, WallShearStress

    def sample_point_cloud_with_WSS(self, mesh, n_points=5000):
        """
        Sample n_points from the surface mesh and get corresponding WallShearStress values.

        Args:
            mesh: PyVista mesh object with WallShearStress data stored in point_data.
            n_points: Number of points to sample.

        Returns:
            A tuple containing the sampled point cloud and corresponding WallShearStress.
        """
        if mesh.n_points > n_points:
            indices = np.random.choice(mesh.n_points, n_points, replace=False)
        else:
            indices = np.arange(mesh.n_points)
            logging.info(f"Mesh has only {mesh.n_points} points. Using all available points.")

        sampled_points = mesh.points[indices]
        sampled_WallShearStress = mesh.point_data['wall'][indices]  # Assuming WallShearStress data is stored under key 'w'
        sampled_WallShearStress = sampled_WallShearStress.flatten()  # Ensure it's a flat array

        return pv.PolyData(sampled_points), sampled_WallShearStress

    def __getitem__(self, idx):
        vtk_file_path = self.vtk_files[idx]
        cache_path = self._get_cache_path(vtk_file_path)

        # Check if the data is already cached
        if os.path.exists(cache_path):
            logging.info(f"Loading cached data from {cache_path}")
            point_cloud, WallShearStress = self._load_from_cache(cache_path)
        else:
            if self.preprocess:
                logging.info(f"Preprocessing and caching data for {vtk_file_path}")
                try:
                    mesh = pv.read(vtk_file_path)
                except Exception as e:
                    logging.error(f"Failed to load VTK file: {vtk_file_path}. Error: {e}")
                    return None, None  # Skip the file and return None

                point_cloud, WallShearStress = self.sample_point_cloud_with_WSS(mesh, self.num_points)

                # Cache the sampled data to a new file
                self._save_to_cache(cache_path, point_cloud, WallShearStress)
            else:
                logging.error(f"Cache file not found for {vtk_file_path} and preprocessing is disabled.")
                return None, None  # Return None if preprocessing is disabled and cache doesn't exist

        point_cloud_np = np.array(point_cloud.points)
        point_cloud_tensor = torch.tensor(point_cloud_np.T[np.newaxis, :, :], dtype=torch.float32)
        WallShearStress_tensor = torch.tensor(WallShearStress[np.newaxis, :], dtype=torch.float32)

        return point_cloud_tensor, WallShearStress

