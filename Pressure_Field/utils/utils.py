# utils.py
"""

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Utility functions for the DrivAerNet pressure prediction project.

This module provides helper functions for logging, random seed setup,
visualization, and other common operations.
"""

import os
import random
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
import pyvista as pv
from data_loader import PRESSURE_MEAN, PRESSURE_STD

def knn(x, k):
    """
    k-nearest neighbors algorithm.

    Args:
        x: Input tensor of shape (batch_size, num_points, feature_dim)
        k: Number of neighbors to consider

    Returns:
        Indices of k-nearest neighbors for each point
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))             # [batch_size, num_points, num_points]
    xx = torch.sum(x ** 2, dim=2, keepdim=True)                 # [batch_size, num_points, 1]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)        # [batch_size, num_points, num_points]

    idx = pairwise_distance.topk(k=k, dim=-1)[1]                # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Construct edge features for graph convolution.

    When you create this function first, Just use 1 batch_size, 2 dims, 2 points
    to Understand this code

    Then Enlarge it meeting the physical need

    Args:
        x: Input tensor of shape (batch_size, feature_dim, num_points)
        k: Number of neighbors to use for graph construction
        idx: Optional pre-computed nearest neighbor indices
        dim9: Whether to use additional dimensional features

    Returns:
        Edge features for graph convolution
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (batch_size, 2*num_dims, num_points, k)

def setup_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file=None, level=logging.INFO):
    """
    Set up the logger for the application.

    Args:
        log_file: Path to the log file
        level: Logging level
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)


def visualize_pressure_field(points, true_pressure, pred_pressure, output_path):
    """
    Visualize the true and predicted pressure fields on a 3D model.

    Args:
        points: 3D point cloud coordinates
        true_pressure: Ground truth pressure values
        pred_pressure: Predicted pressure values
        output_path: Path to save the visualization
    """
    # Reshape the points to be (num_points, 3)
    # The issue is points has shape (3, num_points)
    if points.shape[0] == 3 and points.ndim == 2:
        # Transpose to get (num_points, 3)
        points = points.T

    # Make sure pressure values are 1D arrays
    if true_pressure.ndim > 1:
        true_pressure = true_pressure.squeeze()
    if pred_pressure.ndim > 1:
        pred_pressure = pred_pressure.squeeze()

    # Denormalize pressure values if needed
    # true_pressure = true_pressure * PRESSURE_STD + PRESSURE_MEAN
    # pred_pressure = pred_pressure * PRESSURE_STD + PRESSURE_MEAN

    # Create PyVista point clouds
    true_cloud = pv.PolyData(points)
    true_cloud.point_data['pressure'] = true_pressure

    pred_cloud = pv.PolyData(points)
    pred_cloud.point_data['pressure'] = pred_pressure

    # Create PyVista plotter
    plotter = pv.Plotter(shape=(1, 2), off_screen=True)

    # Plot true pressure
    plotter.subplot(0, 0)
    plotter.add_text("True Pressure", font_size=16)
    plotter.add_mesh(true_cloud, scalars='pressure', cmap='jet', point_size=5)

    # Plot predicted pressure
    plotter.subplot(0, 1)
    plotter.add_text("Predicted Pressure", font_size=16)
    plotter.add_mesh(pred_cloud, scalars='pressure', cmap='jet', point_size=5)

    # Save figure
    plotter.screenshot(output_path)
    plotter.close()

def plot_error_distribution(true_pressure, pred_pressure, output_path):
    """
    Plot the distribution of prediction errors.

    Args:
        true_pressure: Ground truth pressure values
        pred_pressure: Predicted pressure values
        output_path: Path to save the plot
    """
    # Calculate absolute errors
    errors = np.abs(true_pressure - pred_pressure)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def save_training_curve(train_losses, val_losses, output_path):
    """
    Save a plot of training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', linestyle='-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def calculate_metrics(true_values, predicted_values):
    """
    Calculate various evaluation metrics.

    Args:
        true_values: Ground truth values
        predicted_values: Predicted values

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if tensors
    if torch.is_tensor(true_values):
        true_values = true_values.cpu().numpy()
    if torch.is_tensor(predicted_values):
        predicted_values = predicted_values.cpu().numpy()

    # Mean Squared Error
    mse = np.mean((true_values - predicted_values) ** 2)

    # Mean Absolute Error
    mae = np.mean(np.abs(true_values - predicted_values))

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Maximum Absolute Error
    max_error = np.max(np.abs(true_values - predicted_values))

    # Relative L2 Error (normalized)
    rel_l2 = np.mean(np.linalg.norm(true_values - predicted_values, axis=0) /
                     np.linalg.norm(true_values, axis=0))

    # Relative L1 Error (normalized)
    rel_l1 = np.mean(np.sum(np.abs(true_values - predicted_values), axis=0) /
                     np.sum(np.abs(true_values), axis=0))

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Max_Error': max_error,
        'Rel_L2': rel_l2,
        'Rel_L1': rel_l1
    }
