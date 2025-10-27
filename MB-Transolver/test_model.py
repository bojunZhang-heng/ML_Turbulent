import os
import torch
import trimesh
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from scipy.spatial import cKDTree
import pandas as pd
import logging

from abupt_collator import AbuptCollator
from drivaerml_dataset import DrivAerMLDataset
from model import AnchoredBranchedUPT
from preprocessors import MomentNormalizationPreprocessor, PositionNormalizationPreprocessor
from streamline_visualization import plot_streamlines
from utils import plot_pointcloud_single, plot_pointcloud_double, set_seed
from tqdm import tqdm

# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # 输出到stdout
        # 如需保存到文件请取消下面的注释
        # logging.FileHandler("test_model.log", mode="w")
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# Load dataset
# ============================================================
root = "/work/mae-zhangbj/drivaerml"
dataset = DrivAerMLDataset(root=root, split="test", num_sample_frac=0.01)
raw_sample = dict(
    surface_position_vtp=dataset.getitem_surface_position_vtp(0),
    surface_pressure=dataset.getitem_surface_pressure(0),
    surface_wallshearstress=dataset.getitem_surface_wallshearstress(0),
    volume_position=dataset.getitem_volume_position(0),
    volume_totalpcoeff=dataset.getitem_volume_totalpcoeff(0),
    volume_velocity=dataset.getitem_volume_velocity(0),
    # volume_vorticity=dataset.getitem_volume_vorticity(0),
)

logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~raw_sample")
for key, value in raw_sample.items():
    logger.info(f"{key}: {value.shape}")

# ============================================================
# Collator
# ============================================================
collator = AbuptCollator(
    num_geometry_points=65536,
    num_surface_anchor_points=16384,
    num_volume_anchor_points=16384,
    num_geometry_supernodes=16384,
    use_query_positions=True,
    dataset=dataset,
)

# convert a list of samples to a preprocessed batch
batch = collator([raw_sample])

logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~batch")
for key, value in batch.items():
    logger.info(f"{key}: {value.shape}")

# ============================================================
# Load checkpoint
# ============================================================
local_dir = "./checkpoints"
os.makedirs(local_dir, exist_ok=True)

# ============================================================
# Model setup
# ============================================================
abupt = AnchoredBranchedUPT().to("cuda").eval()
checkpoint = torch.load("./checkpoints/ab-upt-drivaerml-tutorial.th", map_location="cuda", weights_only=True)
abupt.load_state_dict(checkpoint["state_dict"])

# ============================================================
# Move batch to GPU
# ============================================================
batch = {key: value.to("cuda") for key, value in batch.items()}

# extract target variables for anchor
target_surface_anchor_pressure = batch.pop("surface_anchor_pressure")
target_surface_anchor_wallshearstress = batch.pop("surface_anchor_wallshearstress")
target_volume_anchor_totalpcoeff = batch.pop("volume_anchor_totalpcoeff")
target_volume_anchor_velocity = batch.pop("volume_anchor_velocity")

# extract target variables for queries
target_surface_query_pressure = batch.pop("surface_query_pressure")
target_surface_query_wallshearstress = batch.pop("surface_query_wallshearstress")
target_volume_query_totalpcoeff = batch.pop("volume_query_totalpcoeff")
target_volume_query_velocity = batch.pop("volume_query_velocity")

# we dont need all 8M surface points for now
num_surface_queries = 16384
batch["surface_query_position"] = batch["surface_query_position"][:, :num_surface_queries]
target_surface_query_pressure = target_surface_query_pressure[:num_surface_queries]
target_surface_query_wallshearstress = target_surface_query_wallshearstress[:num_surface_queries]

logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input")
for key, value in batch.items():
    logger.info(f"{key}: {value.shape}")

# ============================================================
# Inference
# ============================================================
with torch.autocast(device_type="cuda", dtype=torch.float16), torch.no_grad():
    prediction = abupt(**batch)

for key, value in prediction.items():
    logger.info(f"{key}: {value.shape}")

# ============================================================
# Visualization
# ============================================================
save_path = "/work/mae-zhangbj/ML_Turbulent/Current_work/MB-Transolver/figure"
surface_query_positions_plot = batch["surface_query_position"].cpu().squeeze(0)
plot_pointcloud_double(
    [surface_query_positions_plot, surface_query_positions_plot],
    color=[target_surface_query_pressure.cpu().clamp(-2, 2),
           prediction["surface_query_pressure"].cpu().clamp(-2, 2)],
    delta_clamp=(-0.25, 0.25),
    title=["target pressure", "predicted pressure"],
    num_points=2000,
    figsize=(18, 6),
    save_path=save_path,
)

volume_query_positions_plot = batch["volume_query_position"].cpu().squeeze(0)
volume_query_positions_plot = volume_query_positions_plot.clamp(
    torch.tensor([325, 308, 320]),
    torch.tensor([366, 358, 350]),
)
plot_pointcloud_double(
    [volume_query_positions_plot, volume_query_positions_plot],
    color=[target_volume_query_velocity.cpu()[:, 0].clamp(-2, 2),
           prediction["volume_query_velocity"].cpu()[:, 0].clamp(-2, 2)],
    delta_clamp=(-0.25, 0.25),
    title=["target velocity", "predicted velocity"],
    num_points=2000,
    figsize=(18, 6),
    save_path=save_path,
)

# ============================================================
# Compute error metrics
# ============================================================
logger.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~error metrics")
volume_query_velocity_mse = nn.functional.mse_loss(
#    prediction["surface_query_pressure"], target_surface_query_pressure
    prediction["volume_query_velocity"], target_volume_query_velocity
)
logger.info(f"MSE (volume query pressure): {volume_query_velocity_mse}")

volume_normalizer = collator.get_preprocessor(
    lambda c: isinstance(c, MomentNormalizationPreprocessor)
    and c.items == {"volume_velocity"}
)
target_volume_query_velocity_denorm = volume_normalizer.denormalize(target_volume_query_velocity)
pres_volume_query_velocity_denorm = volume_normalizer.denormalize(prediction["volume_query_velocity"])

delta = target_volume_query_velocity_denorm - pres_volume_query_velocity_denorm
l2_error = delta.norm() / target_volume_query_velocity_denorm.norm()
logger.info(f"L2 error (denormalized volume velocity): {l2_error}")

