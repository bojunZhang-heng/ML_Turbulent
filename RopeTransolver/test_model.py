import os
import torch
import time
import numpy as np
import torch.nn as nn
import logging
from create_data_loaders import create_data_loaders
from preprocessors import (
    MomentNormalizationPreprocessor,
    PositionNormalizationPreprocessor,
)
import torch.nn.functional as F
from abupt_collator import AbuptCollator
from drivaerml_dataset import DrivAerMLDataset
from model import AnchoredBranchedUPT
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
batch_size = 1

train_dataloader, _, test_dataloader = create_data_loaders(
    root, batch_size, use_query_positions=False, num_workers=1
)

# ============================================================
# Load checkpoint
# ============================================================
local_dir = "./checkpoints"
os.makedirs(local_dir, exist_ok=True)

# ============================================================
# Model setup
# ============================================================
abupt = AnchoredBranchedUPT().to("cpu").eval()
checkpoint = torch.load("./checkpoints/ab-upt-drivaerml-tutorial.th", map_location="cpu", weights_only=True)
abupt.load_state_dict(checkpoint["state_dict"])

# ============================================================
# Train
# ============================================================
target_keys = [
    "surface_anchor_pressure",
    "surface_anchor_wallshearstress",
    "volume_anchor_totalpcoeff",
    "volume_anchor_velocity",
    "surface_query_pressure",
    "surface_query_wallshearstress",
    "volume_query_totalpcoeff",
    "volume_query_velocity",
]

enabled_target_keys = [
    "volume_anchor_velocity",
    "surface_anchor_pressure",
]

enabled_position_keys = [
    "geometry_position",
    "geometry_batch_idx",
    "geometry_supernode_idx",
    "surface_anchor_position",
    "volume_anchor_position",
#    "surface_query_position",
#    "volume_query_position",
]

def compute_weights(target_keys, enabled_target_keys):
    weights = {k: 0.0 for k in target_keys}

    # 有效数量
    n = len(enabled_target_keys)
    if n == 0:
        raise ValueError("enabled_target_keys 不能为空，否则无法计算 loss 权重。")

    # 每个激活的 key 分配 1/n
    w = 1.0 / n
    for k in enabled_target_keys:
        if k not in weights:
            raise KeyError(f"{k} 不在 batch_keys 中！")
        weights[k] = w

    return weights


criterion = nn.MSELoss()
mse_loss = {k: [] for k in enabled_target_keys}
weights = compute_weights(target_keys, enabled_target_keys)
logging.info("******************** weights:")
for key, val in weights.items():
    logging.info(f"{key}: {val}")
total_loss = 0.0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="[Testing]"):
        logging.info("********************batch:")
        for key, val in batch.items():
            logging.info(f"{key}: {val.shape}")
        start_time = time.time()
        batch = {key: value.to("cpu") for key, value in batch.items()}
        # extract target variables
        targets = {k: batch.pop(k) for k in target_keys if k in batch}


        logging.info("********************targets:")
        for key, val in targets.items():
            logging.info(f"{key}: {val.shape}")

        # filter for enabled_keys
        batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
        logging.info("********************batch_filtered:")
        for key, val in batch_filtered.items():
            logging.info(f"{key}: {val.shape}")

        # extract target variables for queries
        #target_volume_query_velocity = batch.pop("volume_query_velocity")

        prediction = abupt(**batch)

        loss_dict = {}
        for k in enabled_target_keys:
            loss_k = criterion(prediction[k], targets[k])
            loss_dict[k] = loss_k
            mse_loss[k].append(loss_k.item())

        total_loss = sum(weights[k] * loss_dict[k] for k in enabled_target_keys)
        logging.info(f"total_loss: {total_loss}")
        logging.info("********************MSE_loss:")
        for key, val in mse_loss.items():
            logging.info(f"{key}: {val}")


# ================================
# Normalizers
# ================================
def try_get_normalizer_from_collator(dataloader, predicate):
    """尝试从 dataloader.collate_fn（即 collator）获取 preprocessor/normalizer"""
    coll = getattr(dataloader, "collate_fn", None)
    if coll is None:
        return RuntimeError("No collate_fn")
    get_pre = getattr(coll, "get_preprocessor", None)
    if get_pre is None:
        return RuntimeError("No get_preprocessor")
    return get_pre(predicate)

class PreprocessorSelector:
    def __init__(self, target_items):
        self.target_items = target_items

    def __call__(self, c):
        return isinstance(c, MomentNormalizationPreprocessor) and c.items == self.target_items

def get_norm(dataloader, items):
    selector = PreprocessorSelector(items)
    return try_get_normalizer_from_collator(dataloader, selector)

normalizers = {
    "surface_anchor_pressure": get_norm(test_dataloader, {"surface_pressure"}),
    "volume_anchor_velocity": get_norm(test_dataloader, {"volume_velocity"}),
    "surface_anchor_wallshearstress": get_norm(test_dataloader, {"surface_wallshearstress"}),
    "volume_anchor_totalpcoeff": get_norm(test_dataloader, {"volume_totalpcoeff"}),
}
L2_errors = {k: [] for k in normalizers.keys()}
mse_sums = {k: [] for k in normalizers.keys()}

# ============================================================
# Test
# ============================================================
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="[Testing]"):
        for key, val in batch.items():
            logging.info(f"{key}: {val.shape}")
        start_time = time.time()
        batch = {key: value.to("cpu") for key, value in batch.items()}
        # extract target variables for anchor
        targets = {k: batch.pop(k) for k in normalizers.keys()}

        # extract target variables for queries
        target_surface_query_pressure = batch.pop("surface_query_pressure")
        target_surface_query_wallshearstress = batch.pop(
            "surface_query_wallshearstress"
        )
        target_volume_query_totalpcoeff = batch.pop("volume_query_totalpcoeff")
        target_volume_query_velocity = batch.pop("volume_query_velocity")

        prediction = abupt(**batch)

        # denormalize
        for key in normalizers.keys():
            pred_den = normalizers[key].denormalize(prediction[key])
            targ_den = normalizers[key].denormalize(targets[key])

            # MAE loss

            # L2 relative error
            L2_error = (pred_den - targ_den).norm() / targ_den.norm()
            L2_errors[key].append(L2_error.item())

        avg_L2 = {k: sum(v)/len(v) for k, v in L2_errors.items()}
        logging.info("********************avg_L2:")
        for key, val in avg_L2.items():
            logging.info(f"{key}: {val:.6f}")

