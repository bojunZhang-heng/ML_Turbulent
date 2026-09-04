#!/usr/bin/env python3
import os
import yaml
import warnings
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np

from utils_v1 import setup_logger, setup_seed
from colorama import Fore, Style
from modules_RT.model.model_transolver import Model

# from model_tmp import AnchoredBranchedUPT
from preprocessors_DrivAerML import (
    MomentNormalizationPreprocessor,
)
from preprocessors_DrivAerML.create_data_loaders import create_data_loaders
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
from types import SimpleNamespace
from colorama import Fore, Style
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

# colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL

# ============================================================
# Config helpers
# ============================================================
def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return dict_to_namespace(cfg)


# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================
# Problem-specific keys and helpers (kept from your original)
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

# change these to choose which targets/positions are enabled
enabled_target_keys = [
    "volume_anchor_velocity",
 #   "surface_anchor_pressure",
  #  "surface_anchor_wallshearstress",
]

enabled_position_keys = [
    "geometry_position",
    "geometry_batch_idx",
    "geometry_supernode_idx",
    "surface_anchor_position",
    "volume_anchor_position",
]

def try_get_normalizer_from_collator(dataloader, predicate):
    """Try to get preprocessor/normalizer from dataloader.collate_fn."""
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
        return (
            isinstance(c, MomentNormalizationPreprocessor)
            and c.items == self.target_items
        )

def get_norm(dataloader, items):
    selector = PreprocessorSelector(items)
    return try_get_normalizer_from_collator(dataloader, selector)

def compute_weights(target_keys, enabled_target_keys):
    weights = {k: 0.0 for k in target_keys}
    n = len(enabled_target_keys)
    if n == 0:
        raise ValueError("enabled_target_keys 不能为空，否则无法计算 loss 权重。")
    w = 1.0 / n
    for k in enabled_target_keys:
        if k not in weights:
            raise KeyError(f"{k} 不在 batch_keys 中！")
        weights[k] = w
    return weights

# ============================================================
# Training / testing routines
# ============================================================
def run_train(cfg):
    """Train loop."""
    # --- data loaders ---
    # you can adjust roots / batch sizes inside your config if desired
    root_dir = getattr(cfg, "root_dir", "~/DrivAerML_dataset")
    batch_size = getattr(cfg.training, "batch_size", 1) if hasattr(cfg, "training") else 1

    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
        root_dir, batch_size, use_query_positions=True, num_workers=1,
        train_split="train_cpu", val_split="val_cpu", test_split="test_cpu"
    )
    logging.info(
        f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches"
    )

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- model ---
    model = Model(
        hidden_dim=cfg.model.hidden_dim,
        layer_num=cfg.model.layer_num,
        space_dim=cfg.model.input_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        slice_num=cfg.model.slice_num,
        out_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout
    ).to(device)

    # --- training loop ---
    model.train()
    epochs = 3
    logging.info(f"Start training for {epochs} epochs")
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader, desc=f"[Training epoch {epoch+1}/{epochs}]")
        for batch in pbar:
            # prepare inputs from enabled_position_keys
            batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
            data_volume = batch_filtered["volume_anchor_position"]
            pred = model(data_volume)
            print(f"pred.shape: {pred.shape}")
    logging.info("Training finished.")


def run_test(cfg):
    """Test loop (loads best_model.pth from experiments/<exp_name>/best_model.pth)."""
    # --- data loaders ---
    root_dir = getattr(cfg, "root_dir", "~/DrivAerML_dataset")
    batch_size = getattr(cfg.testing, "batch_size", 1) if hasattr(cfg, "testing") else getattr(cfg.training, "batch_size", 1)

    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
        root_dir, batch_size, use_query_positions=True, num_workers=1,
        train_split="train_cpu", val_split="val_cpu", test_split="test_cpu"
    )
    logging.info(
        f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches"
    )

    # normalizers
    normalizers = {
        "surface_anchor_pressure": get_norm(test_dataloader, {"surface_pressure"}),
        "volume_anchor_velocity": get_norm(test_dataloader, {"volume_velocity"}),
        "surface_anchor_wallshearstress": get_norm(test_dataloader, {"surface_wallshearstress"}),
        "volume_anchor_totalpcoeff": get_norm(test_dataloader, {"volume_totalpcoeff"}),
        "surface_query_pressure": get_norm(test_dataloader, {"surface_pressure"}),
        "volume_query_velocity": get_norm(test_dataloader, {"volume_velocity"}),
        "surface_query_wallshearstress": get_norm(test_dataloader, {"surface_wallshearstress"}),
        "volume_query_totalpcoeff": get_norm(test_dataloader, {"volume_totalpcoeff"}),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = Model(
        hidden_dim=cfg.model.hidden_dim,
        layer_num=cfg.model.layer_num,
        space_dim=cfg.model.input_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        slice_num=cfg.model.slice_num,
        out_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
    ).to(device).eval()

    cwd = os.getcwd()
    exp_dir = os.path.join(cwd, "experiments")
    model_dir = os.path.join(exp_dir, cfg.exp_name)
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # remove module. prefix if present
    new_state_dict = {}
    # if checkpoint is a dict with 'state_dict' key (common), adapt
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    for k, v in state.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    criterion = torch.nn.MSELoss()

    # testing
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="[Testing]"):
            batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value) or isinstance(value, torch.Tensor) or True}
            targets = {k: batch.pop(k) for k in target_keys if k in batch}

            # choose input (prefer surface_anchor_position if you used that in test)
            batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
            data_volume = batch_filtered["volume_anchor_position"]

            pred = model(data_volume)

            # pick target for metric (example uses surface_anchor_wallshearstress if available)
            targets_velocity = targets["volume_anchor_velocity"]

            mse_loss = criterion(pred, targets_velocity)
            pred_den = normalizers["surface_anchor_wallshearstress"].denormalize(pred)
            targ_den = normalizers["surface_anchor_wallshearstress"].denormalize(targets_velocity)

            # avoid zero division
            denom = targ_den.norm() if torch.norm(targ_den) != 0 else torch.tensor(1.0, device=pred_den.device)
            L2_error = (pred_den - targ_den).norm() / denom

            logging.info(f"*******************{M}L2_error:{RESET}")
            logging.info(f" {L2_error:.6f}")
            logging.info(f"*******************{M}mse_loss:{RESET}")
            logging.info(f" {mse_loss:.6f}")

    # Visualizations & saving last batch results (reuse your original plotting logic)
    # ensure 'batch' and 'pred' exist (from last loop)
    try:
        figure_dir = os.path.join(model_dir, "figure")
        os.makedirs(figure_dir, exist_ok=True)
        volume_anchor_positions_plot = batch["volume_anchor_position"].squeeze(0)
        volume_anchor_positions_plot = volume_anchor_positions_plot.clamp(
            torch.tensor([325, 308, 320], device=volume_anchor_positions_plot.device),
            torch.tensor([366, 358, 350], device=volume_anchor_positions_plot.device),
        )
        anchor_velocity = os.path.join(figure_dir, "anchor_velocity.png")
        plot_pointcloud_double(
            [volume_anchor_positions_plot, volume_anchor_positions_plot],
            color=[targets["volume_anchor_velocity"].cpu()[:, 0].clamp(-2, 2), pred.cpu()[:, 0].clamp(-2, 2)],
            delta_clamp=(-0.25, 0.25),
            title=["target velocity", "predicted velocity"],
            num_points=10_000,
            figsize=(18, 6),
            save_path=anchor_velocity,
        )
        data_dir = os.path.join(model_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "volume_anchor_position.npy"), volume_anchor_positions_plot.cpu().numpy())
        np.save(os.path.join(data_dir, "anchor_velocity.npy"), pred.cpu().numpy())
        logging.info(f"Saved visualization and data to {figure_dir} and {data_dir}")
    except Exception as e:
        logging.warning(f"Visualization/save skipped due to error: {e}")

# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run training or testing.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    args = parser.parse_args()

    # load config (hardcoded path as before)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_DrivAerML_local.yml")
    cfg = load_config(CONFIG_PATH)

    def namespace_to_dict(ns):
        return {
            k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
            for k, v in vars(ns).items()
        }

    logging.info("Config:\n" + yaml.dump(namespace_to_dict(cfg), sort_keys=False))

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "test":
        run_test(cfg)
    else:
        raise ValueError("Unknown mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()

