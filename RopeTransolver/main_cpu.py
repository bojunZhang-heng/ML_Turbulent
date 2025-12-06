import os
import yaml
import warnings
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np

from preprocessors import (
    MomentNormalizationPreprocessor,
)
from utils_ab_ubt import plot_pointcloud_double
from create_data_loaders import create_data_loaders
from model_transolver import Model
from tqdm import tqdm
from types import SimpleNamespace
from colorama import Fore, Style
warnings.filterwarnings("ignore", category=UserWarning)

# ! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL

# ============================================================
# Load hyperparam

# ============================================================
# Load hyperparam
# ============================================================
def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return dict_to_namespace(cfg)  # ← 必须用这个

args = load_config("config_train_s_wss.yml")

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
root_shuguang="~/drivaerml_N_30_000"
root_qiming="~/drivaerML_N100_000"
batch_size = 1
train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
    root_qiming, batch_size, use_query_positions=True, num_workers=1,
    train_split="train_cpu", val_split="val_cpu", test_split="test_cpu"
)
logging.info(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches"
)

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

# ================================
# Enable setting
# ================================
enabled_target_keys = [
    "volume_anchor_velocity",
    "surface_anchor_pressure",
    "surface_anchor_wallshearstress",
#    "volume_query_velocity",
#    "surface_query_pressure",
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
        return (
            isinstance(c, MomentNormalizationPreprocessor)
            and c.items == self.target_items
        )


def get_norm(dataloader, items):
    selector = PreprocessorSelector(items)
    return try_get_normalizer_from_collator(dataloader, selector)

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

weights = compute_weights(target_keys, enabled_target_keys)

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

## ============================================================
## Model train
## ============================================================
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Model(n_hidden=args.model.hidden,
#              n_layers=args.model.layers,
#              space_dim=args.model.input_dim,
#              mlp_ratio=args.model.mlp_ratio,
#              slice_num=args.model.slice_num,
#              out_dim=args.model.output_dim,
#              ).to(device)
#
## Set up criterion, optimizer, and scheduler
criterion = torch.nn.MSELoss()
#optimizer = optim.AdamW(
#    model.parameters(), lr=args.training.lr, weight_decay=args.training.weight_decay
#)
#scheduler = optim.lr_scheduler.OneCycleLR(
#    optimizer,
#    max_lr=args.training.lr,
#    epochs=args.training.epochs,
#    steps_per_epoch=len(train_dataloader),
#)
#scheduler = torch.optim.lr_scheduler.StepLR(
#    optimizer,
#    step_size=args.training.scheduler_step,
#    gamma=args.training.scheduler_gamma
#)
#
#model.train()
#
#for batch in tqdm(train_dataloader, desc="[Training]"):
#    batch = {key: value.to(device) for key, value in batch.items()}
#
#    # extract target variables for anchor and query
#    targets = {k: batch.pop(k) for k in target_keys if k in batch}
#
#    # extract target variables for anchor and query
#    batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
#    data_volume = batch_filtered["volume_anchor_position"]
#
#    optimizer.zero_grad()
#    pred_velocity = model(data_volume)

# ============================================================
# Model setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(n_hidden=args.model.hidden,
              n_layers=args.model.layers,
              space_dim=args.model.input_dim,
              mlp_ratio=args.model.mlp_ratio,
              slice_num=args.model.slice_num,
              out_dim=args.model.output_dim,
        ).to(device).eval()
cwd = os.getcwd()
exp_dir = os.path.join(cwd, "experiments")
model_dir = os.path.join(exp_dir, args.exp_name)
model_path = os.path.join(model_dir, "best_model.pth")
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

# 去除module.前缀
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 去掉 'module.'
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)


# ============================================================
# Model test
# ============================================================
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="[Testing]"):
        batch = {key: value.to(device) for key, value in batch.items()}
        targets = {k: batch.pop(k) for k in target_keys if k in batch}
        targets_velocity = targets["surface_anchor_wallshearstress"]

        batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
        data_volume = batch_filtered["surface_anchor_position"]

        pred_velocity = model(data_volume)

        mse_loss = criterion(pred_velocity, targets_velocity)
        pred_den = normalizers["surface_anchor_wallshearstress"].denormalize(pred_velocity)
        targ_den = normalizers["surface_anchor_wallshearstress"].denormalize(targets_velocity)
        L2_error = (pred_den - targ_den).norm() / targ_den.norm()
        logging.info(f"*******************{M}L2_error:{RESET}")
        logging.info(f" {L2_error:.6f}")

        logging.info(f"*******************{M}mse_loss:{RESET}")
        logging.info(f" {mse_loss:.6f}")


# ============================================================
# Visualizations
# ============================================================
figure_dir = os.path.join(model_dir, "figure")
os.makedirs(figure_dir, exist_ok=True)

## Anchor pressure
#surface_anchor_positions_plot = batch["surface_anchor_position"].squeeze(0)
#anchor_pressure = os.path.join(figure_dir, "anchor_pressure.png")
#
#plot_pointcloud_double(
#    [surface_anchor_positions_plot, surface_anchor_positions_plot],
#    color=[targets["surface_anchor_pressure"].cpu().clamp(-2, 2), pred_velocity.cpu().clamp(-2, 2)],
#    delta_clamp=(-0.25, 0.25),
#    title=["target pressure", "predicted pressure"],
#    # increas this for more fidelity/larger plot
#    num_points=10_000,
#    figsize=(18, 6),
#    save_path=anchor_pressure,
#)

# Anchor velocity
volume_anchor_positions_plot = batch["volume_anchor_position"].squeeze(0)
# clamp positions to see car better
volume_anchor_positions_plot = volume_anchor_positions_plot.clamp(
    torch.tensor([325, 308, 320]),
    torch.tensor([366, 358, 350]),
)
anchor_velocity = os.path.join(figure_dir, "anchor_velocity.png")
plot_pointcloud_double(
    [volume_anchor_positions_plot, volume_anchor_positions_plot],
    color=[targets["volume_anchor_velocity"].cpu()[:, 0].clamp(-2, 2), pred_velocity.cpu()[:, 0].clamp(-2, 2)],
    delta_clamp=(-0.25, 0.25),
    title=["target velocity", "predicted velocity"],
    # increas this for more fidelity/larger plot
    num_points=10_000,
    figsize=(18, 6),
    save_path=anchor_velocity,
)

# ============================================================
# Save data
# ============================================================
data_dir = os.path.join(model_dir, "data")
os.makedirs(data_dir, exist_ok=True)
#surface_anchor_position_path = os.path.join(data_dir, "surface_anchor_position.npy")
#anchor_pressure_path = os.path.join(data_dir, "anchor_pressure.npy")
#np.save(surface_anchor_position_path, surface_anchor_positions_plot)
#np.save(anchor_pressure_path, pred_velocity)

volume_anchor_position_path = os.path.join(data_dir, "volume_anchor_position.npy")
anchor_velocity_path = os.path.join(data_dir, "anchor_velocity.npy")
np.save(volume_anchor_position_path, volume_anchor_positions_plot)
np.save(anchor_velocity_path, pred_velocity)

