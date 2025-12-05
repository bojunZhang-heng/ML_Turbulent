import os
import yaml
import torch
import torch.nn.functional as F
import logging

from preprocessors import MomentNormalizationPreprocessor
from create_data_loaders import create_data_loaders
from model_ab_ubt import AnchoredBranchedUPT
from tqdm import tqdm
from types import SimpleNamespace

# ============================================================
# Load hyperparam
# ============================================================
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

args = load_config("config.yml")
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
train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
    root, batch_size, use_query_positions=True, num_workers=1
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

weights = compute_weights(target_keys, enabled_target_keys)

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
    "volume_anchor_velocity":  get_norm(test_dataloader, {"volume_velocity"}),
    "surface_anchor_wallshearstress": get_norm(test_dataloader, {"surface_wallshearstress"}),
    "volume_anchor_totalpcoeff": get_norm(test_dataloader, {"volume_totalpcoeff"}),
}


# ============================================================
# Load checkpoint
# ============================================================
local_dir = "./experiments/epoch300_anchor_pressureANDvelocity"
os.makedirs(local_dir, exist_ok=True)

# ============================================================
# Model setup
# ============================================================
abupt = AnchoredBranchedUPT(args).to("cpu").eval()
checkpoint = torch.load("./experiments/epoch300_anchor_pressureANDvelocity/best_model.pth", map_location="cpu", weights_only=True)

# 去除module.前缀
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 去掉 'module.'
    else:
        new_state_dict[k] = v

abupt.load_state_dict(new_state_dict)

# ============================================================
# Move batch to GPU
# ============================================================
L2_errors = {k: [] for k in normalizers.keys()}
mse_sums = {k: [] for k in normalizers.keys()}
mse_counts = {k: 0 for k in normalizers.keys()}
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="[Testing]"):
        batch = {key: value.to("cpu") for key, value in batch.items()}
        targets = {k: batch.pop(k) for k in target_keys if k in batch}
        batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}

        prediction = abupt(**batch_filtered)

        # denormalize
        for key in enabled_target_keys:
            pred_den = normalizers[key].denormalize(prediction[key])
            targ_den = normalizers[key].denormalize(targets[key])

            # MAE loss
            mse_loss = F.mse_loss(prediction[key], targets[key], reduction='mean')
            mse_sums[key].append(mse_loss.item())
            mse_counts[key] += 1

            # L2 relative error
            L2_error = (pred_den - targ_den).norm() / targ_den.norm()
            L2_errors[key].append(L2_error)

    avg_L2 = {k: sum(v)/len(v) for k, v in L2_errors.items() if k in enabled_target_keys}
    avg_mse = {k: sum(mse_sums[k]) / len(mse_sums[k]) for k in enabled_target_keys}

    logging.info("********************avg_L2:")
    for key, val in avg_L2.items():
        logging.info(f"{key}: {val:.6f}")

    logging.info("********************avg_mse:")
    for key, val in avg_mse.items():
        logging.info(f"{key}: {val:.6f}")




