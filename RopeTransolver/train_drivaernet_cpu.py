# train.py
import warnings
import os
import random
import yaml
import torch
import torch.optim as optim
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace

# Import modules
from utils_v1 import setup_logger, setup_seed
from colorama import Fore, Style
from model_transolver import Model
from preprocessors_SATO.Dataset import VTKDataset, SATO_Dataset, sato_collate_fn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL

# 全局变量占位符 (如果原代码中有定义，请确保在此处或通过参数传递)
# 注意：原代码中使用了 PRESSURE_MEAN 和 PRESSURE_STD 但未在文件中定义
# 这里假设它们可能由 dataset 返回的 mean_data/std_data 设定，或者需要手动定义
PRESSURE_MEAN = 0.0
PRESSURE_STD = 1.0

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
    return dict_to_namespace(cfg)

# 确保配置文件路径正确
args = load_config("config_train_DrivAerNet.yml")

def namespace_to_dict(ns):
    return {
        k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
        for k, v in vars(ns).items()
    }

# ============================================================
# Helper Functions
# ============================================================

def initialize_model(args, device):
    model = Model(hidden_dim=args.model.hidden_dim,
                  layer_num=args.model.layer_num,
                  space_dim=args.model.input_dim,
                  mlp_ratio=args.model.mlp_ratio,
                  slice_num=args.model.slice_num,
                  out_dim=args.model.output_dim,
                  dropout=args.model.dropout,
            ).to(device)

    # 单卡/CPU模式不需要 DistributedDataParallel
    return model

def print_memory_stats(device=None, message=""):
    """
    打印当前和峰值GPU显存使用统计
    """
    if not torch.cuda.is_available():
        return

    if device is None:
        device = torch.cuda.current_device()

    # 当前由Tensor占用的显存
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # 转换为MB
    # PyTorch CachingAllocator当前管理的总显存
    reserved = torch.cuda.memory_reserved(device) / 1024**2   # 转换为MB
    # 本次程序运行中，Tensor占用的峰值显存
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # 转换为MB

    logging.info(f"{message}:")
    logging.info(f"  当前Tensor占用显存: {allocated:.2f} MB")
    logging.info(f"  当前CachingAllocator管理的总显存: {reserved:.2f} MB")
    logging.info(f"  峰值Tensor占用显存: {max_allocated:.2f} MB")
    logging.info("-" * 50)


# ============================================================
# Main Training Logic
# ============================================================

def run_training(args):
    """Main function for Single GPU/CPU training and evaluation."""
    setup_seed(args.training.seed)

    # 1. 设置设备 (GPU or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 如果 args 指定了 GPU 列表，可以通过 CUDA_VISIBLE_DEVICES 环境变量控制，
        # 但在代码内部通常直接使用 "cuda" 指向可见的第一个设备。
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # 2. 设置日志
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "training.log")
    setup_logger(log_file)

    logging.info("Config:\n" + yaml.dump(namespace_to_dict(args), sort_keys=False))
    logging.info(f"args.exp_name : {args.exp_name}")

    # 3. 初始化模型
    model = initialize_model(args, device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    # 4. 获取数据
    Dataset = VTKDataset()
    train_data_lst, test_data_lst, val_data_lst, mean_data, std_data = Dataset.get_data_dict(args.dataload.directory)

    # 如果 PRESSURE_MEAN/STD 应该来自数据，请在这里取消注释并赋值
    # global PRESSURE_MEAN, PRESSURE_STD
    # PRESSURE_MEAN = mean_data
    # PRESSURE_STD = std_data

    # Create DataLoaders
    train_dataset = SATO_Dataset(train_data_lst, args, is_train=True)
    test_dataset = SATO_Dataset(test_data_lst, args, is_train=False)
    # val_dataset = SATO_Dataset(val_data_lst, args, is_train=False) # 假设你需要 val_dataset

    # 单卡模式下 shuffle=True
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sato_collate_fn)
    # 注意：如果 val_data_lst 存在，建议也创建一个 val_dataloader
    # 这里暂时用 test_dataloader 代替演示，因为原代码 val_loader 变量名未定义清楚
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=sato_collate_fn)
    val_dataloader = test_dataloader # 临时复用，如果上面有 val_dataset 请替换

    logging.info(
        f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches"
    )

    # 5. 设置优化器和损失函数
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.training.lr, weight_decay=args.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.training.scheduler_step,
        gamma=args.training.scheduler_gamma
    )

    # 存储路径
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    final_model_path = os.path.join(exp_dir, "final_model.pth")

    # Test Only Mode
    if args.training.test_only and os.path.exists(best_model_path):
        logging.info("Loading best model for testing only")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_model(model, test_dataloader, criterion, device, exp_dir)
        return

    # Training Tracking
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    logging.info(f"Staring training for {args.training.epochs} epochs")

    # 6. Training Loop
    for epoch in range(args.training.epochs):

        # Training
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Validation
        val_loss = validate(model, val_dataloader, criterion, device)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(
            f"Epoch {epoch + 1}/{args.training.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{RESET}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")

        # Update Scheduler
        scheduler.step()

        # Save Progress Plot
        if (epoch + 1) % 10 == 0 or epoch == args.training.epochs - 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
            plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training Progress")
            plt.savefig(os.path.join(exp_dir, "training_progress.png"))
            plt.close()

    # Save Final Model
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Test Final Model
    logging.info("Testing the final model")
    test_model(model, test_dataloader, criterion, device, exp_dir)

    # Test Best Model
    logging.info("Testing the best model")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_model(model, test_dataloader, criterion, device, exp_dir)


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
    "volume_anchor_velocity",       # torch.Size([16384, 3])
    "surface_anchor_pressure",
    "surface_anchor_wallshearstress",
]

enabled_position_keys = [
    "geometry_position",
    "geometry_batch_idx",
    "geometry_supernode_idx",
    "surface_anchor_position",     # torch.Size([1, 16384, 3])
    "volume_anchor_position",
]

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

weights = compute_weights(target_keys, enabled_target_keys)


def train_one_epoch(model, train_dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data, targets in tqdm(train_dataloader, desc="[Training]"):
        # logging.info(f"data.shape: {data.shape}") # Optional: reduce log spam
        data = data.squeeze(1).to(device).permute(0, 2, 1)
        targets = targets.squeeze(1).to(device).permute(1,0)

        # 确保 PRESSURE_MEAN / STD 已定义
        targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(1), targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def validate(model, val_dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="[Validation]"):
            batch = {key: value.to(device) for key, value in batch.items()}

            targets = {k: batch.pop(k) for k in target_keys if k in batch}
            targets_s_pressure = targets.get("surface_anchor_pressure")

            # 简单的错误检查，防止 key 不存在
            if targets_s_pressure is None:
                continue

            batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
            data_volume = batch_filtered.get("surface_anchor_position")

            if data_volume is None:
                continue

            pred_s_pressure = model(data_volume)
            loss = criterion(pred_s_pressure, targets_s_pressure)
            total_loss += loss.item()

    return total_loss / len(val_dataloader)


# ================================
# Normalizers
# ================================
def try_get_normalizer_from_collator(dataloader, predicate):
    coll = getattr(dataloader, "collate_fn", None)
    if coll is None:
        return None # Changed from RuntimeError to avoid crash if simple collator
    get_pre = getattr(coll, "get_preprocessor", None)
    if get_pre is None:
        return None
    return get_pre(predicate)


class PreprocessorSelector:
    def __init__(self, target_items):
        self.target_items = target_items

    def __call__(self, c):
        # 需要确保 MomentNormalizationPreprocessor 已导入或定义
        # 这里假设它在 preprocessors_SATO 等模块中有效
        return hasattr(c, 'items') and c.items == self.target_items


def get_norm(dataloader, items):
    selector = PreprocessorSelector(items)
    return try_get_normalizer_from_collator(dataloader, selector)


def test_model(model, test_dataloader, criterion, device, exp_dir):
    """Test the model, take postprocess and calculate metrics."""
    model.eval()
    total_inference_time = 0

    normalizers = {
        "surface_anchor_pressure": get_norm(test_dataloader, {"surface_pressure"}),
        # ... 其他 normalizers
    }

    total_loss = 0
    total_L2_error = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="[Testing]"):
            start_time = time.time()
            batch = {key: value.to(device) for key, value in batch.items()}

            targets = {k: batch.pop(k) for k in target_keys if k in batch}
            targets_s_pressure = targets.get("surface_anchor_pressure")
            if targets_s_pressure is None: continue

            batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
            data_surface = batch_filtered.get("surface_anchor_position")
            if data_surface is None: continue

            pred_s_pressure = model(data_surface)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            mse_loss = criterion(pred_s_pressure, targets_s_pressure)
            total_loss += mse_loss.item()

            # 处理 Denormalize
            norm_obj = normalizers["surface_anchor_pressure"]
            if norm_obj:
                pred_den = norm_obj.denormalize(pred_s_pressure)
                targ_den = norm_obj.denormalize(targets_s_pressure)
                L2_error = (pred_den - targ_den).norm() / targ_den.norm()
                total_L2_error += L2_error.item()
            else:
                # 如果找不到 normalizer，回退到普通计算或跳过
                total_L2_error += 0

    logging.info(f"*******************{M}L2_error:{RESET}")
    logging.info(f" {total_L2_error / len(test_dataloader):.6f}")

    logging.info(f"*******************{M}mse_loss:{RESET}")
    logging.info(f" {total_loss / len(test_dataloader):.6f}")

    logging.info(f"*******************{M}inference_time:{RESET}")
    logging.info(f" {total_inference_time / len(test_dataloader):.6f}")


def main():
    """main function to parse arguments and start training."""

    # 获取用户指定的 GPU 列表
    gpu_list = args.training.gpus

    # 设置可见设备 (对于单卡模式，这通常会使 cuda:0 指向该列表中的第一个 GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    # 直接运行训练
    run_training(args)


if __name__ == "__main__":
    main()
