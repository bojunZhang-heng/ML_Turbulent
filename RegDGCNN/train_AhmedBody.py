# train.py
import warnings
import os
import yaml
import argparse
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
from modules_RT.model.model_transolver import Model
from torch.utils.data import DataLoader

# from model_tmp import AnchoredBranchedUPT
from preprocessors_AhmedBody.Dataset import VTKDataset, SATO_Dataset, sato_collate_fn
warnings.filterwarnings("ignore", category=UserWarning)

# ! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL



def initialize_model(args, device):
    """Initialize and return the RegDGCN model."""

    model = Model(hidden_dim=args.model.hidden_dim,
                  layer_num=args.model.layer_num,
                  space_dim=args.model.input_dim,
                  mlp_ratio=args.model.mlp_ratio,
                  slice_num=args.model.slice_num,
                  out_dim=args.model.output_dim,
                  dropout=args.model.dropout,
            ).to(device)

    return model

def print_memory_stats(device=None, message=""):
    """
    打印当前和峰值GPU显存使用统计
    """
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


def train_and_evaluate(args, device):
    """main function for Distributed training and evaluation."""
    setup_seed(args.training.seed)

    exp_dir = os.path.join("experiments_AhmedBody", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "training.log")
    setup_logger(log_file)
    logging.info("Config:\n" + yaml.dump(namespace_to_dict(args), sort_keys=False))
    logging.info(f"args.exp_name : {args.exp_name}")
    logging.info("Starting training with 1 GPUs")

    # Initialize model
    model = initialize_model(args, device)

    # get data
    Dataset = VTKDataset()

    train_data_lst, test_data_lst, val_data_lst = Dataset.get_data_dict(args.dataload.directory)

    # Create DataLoaders
    train_dataset = SATO_Dataset(train_data_lst, args, is_train=True)
    val_dataset = SATO_Dataset(val_data_lst, args, is_train=True)
    test_dataset = SATO_Dataset(test_data_lst, args, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sato_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sato_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=sato_collate_fn)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    # Log dataset info
    logging.info(
        f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches"
    )

    # Set up criterion, optimizer, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.training.lr, weight_decay=args.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.training.scheduler_step,
        gamma=args.training.scheduler_gamma
    )

    # Store the model
    best_model_path = os.path.join("experiments_AhmedBody", args.exp_name, "best_model.pth")
    final_model_path = os.path.join("experiments_AhmedBody", args.exp_name, "final_model.pth")

    # Check if test_only and model exists
    if args.training.test_only and os.path.exists(best_model_path):
        logging.info("Loading best model for testing only")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_model(model, test_dataloader, criterion, device, os.path.join('experiments_AhmedBody', args.exp_name), args)
        return

    # Training tracking
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    logging.info(f"Staring training for {args.training.epochs} epochs")

    # Training loop
    for epoch in range(args.training.epochs):
        # Set epoch for the DistributedSampler

        # Training
        torch.cuda.empty_cache()
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, args)
        torch.cuda.empty_cache()

        # Validation
        torch.cuda.empty_cache()
        val_loss = validate(model, val_dataloader, criterion, device, args)
        torch.cuda.empty_cache()

        # Record losses.
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(
            f"Epoch {epoch + 1}/{args.training.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{RESET}"
        )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")

        # Update learning rate scheduler
        scheduler.step()

        # Save progress rate scheduler
        if (epoch + 1) % 10 == 0 or epoch == args.training.epochs - 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
            plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training Progress - AB-UBT")
            plt.savefig(
                os.path.join("experiments_AhmedBody", args.exp_name, "training_progress.png")
            )
            plt.close()

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Test the final model
    logging.info("Testing the final model")
    test_model(
        model,
        test_dataloader,
        criterion,
        device,
        os.path.join("experiments_AhmedBody", args.exp_name),
        args,
    )

    # Test the best model
    logging.info("Testing the best model")
    model.load_state_dict(
            torch.load(best_model_path, map_location=device)
    )
    test_model(
        model,
        test_dataloader,
        criterion,
        device,
        os.path.join("experiments_AhmedBody", args.exp_name),
        args,
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
    "volume_anchor_velocity",      # torch.Size([16384, 3])
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


def train_one_epoch(model, train_dataloader, optimizer, criterion, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data in tqdm(train_dataloader, desc="[Training]"):
        x = data[args.training.input].to(device)
        y = data[args.training.target][:,:,0:1].to(device)
    #    X_MEAN = torch.tensor(args.training.x_mean, dtype=x.dtype, device=x.device)
    #    X_STD  = torch.tensor(args.training.x_std, dtype=x.dtype, device=x.device)
        Y_MEAN = torch.tensor(args.training.y_mean, dtype=x.dtype, device=x.device)
        Y_STD = torch.tensor(args.training.y_std, dtype=x.dtype, device=x.device)
    #    x = (x - X_MEAN) / X_STD
        y = (y - Y_MEAN) / Y_STD

        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def validate(model, val_dataloader, criterion, device, args):
    """Validate the model"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(val_dataloader, desc="[Validation]"):
            x = data[args.training.input].to(device)
            y = data[args.training.target][:,:,0:1].to(device)

    #        X_MEAN = torch.tensor(args.training.x_mean, dtype=x.dtype, device=x.device)
    #        X_STD  = torch.tensor(args.training.x_std, dtype=x.dtype, device=x.device)
            Y_MEAN = torch.tensor(args.training.y_mean, dtype=x.dtype, device=x.device)
            Y_STD = torch.tensor(args.training.y_std, dtype=x.dtype, device=x.device)
    #        x = (x - X_MEAN) / X_STD
            y = (y - Y_MEAN) / Y_STD
            y_hat = model(x)
            #y_hat = y_hat * PRESSURE_STD + PRESSURE_MEAN
            loss = criterion(y_hat, y)

            total_loss += loss.item()

    return total_loss / len(val_dataloader)


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


def test_model(model, test_dataloader, criterion, device, exp_dir, args):
    """Test the model, take postprocess and calculate metrics."""
    model.eval()
    total_inference_time = 0

    total_loss = 0
    total_L2_error = 0

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="[test_dataloader]"):
            x = data[args.training.input].to(device)
            y = data[args.training.target][:,:,0:1].to(device)

    #        X_MEAN = torch.tensor(args.training.x_mean, dtype=x.dtype, device=x.device)
    #        X_STD  = torch.tensor(args.training.x_std, dtype=x.dtype, device=x.device)
            Y_MEAN = torch.tensor(args.training.y_mean, dtype=x.dtype, device=x.device)
            Y_STD = torch.tensor(args.training.y_std, dtype=x.dtype, device=x.device)

    #        x = (x - X_MEAN) / X_STD
            y = (y - Y_MEAN) / Y_STD
            y_hat = model(x)

            loss = criterion(y_hat, y)
            total_loss += loss.item()

            y = y * Y_STD + Y_MEAN
            y_hat = y_hat * Y_STD + Y_MEAN
            rel_l2 = (y_hat - y).norm() / y.norm()
            total_L2_error += rel_l2.item()

        logging.info(f"*******************{M}L2_erro:{RESET}")
        logging.info(f" {total_L2_error / len(test_dataloader):.6f}")

        logging.info(f"*******************{M}mse_loss:{RESET}")
        logging.info(f" {total_loss / len(test_dataloader):.6f}")

        logging.info(f"*******************{M}inference_time:{RESET}")
        logging.info(f" {total_inference_time / len(test_dataloader):.6f}")

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

def namespace_to_dict(ns):
    return {
        k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
        for k, v in vars(ns).items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to config yaml file (e.g. config_train_velocity.yml)"
    )
    return parser.parse_args()

def setup_device():
    if torch.cuda.is_available():
        # 总是使用 cuda:0，因为 PyTorch 会对可见 GPU 重新编号
        device = torch.device("cuda:0")

        # 打印调试信息
        print(f"CUDA available GPUs: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        # 如果设置了环境变量，显示物理 GPU 编号
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            visible_gpus = [x.strip() for x in cuda_visible.split(',') if x.strip()]
            print(f"Physical GPU(s) specified in CUDA_VISIBLE_DEVICES: {visible_gpus}")
            if visible_gpus:
                print(f"Using physical GPU {visible_gpus[0]} (visible as cuda:0 in PyTorch)")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    return device

def main():
    """main function to parse arguments and start training."""

    args_cmd = parse_args()
    args = load_config(args_cmd.config)
    args.dataload.directory = os.path.expanduser(args.dataload.directory)

    exp_dir = os.path.join("experiments_AhmedBody", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    device = setup_device()
    train_and_evaluate(args, device)


if __name__ == "__main__":
    main()

