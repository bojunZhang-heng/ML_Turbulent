# train.py
import warnings
import os
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import time
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import modules
# from torch.utils.data.distributed import DistributedSampler
from utils_v1 import setup_logger, setup_seed
from colorama import Fore, Style
from model_ab_ubt import AnchoredBranchedUPT

# from model_tmp import AnchoredBranchedUPT
from create_data_loaders import create_data_loaders
from preprocessors import (
    MomentNormalizationPreprocessor,
)
warnings.filterwarnings("ignore", category=UserWarning)

# ! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multi physcial field prediction models on MB-Transolver"
    )

    # Basic settings
    parser.add_argument(
        "--exp_name", type=str, default="PressurePrediction", help="Experiment name"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Data settings
    parser.add_argument("--Cdataset_path", type=str, help="Path to dataset")
    parser.add_argument("--Pdataset_path", type=str, help="Path to dataset")
    parser.add_argument("--Wdataset_path", type=str, help="Path to dataset")
    parser.add_argument("--Vdataset_path", type=str, help="Path to dataset")
    parser.add_argument("--Ccache_dir", type=str, help="Path to cache directory")
    parser.add_argument("--Pcache_dir", type=str, help="Path to cache directory")
    parser.add_argument("--Wcache_dir", type=str, help="Path to cache directory")
    parser.add_argument("--Vcache_dir", type=str, help="Path to cache directory")
    parser.add_argument("--subset_dir", type=str, help="Path to train/val/test splits")
    parser.add_argument("--root_dir", type=str, help="Path to train/val/test splits")
    parser.add_argument(
        "--num_points", type=int, default=10000, help="Number of points to sample"
    )

    # Training settings
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--test_only", type=int, default=0, help="Only test the model, no training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="GPUs to use (comma-separated)"
    )

    # Model settings
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument(
        "--emb_dims", type=int, default=1024, help="Embedding dimensions"
    )
    parser.add_argument("--k", type=int, default=40, help="Number of nearest neighbors")
    parser.add_argument(
        "--output_channels", type=int, default=1, help="Number of output channels"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--n_layers", type=int, default=3, help="layers")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--slice_num", type=int, default=32)
    parser.add_argument("--ref", type=int, default=8)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--mlp_ratio", type=int, default=1)
    parser.add_argument("--ndim", type=int, default=3)
    parser.add_argument("--n_input", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim_surface", type=int, default=1)
    parser.add_argument("--output_dim_volume", type=int, default=3)
    parser.add_argument("--geometry_depth", type=int, default=1)
    parser.add_argument("--num_surf_blocks", type=int, default=6)
    parser.add_argument("--num_volume_blocks", type=int, default=6)
    parser.add_argument("--blocks", type=str, default="pscscs")
    parser.add_argument("--res", type=str, default="True")
    parser.add_argument("--dim_head", type=int, default="64")
    parser.add_argument("--radius", type=float, default="0.25")

    return parser.parse_args()


def initialize_model(args, local_rank):
    """Initialize and return the RegDGCN model."""

    model = AnchoredBranchedUPT(args).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
        output_device=local_rank,
    )

    return model


def train_and_evaluate(rank, world_size, args):
    """main function for Distributed training and evaluation."""
    setup_seed(args.seed)

    # Initialize process group for DDP
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    local_rank = rank
    torch.cuda.set_device(local_rank)

    # Set up logging (only on rank 0)
    if local_rank == 0:
        exp_dir = os.path.join("experiments", args.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, "training.log")
        setup_logger(log_file)
        logging.info(f"args.exp_name : {args.exp_name}")
        logging.info(f"Starting training with {world_size} GPUs")

    # Initialize model
    model = initialize_model(args, local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")

    # Dataload
    # BUG is here
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
        args.root_dir, args.batch_size, use_query_positions=True, num_workers=1
    )

    # Log dataset info
    if local_rank == 0:
        logging.info(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches"
        )

    # Set up criterion, optimizer, and scheduler
    #! There is a puzzle!######
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dataloader),
    )

    # Store the model
    best_model_path = os.path.join("experiments", args.exp_name, "best_model.pth")
    final_model_path = os.path.join("experiments", args.exp_name, "final_model.pth")

    # Check if test_only and model exists
    if args.test_only and os.path.exists(best_model_path):
        if local_rank == 0:
            logging.info("Loading best model for testing only")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
        logging.info(f"*******************{M} Best model: {RESET}")
        test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))
        dist.destroy_process_group()
        return

    # Training tracking
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if local_rank == 0:
        logging.info(f"Staring training for {args.epochs} epochs")

    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for the DistributedSampler
        # train_dataloader.sampler.set_epoch(epoch)

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank)

        # Validation
        val_loss = validate(model, val_dataloader, criterion, local_rank)

        # Record losses.
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logging.info(
                f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{RESET}"
            )

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Save progress rate scheduler
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
                plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.title("Training Progress - AB-UBT")
                plt.savefig(
                    os.path.join("experiments", args.exp_name, "training_progress.png")
                )
                plt.close()

    # Save final model
    if local_rank == 0:
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    # Make sure all processes sync up before testing
    dist.barrier()

    # Test the final model
    if local_rank == 0:
        logging.info("Testing the final model")
    logging.info(f"*******************{M} Best model: {RESET}")
    test_model(
        model,
        test_dataloader,
        criterion,
        local_rank,
        os.path.join("experiments", args.exp_name),
    )

    # Test the best model
    if local_rank == 0:
        logging.info("Testing the best model")
        model.load_state_dict(
            torch.load(best_model_path, map_location=f"cuda:{local_rank}")
        )
    logging.info(f"*******************{M} Final model: {RESET}")
    test_model(
        model,
        test_dataloader,
        criterion,
        local_rank,
        os.path.join("experiments", args.exp_name),
    )

    # Clean up
    dist.destroy_process_group()

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


def train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = 0
    total_samples = 0
    total_time = 0.0  # seconds

    mse_loss = {k: [] for k in enabled_target_keys}

    for batch in tqdm(train_dataloader, desc="[Training]"):
        batch = {key: value.to(local_rank) for key, value in batch.items()}

        # extract target variables for anchor and query
        targets = {k: batch.pop(k) for k in target_keys if k in batch}

        # extract target variables for anchor and query
        batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}

        batch_size = 1
        # timing start (make sure GPU kernels are finished before starting timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # sync prior async work
        t0 = time.time()

        optimizer.zero_grad()
        prediction = model(**batch_filtered)

        loss_dict = {}
        for k in enabled_target_keys:
            loss_k = criterion(prediction[k], targets[k])
            loss_dict[k] = loss_k
            mse_loss[k].append(loss_k.item())
        loss = sum(weights[k] * loss_dict[k] for k in enabled_target_keys)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # timing end (sync to ensure all GPU work finished)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        batch_time = t1 - t0
        total_time += batch_time
        total_batches += 1
        total_samples += batch_size

    avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0.0
    avg_time_per_batch = total_time / total_batches if total_batches > 0 else 0.0

    for k, v in mse_loss.items():
        key_loss = sum(v) / len(train_dataloader)
        logging.info(f"{k}_loss: {key_loss}")
    logging.info(
        f"[Timing][Epoch] total_samples={total_samples}, "
        f"total_time={total_time:.4f}s, "
        f"avg_time_per_sample={avg_time_per_sample:.6f}s, "
        f"avg_time_per_batch={avg_time_per_batch:.4f}s"
    )

    return total_loss / len(train_dataloader)


def validate(model, val_dataloader, criterion, local_rank):
    """Validate the model"""

    model.eval()
    total_loss = 0
    mse_loss = {k: [] for k in enabled_target_keys}
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="[Validation]"):
            batch = {key: value.to(local_rank) for key, value in batch.items()}

            # extract target variables for anchor and query
            targets = {k: batch.pop(k) for k in target_keys if k in batch}

            # extract target variables for anchor and query
            batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
            prediction = model(**batch_filtered)

            loss_dict = {}
            for k in enabled_target_keys:
                loss_k = criterion(prediction[k], targets[k])
                loss_dict[k] = loss_k
                mse_loss[k].append(loss_k.item())
            loss = sum(weights[k] * loss_dict[k] for k in enabled_target_keys)

            total_loss += loss.item()

        for k, v in mse_loss.items():
            key_loss = sum(v) / len(val_dataloader)
            logging.info(f"{k}_loss: {key_loss}")

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


def test_model(model, test_dataloader, criterion, local_rank, exp_dir):
    """Test the model, take postprocess and calculate metrics."""
    model.eval()
    total_inference_time = 0

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
    L2_errors = {k: [] for k in normalizers.keys()}
    mse_sums = {k: [] for k in normalizers.keys()}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="[Testing]"):
            start_time = time.time()
            batch = {key: value.to(local_rank) for key, value in batch.items()}
            # extract target variables for anchor

            targets = {k: batch.pop(k) for k in target_keys if k in batch}

            batch_filtered = {k: batch[k] for k in enabled_position_keys if k in batch}
            prediction = model(**batch_filtered)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # denormalize
            for key in normalizers.keys():
                if key in enabled_target_keys:
                    pred_den = normalizers[key].denormalize(prediction[key])
                    targ_den = normalizers[key].denormalize(targets[key])

                    # MAE loss
                    mse_loss = criterion(prediction[key], targets[key])
                    mse_sums[key].append(mse_loss.item())

                    # L2 relative error
                    L2_error = (pred_den - targ_den).norm() / targ_den.norm()
                    L2_errors[key].append(L2_error.item())

        avg_L2 = {k: sum(v) / len(test_dataloader) for k, v in L2_errors.items()}
        avg_mse = {k: sum(mse_sums[k]) / len(test_dataloader) for k in enabled_target_keys}

        logging.info(f"*******************{M}avg_L2:{RESET}")
        for key, val in avg_L2.items():
            logging.info(f"{key}: {val:.6f}")

        logging.info(f"*******************{M}avg_mse:{RESET}")
        for key, val in avg_mse.items():
            logging.info(f"{key}: {val:.6f}")

    # Checkout the value


#    if dist.get_rank() == 0:
#        logging.info(f"Total MSE across all processes: {total_mse_tensor.item()}")
#
#    if local_rank == 0:
#        # Calculate aggregated metrics
#        avg_mse = total_mse_tensor.item() / total_samples_tensor.item()
#        avg_mae = total_mae_tensor.item() / total_samples_tensor.item()
#        avg_rel_l2 = total_rel_l2_tensor.item() / total_samples_tensor.item()
#        avg_rel_l1 = total_rel_l1_tensor.item() / total_samples_tensor.item()
#
#        # Calculate R² score - only on rank 0 with locally collected data
#        all_outputs = torch.cat(all_outputs, dim=0).numpy()
#        all_targets = torch.cat(all_targets, dim=0).numpy()
#        tmp = np.mean(all_targets)
#        logging.info("mean value for all_targets: {tmp}")
#        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
#        ss_res = np.sum((all_targets - all_outputs) ** 2)
#        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
#
#        # Calculate max AE
#        max_ae = np.max(np.abs(all_targets - all_outputs))
#        logging.info(
#            f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max AE: {max_ae:.6f}, Test R2: {r_squared:.4f}"
#        )
#        logging.info(
#            f"Relative L2 Error: {avg_rel_l2:.6f}, Relative L1 error: {avg_rel_l1:.6f}"
#        )
#        logging.info(
#            f"Total inference time: {total_inference_time: .2f}s for {total_samples_tensor.item()} samples"
#        )
#
#        # Save metrics to a text file
#        metrics_file = os.path.join(exp_dir, "test_metrics.txt")
#        with open(metrics_file, "w") as f:
#            f.write(f"Test MSE: {avg_mse:.6f}\n")
#            f.write(f"Test MAE: {avg_mae:.6f}\n")
#            f.write(f"Max MAE: {max_ae:.6f}\n")
#            f.write(f"Test R2: {r_squared:.4f}\n")
#            f.write(f"Relative L2 Error: {avg_rel_l2:.6f}\n")
#            f.write(f"Relative L1 error: {avg_rel_l1:.6f}\n")
#            f.write(
#                f"Total inference time: {total_inference_time: .2f}s for {total_samples_tensor.item()} samples\n"
#            )
#


def main():
    """main function to parse arguments and start training."""
    args = parse_args()

    # Set the master address and port for DDP
    os.environ["MASTER_ADDR"] = "localhost"
    port = random.randint(1024, 65535)
    os.environ["MASTER_PORT"] = str(port)


    # Set visible GPUS
    gpu_list = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    # Count number of GPUs to use
    world_size = len(gpu_list.split(","))

    # Create experiment directory
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Start distributed training
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
