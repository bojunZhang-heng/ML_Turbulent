# train.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import pprint

# Import modules
from model.model_dict import get_model
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from utils.utils import setup_logger, setup_seed
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from colorama import Fore, Style

#! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train pressure prediction models on DrivAerNet++')

    # Basic settings
    parser.add_argument('--exp_name', type=str, default='PressurePrediction', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--dataset_path', type=str,  help='Path to dataset')
    parser.add_argument('--subset_dir', type=str, help='Path to train/val/test splits')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_only', type=int, default=0, help='Only test the model, no training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--model', type=str, default='Transolver_2D')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_hidden', type=int, default=64, help='hidden dim')
    parser.add_argument('--n_layers', type=int, default=3, help='layers')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--slice_num', type=int, default=32)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--mlp_ratio', type=int, default=1)

    return parser.parse_args()

def initialize_model(args, local_rank):
    """ Initialize and return the RegDGCN model. """
   # args = vars(args)
    model = get_model(args).Model(space_dim=3,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  Time_Input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=1,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos,
                                  k=args.k
                                  ).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            output_device=local_rank
    )

    return model

def train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data, targets in tqdm(train_dataloader, desc="[Training]"):
        global PRESSURE_MEAN, PRESSURE_STD

        # Make data and perssure same shape
        data = data.squeeze(1).to(local_rank)                          # [B, 1, point_dim, num_points] -> [B, point_dim, num_points]
        data = data.permute(0, 2, 1).contiguous()                      # [B, point_dim, num_points]    -> [B, num_points, point_dim]

        targets = targets.to(local_rank)
        targets = targets.permute(0, 2, 1).contiguous()                # [B, pressure_dim, num_points] -> [B, num_points, pressure_dim]

        # Normalize targets
        targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

        optimizer.zero_grad()
        outputs = model(data)                                          # [B, num_points, point_dim]    -> [B, num_points, pressure_dim]
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

def validate(model, val_dataloader, criterion, local_rank):
    """ Validate the model"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, targets in tqdm(val_dataloader, desc="[Validation]"):

            # Make data and perssure same shape
            data = data.squeeze(1).to(local_rank)
            data = data.permute(0, 2, 1).contiguous()
            targets = targets.to(local_rank)
            targets = targets.permute(0, 2, 1).contiguous()

            # Normalize targets
            targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs     = model(data)
            loss        = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_dataloader)

def test_model(model, test_dataloader, criterion, local_rank, exp_dir):
    """ Test the model, take postprocess and calculate metrics. """
    model.eval()
    total_mse, total_mae = 0, 0
    total_rel_l2, total_rel_l1 = 0, 0
    total_inference_time = 0
    total_samples = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(test_dataloader, desc="[Testing]"):
            start_time = time.time()

            # Make data and perssure same shape
            data = data.squeeze(1).to(local_rank)
            data = data.permute(0, 2, 1).contiguous()
            targets = targets.to(local_rank)
            targets = targets.permute(0, 2, 1).contiguous()

            # Normalize targets
            normalized_targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs = model(data)
            normalized_outputs = outputs.squeeze(1)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Calculate metrics
            mse = criterion(normalized_outputs, normalized_targets)
            mae = F.l1_loss(normalized_outputs, normalized_targets)

            # Calculate relative errors
            rel_l2 = torch.mean(torch.norm(normalized_outputs - normalized_targets, p=2, dim=-1) /
                                torch.norm(normalized_targets, p=2, dim=-1))
            rel_l1 = torch.mean(torch.norm(normalized_outputs - normalized_targets, p=1, dim=-1) /
                                torch.norm(normalized_targets, p=1, dim=-1))

            batch_size = targets.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_rel_l2 += rel_l2.item() * batch_size
            total_rel_l1 += rel_l1.item() * batch_size
            total_samples += batch_size

            # Store normalized predictions and targets for R² calculation
            all_outputs.append(normalized_outputs.cpu())
            all_targets.append(normalized_targets.cpu())

    # Aggregate results across all processes
    total_mse_tensor = torch.tensor(total_mse).to(local_rank)
    total_mae_tensor = torch.tensor(total_mae).to(local_rank)
    total_rel_l2_tensor  = torch.tensor(total_rel_l2).to(local_rank)
    total_rel_l1_tensor  = torch.tensor(total_rel_l1).to(local_rank)
    total_samples_tensor = torch.tensor(total_samples).to(local_rank)

    dist.reduce(total_mse_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_mae_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_rel_l2_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_rel_l1_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Checkout the value
    if dist.get_rank() == 0:
      logging.info(f"Total MSE across all processes: {total_mse_tensor.item()}")

    if local_rank ==0:
        # Calculate aggregated metrics
        avg_mse = total_mse_tensor.item() / total_samples_tensor.item()
        avg_mae = total_mae_tensor.item() / total_samples_tensor.item()
        avg_rel_l2 = total_rel_l2_tensor.item() / total_samples_tensor.item()
        avg_rel_l1 = total_rel_l1_tensor.item() / total_samples_tensor.item()

        # Calculate R² score - only on rank 0 with locally collected data
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        tmp = np.mean(all_targets)
        logging.info("mean value for all_targets: {tmp}")
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate max AE
        max_ae = np.max(np.abs(all_targets - all_outputs))
        logging.info(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max AE: {max_ae:.6f}, Test R2: {r_squared:.4f}")
        logging.info(f"Relative L2 Error: {avg_rel_l2:.6f}, Relative L1 error: {avg_rel_l1:.6f}")
        logging.info(f"Total inference time: {total_inference_time: .2f}s for {total_samples_tensor.item()} samples")

        # Save metrics to a text file
        metrics_file = os.path.join(exp_dir, 'test_metrics.txt')
        with open(metrics_file, 'w') as f:
          f.write(f"Test MSE: {avg_mse:.6f}\n")
          f.write(f"Test MAE: {avg_mae:.6f}\n")
          f.write(f"Max MAE: {max_ae:.6f}\n")
          f.write(f"Test R2: {r_squared:.4f}\n")
          f.write(f"Relative L2 Error: {avg_rel_l2:.6f}\n")
          f.write(f"Relative L1 error: {avg_rel_l1:.6f}\n")
          f.write(f"Total inference time: {total_inference_time: .2f}s for {total_samples_tensor.item()} samples\n")

def train_and_evaluate(rank, world_size, args):
    """ main function for Distributed training and evaluation. """
    setup_seed(args.seed)

    # Initialize process group for DDP
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    local_rank = rank
    torch.cuda.set_device(local_rank)

    # Set up logging (only on rank 0)
    if local_rank == 0:
        exp_dir = os.path.join('experiments', args.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, 'training.log')
        setup_logger(log_file)
        logging.info(f"args.exp_name : {args.exp_name}")
        logging.info(f"Arguments:\n" + pprint.pformat(vars(args), indent=2))
        logging.info(f"{Fore.RED}*******************************Starting training with {world_size} GPUs{Style.RESET_ALL}")

    # Initialize model
    model = initialize_model(args, local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")

    # Prepare DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args.dataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.cache_dir,
        args.num_workers
    )

    # Log dataset info
    if local_rank == 0:
        logging.info(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")

    # Set up criterion, optimizer, and scheduler
    #! There is a puzzle!######
    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_dataloader))

    myloss = TestLoss(size_average=False)
    de_x   = TestLoss(size_average=False)
    de_y   = TestLoss(size_average=False)

    # Store the model
    best_model_path  = os.path.join('experiments', args.exp_name, 'best_model.pth')
    final_model_path = os.path.join('experiments', args.exp_name, 'final_model.pth')

    # Check if test_only and model exists
    if args.test_only and os.path.exists(best_model_path):
        if local_rank == 0:
            logging.info("Loading best model for testing only")
            print("Testing the best model:")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
        test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))
        dist.destroy_process_group()
        return

    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if local_rank == 0:
        logging.info(f"Staring training for {args.epochs} epochs")

    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for the DistributedSampler
        train_dataloader.sampler.set_epoch(epoch)

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank)

        # Validation
        val_loss = validate(model, val_dataloader, criterion, local_rank)

        # Record losses.
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")

            # Update learning rate scheduler
            # scheduler.step(val_loss)

            # Save progress rate scheduler
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
                plt.plot(range(1, epoch + 2), val_losses,   label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'Training Progress - RegDGCNN')
                plt.savefig(os.path.join('experiments', args.exp_name, f'training_progress.png'))
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
    test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))

    # Test the best model
    if local_rank == 0:
        logging.info("Testing the best model")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
    test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))

    # Clean up
    dist.destroy_process_group()

def main():
    """ main function to parse arguments and start training."""
    args = parse_args()

    # Set the master address and port for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set visible GPUS
    gpu_list = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Count number of GPUs to use
    world_size = len(gpu_list.split(','))

    # Create experiment directory
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)


    # Start distributed training
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)


if __name__=="__main__":
    main()
