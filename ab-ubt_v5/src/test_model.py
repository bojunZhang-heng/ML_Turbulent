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
import logging
import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import modules
from torch.utils.data.distributed import DistributedSampler
from utils.utils import setup_logger, setup_seed
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from colorama import Fore, Style

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

