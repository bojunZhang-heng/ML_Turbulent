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
from src.train_one_epoch import train_one_epoch
from src.test_model import test_model
from src.train_and_evaluate import train_and_evaluate

# Import modules
from torch.utils.data.distributed import DistributedSampler
from model.Driver_MBTransolver import Model
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from WSSdata_loader import get_WSSdataloaders
from CADdata_loader import get_CADdataloaders
from CombinedDataset import CombinedDataset
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
    parser = argparse.ArgumentParser(description='Train multi physcial field prediction models on MB-Transolver')

    # Basic settings
    parser.add_argument('--exp_name', type=str, default='PressurePrediction', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--Cdataset_path', type=str,  help='Path to dataset')
    parser.add_argument('--Pdataset_path', type=str,  help='Path to dataset')
    parser.add_argument('--Wdataset_path', type=str,  help='Path to dataset')
    parser.add_argument('--Vdataset_path', type=str,  help='Path to dataset')
    parser.add_argument('--Ccache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--Pcache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--Wcache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--Vcache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--subset_dir', type=str, help='Path to train/val/test splits')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_only', type=int, default=0, help='Only test the model, no training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_hidden', type=int, default=128, help='hidden dim')
    parser.add_argument('--n_layers', type=int, default=3, help='layers')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--slice_num', type=int, default=32)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--n_dim', type=int, default=3)
    parser.add_argument('--n_input', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim_surface', type=int, default=4)
    parser.add_argument('--output_dim_volume', type=int, default=7)
    parser.add_argument('--geometry_depth', type=int, default=1)
    parser.add_argument('--num_volume_blocks', type=int, default=6)
    parser.add_argument('--num_surface_blocks', type=int, default=6)
    parser.add_argument('--blocks', type=str, default="pscscs")
    parser.add_argument('--res', type=str, default="True")

    return parser.parse_args()

def initialize_model(args, local_rank):
    """ Initialize and return the RegDGCN model. """

    model = Model(args).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            output_device=local_rank
    )

    return model

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
