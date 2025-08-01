#!/usr/bin/env python3
# run_pipeline.py
"""
@author: Bojun Zhang

Pipeline script for the Transolver pressure field prediction project.

This script provides a complete pipeline for training and evaluating
pressure field prediction models on the DrivAerNet++ dataset, including
data preprocessing, model training, and result visualization.
"""

import os
import argparse
import subprocess
import logging
import time
import pprint
from datetime import datetime
from utils import setup_logger
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
    parser = argparse.ArgumentParser(description="Run the complete Transolver pipeline")
    # Pipeline control
    parser.add_argument('--stages', type=str, default='all',
                        choices=['preprocess', 'train', 'evaluate', 'all'],
                        help='Pipeline stages to run')
    # Basic settings
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--subset_dir', type=str, required=True, help='Path to train/val/test splits')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default="0", help='GPUs to use (comma-separated)')

    # Model settings
    parser.add_argument('--model', type=str, default='Transolver_2D')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
    parser.add_argument('--n-layers', type=int, default=3, help='layers')
    parser.add_argument('--n-heads', type=int, default=4)

    # Evaluation settings
    parser.add_argument('--num_eval_samples', type=int, default=5, help='Number of samples to evaluate in detail')



def main():
    """ main function to run the complete pipeline. """
    args = parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "Tran Test"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    setup_logger(log_file)
    logging.info(f"{Fore.RED}*************************Benchmark: Start pressure prediction in Darcy flow.{Style.RESET_ALL}")


if __name__=="__main__":
    exit(main())
