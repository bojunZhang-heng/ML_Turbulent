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
from utils.utils import setup_logger
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
    parser.add_argument('--test_only', type=int, default=0, help='Only test the model, no training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default="0", help='GPUs to use (comma-separated)')

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
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--slice_num', type=int, default=32)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--downsample', type=int, default=5)

    # Evaluation settings
    parser.add_argument('--num_eval_samples', type=int, default=5, help='Number of samples to evaluate in detail')

    return parser.parse_args()

def preprocess_data(args):

    """
    Preprocess the dataset to create cached point cloud data.

    Args:
        True if preprocessing was successful, False otherwise
    """
    logging.info("**************************Starting data preprocessing...")

    # Create cache directory if it doesn't exist
    cache_dir = args.cache_dir or os.path.join(args.dataset_path, "processed_data")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Import required modules for preprocessing
        from data_loader import SurfacePressureDataset

        # Create the dataset with preprocessing enabled
        dataset = SurfacePressureDataset(
            root_dir = args.dataset_path,
            num_points = args.num_points,
            preprocess = True,
            cache_dir = cache_dir
            )

        # Process all files
        logging.info(f"Processing {len(dataset.vtk_files)} VTK files with {args.num_points} points per sample")
        for ii, vtk_file in enumerate(dataset.vtk_files):
            logging.info(f"Processing file {ii+1} / {len(dataset.vtk_files)}: {os.path.basename(vtk_file)}")
            _ = dataset[ii] # This will trigger preprocessing and caching

        logging.info(f"{Fore.MAGENTA}Data preprocessing complete. Cache data saved to {cache_dir}{Style.RESET_ALL}")
        return True

    except Exception as e:
        logging.error(f"Preprocessing failed with error: {e}")
        return False

def train_model(args):
    logging.info(f"{R} *************************Starting model training... {RESET}")

    # Prepare command for training script
    cmd = [
        "python", "train.py",
        "--exp_name", args.exp_name,
        "--model", args.model,
        "--dataset_path", args.dataset_path,
        "--subset_dir", args.subset_dir,
        "--num_points", str(args.num_points),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--slice_num", str(args.slice_num),
        "--dropout", str(args.dropout),
        "--emb_dims", str(args.emb_dims),
        "--k", str(args.k),
        "--output_channels", str(args.output_channels),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--test_only", str(args.test_only),
        "--n_hidden", str(args.n_hidden),
        "--n_heads", str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--lr", str(args.lr),
        "--max_grad_norm", str(args.max_grad_norm),
        "--unified_pos", str(args.unified_pos),
        "--ref", str(args.ref),
        "--downsample", str(args.downsample),
        "--mlp_ratio", str(args.mlp_ratio)

    ]

    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])

    # Set up environment variables for distributed training
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Run the training script
    start_time = time.time()
    process = subprocess.Popen(cmd, env=env)
    process.wait()

    if process.returncode != 0:
        logging.error(f"{R} Training failed! {RESET}")
        return False

    elapsed_time = time.time() - start_time
    logging.info(f"**********************Model training completed ")
    logging.info(f"Model training completed in {elapsed_time:.2f} seconds")

    return True

def main():
    """ main function to run the complete pipeline. """
    args = parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "Test"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    setup_logger(log_file)
    logging.info(f"{R} ************************* Start pressure prediction for 3D automobile geometry.{RESET}")
    logging.info(f"{G} Arguments: {RESET}\n" + pprint.pformat(vars(args), indent=2))

    # Execute the selected pipeline stages
    stages = args.stages.split(',') if ',' in args.stages else [args.stages]
    if 'all' in stages:
        stages = ['preporcess', 'train', 'evaluate']

    results = {}

    # Preprocess stage
    if 'preprocess' in stages:
        results['preprocess'] = preprocess_data(args)
    else:
        results['preprocess'] = True
        logging.info("Preprocessing stage skipped.")

    # Train model stage
    if 'train' in stages and results['preprocess']:
        results['train'] = train_model(args)
    else:
        if 'train' not in stages:
            results['train'] = True
            logging.info(f"Training stage skipped.")
    # Evaluate model stage
    if 'evaluate' in stages and results.get('train', False):
        results['evaluate'] = evaluate_model(args)
    else:
        if 'evaluate' not in stages:
            results['evaluate'] = True
            logging.info(f"Evluation stage skipped.")

    # Print Summary
    logging.info(f"Pipeline execution complete.")
    logging.info(f"{R} Results summary: {RESET} ")
    for stage, success in results.items():
        status = "Success" if success else "Failed"
        logging.info(f" {stage}: {status}")


if __name__=="__main__":
    exit(main())
