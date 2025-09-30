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
from torch.utils.data.distributed import DistributedSampler

# Import modules
from data_loader import get_dataloaders
from WSSdata_loader import get_WSSdataloaders
from CADdata_loader import get_CADdataloaders
from CombinedDataset import CombinedDataset
from model.Driver_MBTransolver import Model
from src.test_model import test_model
from src.train_one_epoch import train_one_epoch
from utils.utils import setup_logger, setup_seed
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from colorama import Fore, Style

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
        #logging.info(f"Arguments:\n" + pprint.pformat(vars(args), indent=2))
        logging.info(f"Starting training with {world_size} GPUs")

    # Initialize model
    model = initialize_model(args, local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")

    # Prepare Pressure DataLoaders
    Ptrain_dataloader, Pval_dataloader, Ptest_dataloader = get_dataloaders(
        args.Pdataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.Pcache_dir, args.num_workers
        )

    # Prepare wall shear stress  DataLoaders
    WSStrain_dataloader, WSSval_dataloader, WSStest_dataloader = get_WSSdataloaders(
        args.Wdataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.Wcache_dir, args.num_workers
        )

    # Prepare geometry DataLoaders
    Ctrain_dataloader, Cval_dataloader, Ctest_dataloader = get_CADdataloaders(
        args.Cdataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.Ccache_dir, args.num_workers
        )

    # Combined them
    Combined_TrainDataset = CombinedDataset(Ptrain_dataloader.dataset, WSStrain_dataloader.dataset, Ctrain_dataloader.dataset)
    train_sampler = DistributedSampler(
            Combined_TrainDataset,
            num_replicas=world_size,   # usually torch.distributed.get_world_size()
            rank=rank,                      # usually torch.distributed.get_rank()
            shuffle=True
    )
    train_dataloader = DataLoader(
            Combined_TrainDataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            drop_last=True)

    # Log dataset info
    if local_rank == 0:
        logging.info(
           # f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")
            f"Data loaded: {len(train_dataloader)} training batches ")

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


