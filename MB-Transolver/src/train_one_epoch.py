import os
import torch
import logging
from tqdm import tqdm
from colorama import Fore, Style

def train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for combined_data in tqdm(train_dataloader, desc="[Training]"):
        data = dict(
            geometry_position = combined_data["geometry_position"],
            surf_position = combined_data["surf_position"],
            surf_pressure = combined_data["surf_pressure"],
            surf_wss = combined_data["surf_wss"],
            volume_position = combined_data["volume_position"],
            volume_pressure = combined_data["volume_pressure"],
            volume_wss = combined_data["volume_wss"],
            volume_vel = combined_data["volume_vel"]
        )

        data["geometry_position"] = data["geometry_position"].squeeze(1).to(local_rank)
        data["geometry_position"] = data["geometry_position"].permute(0, 2 ,1).contiguous()

        data["surf_position"] = data["surf_position"].squeeze(1).to(local_rank)
        data["surf_position"] = data["surf_position"].permute(0, 2 ,1).contiguous()

        data["volume_position"] = data["volume_position"].squeeze(1).to(local_rank)
        data["volume_position"] = data["volume_position"].permute(0, 2 ,1).contiguous()

        optimizer.zero_grad()
        outputs = model(**data)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

