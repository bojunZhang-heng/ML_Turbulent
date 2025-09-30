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
            geometry_position = combined_data["geometry"],
            wss_position = combined_data["wss_point"],
            wss_value = combined_data["wss_value"],
            pressure_position = combined_data["pressure_point"],
            pressure_value = combined_data["pressure_value"]
        )

        data["geometry_position"] = data["geometry_position"].squeeze(1).to(local_rank)
        data["geometry_position"] = data["geometry_position"].permute(0, 2 ,1).contiguous()

        optimizer.zero_grad()
        outputs = model(**data)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

