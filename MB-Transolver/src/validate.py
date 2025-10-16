import os
import torch
import logging
from tqdm import tqdm
from colorama import Fore, Style

PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25
def validate(model, val_dataloader, criterion, local_rank):
    """ Validate the model"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for combined_data in tqdm(val_dataloader, desc="[Validation]"):

            bs, num_points, _ = combined_data["geometry_position"].shape
            num_geometry_supernodes = int(num_points * 0.2)
            num_volume_anchors = num_geometry_supernodes
            num_surface_queries = num_geometry_supernodes
            num_volume_queries = num_geometry_supernodes
            num_surface_anchors = num_geometry_supernodes

            data = dict(
                geometry_position = combined_data["geometry_position"],
                geometry_supernode_idx = torch.rand(num_geometry_supernodes),
                geometry_batch_idx=None,
                # anchors
                surface_anchor_position = torch.rand(bs, num_points, 3),
                volume_anchor_position = torch.rand(bs, num_points, 3),
                # queries
                surface_query_position = torch.rand(bs, num_points, 3),
                volume_query_position = torch.rand(bs, num_points, 3),
            )
            # geometry
            data["geometry_position"] = data["geometry_position"].squeeze(1).to(local_rank)
            data["geometry_position"] = data["geometry_position"].permute(0, 2 ,1).contiguous()

            geometry_position = data["geometry_position"]
            geometry_perm = torch.randperm(geometry_position.shape[1])
            geometry_supernode_idxs = geometry_perm[:num_geometry_supernodes]
            data["geometry_supernode_idxs"] = geometry_supernode_idxs
            data["geometry_position"] = data["geometry_position"].squeeze(0)

            # surface
            surface_position = combined_data["surface_position_1"]
            surface_perm = torch.randperm(surface_position.shape[1])
            surface_anchor_idxs = surface_perm[:num_surface_anchors]
            surface_query_idxs = surface_perm[num_surface_anchors:]
            data["surface_anchor_position"] = surface_position[:, surface_anchor_idxs, :]
            data["surface_query_position"] = surface_position[:, surface_query_idxs, :]

            # volume
            volume_position = combined_data["volume_position"]
            volume_perm = torch.randperm(volume_position.shape[1])
            volume_anchor_idxs = volume_perm[:num_volume_anchors]
            volume_query_idxs = volume_perm[num_volume_anchors:]
            data["volume_anchor_position"] = volume_position[:, volume_anchor_idxs, :]
            data["volume_query_position"] = volume_position[:, volume_query_idxs, :]


            # Make data and perssure same shape
            targets_surface_pressure = combined_data["surface_pressure"]
            targets_surface_pressure = targets_surface_pressure.to(local_rank)
            targets_surface_pressure = targets_surface_pressure.squeeze(1).contiguous()

            outputs     = model(**data)
            pre_surface_pressure = outputs["surface_pressure"]                   #(B, N, pressure_dim)

            # Normalize targets
            targets_surface_pressure = combined_data["surface_pressure"]
            targets_surface_pressure = (targets_surface_pressure - PRESSURE_MEAN) / PRESSURE_STD
            targets_surface_pressure = targets_surface_pressure.permute(0, 2, 1).contiguous().to(local_rank)

            logging.info(f"pre_surface_pressure.shape: {pre_surface_pressure.shape}")
            logging.info(f"targets_surface_pressure.shape: {targets_surface_pressure.shape}")

            loss        = criterion(pre_surface_pressure, targets_surface_pressure)
            total_loss += loss.item()

    return total_loss / len(val_dataloader)

