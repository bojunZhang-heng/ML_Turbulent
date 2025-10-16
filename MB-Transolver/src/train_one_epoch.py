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

        bs, num_points, _ = combined_data["geometry_position"].shape
        num_geometry_positions = 655
        num_geometry_supernodes = int(num_points * 0.2)
        num_volume_anchors = 280
        num_surface_queries = 301
        num_volume_queries = 321
        num_surf_anchors = int(num_points * 0.2)

        data = dict(
            geometry_position = combined_data["geometry_position"],
            geometry_supernode_idxs = torch.rand(num_geometry_supernodes)
            surf_position = combined_data["surf_position_1"],
            surf_position_2 = combined_data["surf_position_2"],
            volume_position = combined_data["volume_position"],
            # anchors
            surf_anchor_position = torch.rand(bs, num_points, 3)
            volume_anchor_position = torch.rand(bs, num_points, 3)
            # queries
            surf_query_position = torch.rand(bs, num_points, 3)
            volume_query_position = torch.rand(bs, num_points, 3)

        )

        data["geometry_position"] = data["geometry_position"].squeeze(1).to(local_rank)
        data["geometry_position"] = data["geometry_position"].permute(0, 2 ,1).contiguous()

        data["surf_position"] = data["surf_position"].squeeze(1).to(local_rank)
        data["surf_position"] = data["surf_position"].permute(0, 2 ,1).contiguous()

        data["surf_position_2"] = data["surf_position_2"].squeeze(1).to(local_rank)
        data["surf_position_2"] = data["surf_position_2"].permute(0, 2 ,1).contiguous()

        data["volume_position"] = data["volume_position"].squeeze(1).to(local_rank)
        data["volume_position"] = data["volume_position"].permute(0, 2 ,1).contiguous()

        # geometry
        geometry_position = data["geometry_position"]
        geometry_perm = torch.randperm(geometry_position.shape[1])
        geometry_supernode_idxs = geometry_perm[:num_geometry_supernodes]
        data["geometry_supernode_idxs"] = geometry_supernode_idxs

        # surf
        surf_position = data["surf_position"]
        surf_perm = torch.randperm(len(surf_position[1]))
        surf_anchor_idxs = surf_perm[:num_surf_anchors]
        surf_query_idxs = surf_perm[num_surf_anchors:]
        data["surf_anchor_position"] = surf_position[:, surf_anchor_idxs, :]
        data["surf_query_position"] = surf_position[:, surf_query_idxs, :]

        # volume
        volume_position = data["volume_position"]
        volume_perm = torch.randperm(len(volume_position[1]))
        volume_anchor_idxs = volume_perm[:num_volume_anchors]
        volume_query_idxs = volume_perm[num_volume_anchors:]
        data["volume_anchor_position"] = volume_position[:, volume_anchor_idxs, :]
        data["volume_query_positio"] = volume_position[:, volume_query_idxs, :]
        surf_position = data["surface_position"]
        surface_perm = torch.randperm(len(surface_position[1]))
        surface_anchor_idxs = surface_perm[:num_surface_anchors]
        surface_query_idxs = surface_perm[num_surface_anchors:]
        data["surf_anchor_position"] = surface_position[:, surface_anchor_idxs, :]

        optimizer.zero_grad()
        outputs = model(**data)

        pre_surf_pressure = outputs["surf_pressure"]                   #(B, N, pressure_dim)
        targets_surf_pressure = data["surf_pressure"]
        targets_surf_pressure = targets_surf_pressure.permute(0, 2, 1).contiguous().to(local_rank)

 #       logging.info(f"pre_surf_pressure.shape: {pre_surf_pressure.shape}")
 #       logging.info(f"targets_surf_pressure.shape: {targets_surf_pressure.shape}")

        loss = criterion(pre_surf_pressure, targets_surf_pressure)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

