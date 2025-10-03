from functools import partial

import einops
import torch
import logging
from torch import nn

from modules.attention import (
    AnchorAttention,
    SharedweightsCrossattnAttention,
    SharedweightsSplitattnAttention,
)
from model.transolver_block import TransolverBlock
from model.sharedweights_split_attention import SharedweightsSplitAttention
from model.MLP import MLP
from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # geometry
        self.preprocess = MLP(n_input=args.n_dim, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=args.n_layers, res=args.res)
        self.geometry_blocks = nn.ModuleList(
            [
                TransolverBlock(args, last_layer=(_ == args.geometry_depth - 1))
                for _ in range(args.geometry_depth)
            ]
        )

        # pos_embed with separate MLP for surface/volume
#        self.pos_embed = ContinuousSincosEmbed(ndim=args.n_dim, dim=args.n_hidden)
        self.pos_embed = MLP(n_input=args.n_dim, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=args.n_layers, res=args.res)
        self.surface_bias = nn.Sequential(
            nn.Linear(args.n_hidden, args.n_hidden),
            nn.GELU(),
            nn.Linear(args.n_hidden, args.n_hidden)
        )
        self.volume_bias = nn.Sequential(
            nn.Linear(args.n_hidden, args.n_hidden),
            nn.GELU(),
            nn.Linear(args.n_hidden, args.n_hidden)
        )

        # weight-shared blocks
        self.blocks = nn.ModuleList()
        for block in args.blocks:
            if block == "s":
                # weight-shared self-attention within surface/volume tokens
                block_ctor = partial(TransolverBlock, attn_ctor=SharedweightsSplitAttention)
                for _ in range(args.geometry_depth):
                    is_last = (_ == args.geometry_depth - 1)
                    self.blocks.append(block_ctor(args, last_layer=is_last))
            elif block == "c":
                # weight-shared cross-attention between surface/volume tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsCrossattnAttention)
                self.blocks.append(block_ctor(args))
            elif block == "p":
                # weight-shared cross-attention from surface/volume tokens to geometry tokens
                block_ctor = PerceiverBlock
                self.blocks.append(block_ctor(args))
            else:
                raise NotImplementedError

    def forward(
            self,
            # Geometry
            geometry_position: torch.Tensor,
            # Surface
            surf_position: torch.Tensor,
            surf_wss: torch.Tensor,
            surf_pressure: torch.Tensor,
            # Volume
            volume_position: torch.Tensor,
            volume_pressure: torch.Tensor,
            volume_wss: torch.Tensor,
            volume_vel: torch.Tensor
    )-> dict[str, torch.Tensor]:

        # Create split size
        split_size = [surf_position.size(1), volume_position.size(1)]

        # Encode geometry
        geometry_position = self.preprocess(geometry_position)
        for geometry_block in self.geometry_blocks:
            geometry_position = geometry_block(geometry_position)
        geometry_encoding = geometry_position

        # Shared-weights model (all tokens are concatenated into a single sequence for high GPU utilization)
        sn_dim = surf_position.size(-1)
        vn_dim = volume_position.size(-1)
        assert sn_dim == 3 and vn_dim == 3

        surface_pos_embed = self.surface_bias(self.pos_embed(surf_position))
        volume_pos_embed = self.volume_bias(self.pos_embed(volume_position))
        x = torch.concat([surface_pos_embed, volume_pos_embed], dim=1)
        logging.info(f"x.shape: {x.shape}")
        for block in self.blocks:
            x = block(x, attn_kwargs=dict(split_size=split_size))

        x_surf, x_volume = x.split(split_size, dim=1)


        return geometry_position


