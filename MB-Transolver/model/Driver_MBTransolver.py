from functools import partial

import einops
import torch
from torch import nn

from modules.attention import (
    AnchorAttention,
    SharedweightsCrossattnAttention,
    SharedweightsSplitattnAttention,
)
from model.Transolver_block import Transolver_block
from model.MLP import MLP
from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
    #    self.rope = RopeFrequency(dim=dim // num_heads, ndim=input_dim)
        # geometry
        self.preprocess = MLP(n_input=args.n_dim, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=args.n_layers, res=args.res)
        self.geometry_blocks = nn.ModuleList(
            [
                Transolver_block(args, last_layer=(_ == args.geometry_depth - 1))
                for _ in range(args.geometry_depth)
            ]
        )

    def forward(
            self,
            # Geometry
            geometry_position: torch.Tensor,
            # Surface
            wss_position: torch.Tensor,
            wss_value: torch.Tensor,
            pressure_position: torch.Tensor,
            pressure_value: torch.Tensor
    )-> dict[str, torch.Tensor]:
        geometry_position = self.preprocess(geometry_position)

        for geometry_block in self.geometry_blocks:
            geometry_position = geometry_block(geometry_position)


        return geometry_position


