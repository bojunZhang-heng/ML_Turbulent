import einops
import torch
import logging

from torch import nn
from functools import partial
from model.MLP import MLP
from model.transolver_block import TransolverBlock
from model.perceiver_block import PerceiverBlock
from model.perceiver_attention import PerceiverAttention
from model.sharedweights_split_attention import SharedweightsSplitAttention
from model.sharedweights_cross_attention import SharedweightsCrossAttention

from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.rope = RopeFrequency(dim=args.n_hidden // agrs.n_heads , ndim=args.n_dim)
        # geometry
        self.preprocess = MLP(n_input=args.n_dim, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=args.n_layers // 2, res=args.res)
        self.geometry_blocks = nn.ModuleList(
            [
                TransolverBlock(args, last_layer=(_ != args.geometry_depth - 1))
                for _ in range(args.geometry_depth)
            ]
        )

       # pos_embed with separate MLP for surface/volume
        #self.pos_embed = ContinuousSincosEmbed(ndim=args.n_dim, dim=args.n_hidden)
        self.pos_embed = MLP(n_input=args.n_dim, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=args.n_layers // 2, res=args.res)
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
        x = block(q=x, kv=geometry_encoding, attn_kwargs=geometry_perceiver_attn_kwargs)
        self.perceiver = PerceiverBlock(args=args, attn_ctor=PerceiverAttention)
        self.blocks = nn.ModuleList()
        for block in args.blocks:
            if block == "s":
                # weight-shared self-attention within surface/volume tokens
                block_ctor = partial(TransolverBlock, attn_ctor=SharedweightsSplitAttention)
                for _ in range(args.geometry_depth):
                    is_last = (_ != args.geometry_depth - 1)
                    self.blocks.append(block_ctor(args, last_layer=is_last))
            elif block == "c":
                # weight-shared cross-attention between surface/volume tokens
                block_ctor = partial(TransolverBlock, attn_ctor=SharedweightsCrossAttention)
                for _ in range(args.geometry_depth):
                    is_last = (_ != args.geometry_depth - 1)
                    self.blocks.append(block_ctor(args, last_layer=is_last))
            else:
                raise NotImplementedError

        # surface-specific blocks
        self.surface_blocks = nn.ModuleList(
            [
                TransolverBlock(args, last_layer=False)
                for _ in range(args.surf_depth)
            ]
        )
        self.surface_decoder = nn.Linear(args.n_hidden, args.output_dim_surface)

        # volume-specific blocks
        self.volume_blocks = nn.ModuleList(
            [
                TransolverBlock(args, last_layer=False)
                for _ in range(args.volume_depth)
            ]
        )
        self.volume_decoder = nn.Linear(args.n_hidden, args.output_dim_volume)

    def forward(
            self,
            # Geometry
            geometry_position: torch.Tensor,
            geometry_batch_idx: torch.Tensor | None,
            # Surface
            surf_position: torch.Tensor,
            surf_position_2: torch.Tensor,
            surf_wss: torch.Tensor,
            surf_pressure: torch.Tensor,
            # Volume
            volume_position: torch.Tensor,
            volume_pressure: torch.Tensor,
            volume_wss: torch.Tensor,
            volume_vel: torch.Tensor
    )-> dict[str, torch.Tensor]:

        outputs = {}
        geometry_attn_kwargs = {}
        surface_decoder_attn_kwargs = {}
        volume_decoder_attn_kwargs = {}
        geometry_perceiver_attn_kwargs = {}
        shared_attn_kwargs = {}

        # Create split size
        volume_position = surf_position_2
        split_size = [surf_position.size(1), volume_position.size(1)]

        # rope frequencies
        assert geometry_batch_idx is None or geometry_batch_idx.unique().numel() == 1, "batch_size > 1 not supported"
        geometry_rope = self.rope(geometry_position.unsqueeze(0))
        geometry_attn_kwargs["freqs"] = geometry_rope
        rope_surface_all = self.rope(surf_position)
        rope_volume_all = self.rope(volume_position)
        rope_all = torch.concat([rope_surface_all, rope_volume_all], dim=1)

        geometry_perceiver_attn_kwargs["q_freqs"] = rope_all
        geometry_perceiver_attn_kwargs["k_freqs"] = geometry_rope
        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        volume_decoder_attn_kwargs["freqs"] = rope_volume_all
        shared_attn_kwargs["freqs"] = rope_all

        # Encode geometry
        geometry_position = self.preprocess(geometry_position)
        for geometry_block in self.geometry_blocks:
            geometry_position = geometry_block(geometry_position, attn_kwargs=geometry_attn_kwargs)
        geometry_encoding = geometry_position

        # Shared-weights model (all tokens are concatenated into a single sequence for high GPU utilization)
        sn_dim = surf_position.size(-1)
        vn_dim = volume_position.size(-1)
        assert sn_dim == 3 and vn_dim == 3

        surface_pos_embed = self.surface_bias(self.pos_embed(surf_position))
        volume_pos_embed = self.volume_bias(self.pos_embed(volume_position))

        # perceiver_block
        x_surf, x_volume = self.perceiver(surface_pos_embed, volume_pos_embed, geometry_encoding,
                                          attn_kwargs=geometry_perceiver_attn_kwargs)
        x = torch.concat([x_surf, x_volume], dim=1)

        for block in self.blocks:
            x = block(x, attn_kwargs=dict(split_size=split_size))

        x_surf, x_volume = x.split(split_size, dim=1)

        # surface blocks
        for block in self.surface_blocks:
            x_surf = block(x_surf)
        x_surf = self.surface_decoder(x_surf)            #(6, 10000, 3)

        outputs = {}
        outputs["surf_pressure"] = x_surf                # (B, N, pressure_dim)

        # volume blocks
        for block in self.volume_blocks:
            x_volume = block(x_volume)
        x_volume = self.volume_decoder(x_volume)             #(6, 10000, 3)

        outputs["surf_wss"] = x_volume                       #(6, 10000, 3)


        return outputs


