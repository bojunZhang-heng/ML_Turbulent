from typing import Any

import torch
from torch import nn

from model.perceiver_attention import PerceiverAttention
from model.MLP import MLP


class PerceiverBlock(nn.Module):
    """The PerceiverBlock takes different input tensors for the query and the key/value.

    Args:
        args
        attn_ctor
    """

    def __init__(self, args, attn_ctor: type[nn.Module]=PerceiverAttention):
        super().__init__()
        self.ln_1q = nn.LayerNorm(args.n_hidden, eps=1e-6)
        self.ln_1k = nn.LayerNorm(args.n_hidden, eps=1e-6)
        self.ln_1v = nn.LayerNorm(args.n_hidden, eps=1e-6)
        self.Attn = PerceiverAttention(args)
        self.ln_2 = nn.LayerNorm(args.n_hidden, eps=1e-6)
        self.mlp = MLP(n_input=args.n_hidden, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=3, res=args.res)

    def forward(
        self,
        q_surf: torch.Tensor,
        q_volume: torch.Tensor,
        kv_geometry: torch.Tensor,
        attn_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        """Forward pass of the PerceiverBlock.

        Args:
            q_surf: Input tensor with shape (batch_size, num_points, n_hidden) for the query representations.
            q_volume: Input tensor with shape (batch_size, num_points, n_hidden) for the query representations.
            kv: Input tensor with shape (batch_size, num_points, n_hidden) for the key and value representations.
            attn_kwargs: Dict with arguments for the attention (such as rope frequencies). Defaults to None.

        Returns:
            x_surf: (batch_size, num_points, n_hidden)
            x_volume: (batch_size, num_points, n_hidden)
        """
        q_surf = self.ln_1q(q_surf)
        q_volume = self.ln_1q(q_volume)
        k_geometry = self.ln_1k(kv_geometry)
        v_geometry = self.ln_1v(kv_geometry)
        x_surf, x_volume = self.Attn(
                q_surf=q_surf, q_volume=q_volume,
                k_geometry=k_geometry, v_geometry=v_geometry,
                **(attn_kwargs or {}),
                )
        x_surf = x_surf + q_surf
        x_volume = x_volume + q_volume

        x_surf = self.mlp(self.ln_2(x_surf)) + x_surf
        x_volume = self.mlp(self.ln_2(x_volume)) + x_volume

        return x_surf, x_volume

