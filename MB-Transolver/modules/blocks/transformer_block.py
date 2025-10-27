from typing import Any

import torch
from torch import nn

from modules.attention import DotProductAttention
from modules.mlp import Mlp
from typing import Optional


class TransformerBlock(nn.Module):
    """A transformer block with a single attention layer and a feedforward layer.

    Args:
        dim: hidden Dimension of the transformer block.
        num_heads: Number of attention heads.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            patch_size: int = 20,
            shift: int = 2,
            dropout: int = 0.1,
            attn_ctor: type[nn.Module] = DotProductAttention):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attn_ctor(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(
        self,
        x: torch.Tensor,
        order: Optional[torch.Tensor] = None,
        inverse: Optional[torch.Tensor] = None,
        geometry: Optional[bool] = False,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor with shape (batch_size, seqlen/num_tokens, dim).
            attn_kwargs: Dict with arguments for the attention (such as the rope frequencies). Defaults to None.

        Returns:
            (batch_size, num_tokens, dim)
        """
        x = x + self.attn(self.norm1(x), order, inverse, geometry, **(attn_kwargs or {}))
        x = x + self.mlp(self.norm2(x))
        return x
