import logging
import einops
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

from modules.rope import rope
from modules.attention.serialized_attention import Serialized_Attention


class DotProductAttention(nn.Module):
    """Scaled dot-product attention module.

    Args:
        dim: Input dimension of the attention module.
        num_heads: Number of attention heads. Defaults to 8.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        patch_size: int = 20,
        shift: int = 2,
        dropout: int = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.local_ln_1 = nn.LayerNorm(dim)
        self.local_attention = Serialized_Attention(
            dim, num_heads, patch_size, shift, dropout=0.1
        )
        self.local_ln_2 = nn.LayerNorm(dim)
        self.local_gate = nn.Parameter(torch.tensor([0.0]))
        self.ln_3 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        order: torch.Tensor,
        inverse: torch.Tensor,
        geometry: Optional[bool] = False,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, dim).
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.

        Returns:
            (batch_size, sequence_length, dim)
        """
        xn = x
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs)

        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(
            x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)"
        )
        x = self.proj(x)

        if geometry is not True:
            x1 = self.proj(x)  # dim = 192

            # local attention
            x2 = torch.zeros_like(x1)

            logging.info(f"xn.shape: {xn.shape}")
            logging.info(f"x1.shape: {x1.shape}")
            logging.info(f"x2.shape: {x2.shape}")
            logging.info(f"order.shape: {order.shape}")
            logging.info(f"inverse.shape: {inverse.shape}")

            for ii in range(x2.shape[0]):
                x2[ii] = (xn - x1)[ii, order[ii], :]  # serialized
            x2 = self.local_ln_2(self.local_attention(self.local_ln_1(x2)))
            for ii in range(x2.shape[0]):
                x2[ii] = x2[ii, inverse[ii], :]  # deserialized

            #        x = self.mlp(self.ln_3(x0 + x1 + self.local_gate * x2)) + x0
            x = self.ln_3(x1 + self.local_gate * x2)

        return x
