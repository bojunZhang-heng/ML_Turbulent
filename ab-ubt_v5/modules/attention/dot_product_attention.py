import logging
import einops
import torch
import torch.nn.functional as F
from torch import nn

from modules.rope import rope


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
        slice_num: int = 64,
        dropout: float=0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)

        self.qkv = nn.Linear(dim, 3*dim)
        self.rope_x = nn.Linear(dim, dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_fx = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.head_dim, slice_num)
        self.in_project_deslice = nn.Linear(self.head_dim, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
#        self.in_project_slice = nn.Linear(self.head_dim, self.head_dim)

        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, dim).
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.

        Returns:
            (batch_size, sequence_length, dim)
        """

        B, N, C = x.shape
        # head_dim = 64 C
        # heads = 3 H
        # slice_num = 64 G

        # add rope
        x = einops.rearrange(
            self.rope_x(x),
            "bs seqlen (num_heads head_dim) -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        x = rope(x, freqs=freqs)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")

        ### (1) anchor+query Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights_slice = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_weights_deslice = self.softmax(self.in_project_deslice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights_slice.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_slice)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim)) # B H G C

        ### (2) Attention among anchor+query slice tokens
        q_slice_token = self.to_q(slice_token) # B H G C
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G C

        #out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token) # B H G C

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights_deslice)
        out_x = einops.rearrange(out_x, 'b h n c -> b n (h c)')
        x = self.proj(x)

        return x
