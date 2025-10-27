import einops
import torch
import torch.nn.functional as F

from .dot_product_attention import DotProductAttention
from modules.rope import rope
from typing import Optional


class AnchorAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        order: torch.torch,
        inverse: torch.torch,
        geometry: Optional[bool] = False,
        freqs: Optional[torch.Tensor] = None,
        num_anchor_tokens: int | None = None,
    ) -> torch.Tensor:
        """Self-attention between anchor tokens, other tokens (query tokens) have only cross-attention to anchor tokens

        Args:
            x: Tensor to apply self-attention over, shape (batch_size, sequence_length, dim).
            freqs: Frequencies for RoPE.
            num_anchor_tokens: Number of anchor tokens. If provided, the first num_anchor_tokens of x will be the
                anchors (full self-attention) and the other tokens will be the queries (only cross-attention to the
                anchor tokens).

        Returns:
            (batch_size, sequence_length, dim)
        """
        xn = x
        if num_anchor_tokens is None:
            return super().forward(x=x, order=order, inverse=inverse, freqs=freqs)
        else:
            x, anchor = x.split([num_anchor_tokens, x.size(1) - num_anchor_tokens], dim=1)
        # x is anchor
        # anchor is query
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        # q k v is anchor
        anchor = einops.rearrange(
            F.linear(
                anchor,
                weight=self.qkv.weight[: self.dim],
                bias=None if self.qkv.bias is None else self.qkv.bias[: self.dim],
            ),
            "bs seqlen (num_heads head_dim) -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        q = torch.concat([q, anchor], dim=2)
        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs[:, :num_anchor_tokens])
        x = F.scaled_dot_product_attention(q, k, v)
        # q is anchor+query
        # k is anchor
        # v is anchor
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        # x is anchor+query

        #x = self.proj(x)
        x1 = self.proj(x)
        # x1 is anchor+query

        # local attention
        x2 = torch.zeros_like(x1)
        # x2 is anchor+query
        for ii in range(x2.shape[0]):
            x2[ii] = (xn - x1)[ii, order[ii], :] # serialized
        x2 = self.local_ln_2(self.local_attention(self.local_ln_1(x2)))
        for ii in range(x2.shape[0]):
            x2[ii] = x2[ii, inverse[ii], :]  # deserialized

#        x = self.mlp(self.ln_3(x0 + x1 + self.local_gate * x2)) + x0
        x = self.ln_3(x1 + self.local_gate * x2)

        return x
