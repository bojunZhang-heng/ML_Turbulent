import logging
import einops
import torch
import torch.nn.functional as F

from .dot_product_attention import DotProductAttention
from modules.rope import rope


class AnchorAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
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
        if num_anchor_tokens is None:
            return super().forward(x=x, freqs=freqs)
        else:
            # x1 is anchor
            # anchor is query
            #x1, _ = x.split([num_anchor_tokens, x.size(1) - num_anchor_tokens], dim=1)
            anchor, _ = x.split([num_anchor_tokens, x.size(1) - num_anchor_tokens], dim=1)

        # B N C
        B, N, C = x.shape
        B_anchor, N_anchor, C_anchor = anchor.shape

        # head_dim = 64 C
        # heads = 3 H
        # slice_num = 64 G
        x = einops.rearrange(
            self.rope_x(x),
            "bs seqlen (num_heads head_dim) -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        anchor = einops.rearrange(
                self.rope_x(anchor),
                "bs seqlen (num_heads head_dim) -> bs num_heads seqlen head_dim",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
        )
        x = rope(x, freqs=freqs)
        anchor = rope(anchor, freqs=freqs[:, :num_anchor_tokens])
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        anchor = einops.rearrange(anchor, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")

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

        ### (1) anchor Slice
        fx_mid_anchor = self.in_project_fx(anchor).reshape(B_anchor, N_anchor, self.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid_anchor = self.in_project_x(anchor).reshape(B_anchor, N_anchor, self.num_heads, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights_anchor = self.softmax(self.in_project_slice(x_mid_anchor) / self.temperature)  # B H N G
        slice_norm_anchor = slice_weights_anchor.sum(2)  # B H G
        slice_token_anchor = torch.einsum("bhnc,bhng->bhgc", fx_mid_anchor, slice_weights_anchor)
        slice_token_anchor = slice_token_anchor / ((slice_norm_anchor + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim)) # B H G C


        ### (2) Attention among anchor+query slice tokens
        q_slice_token = self.to_q(slice_token) # B H G C

        ### (2) Attention among anchor slice tokens
        k_slice_token_anchor = self.to_k(slice_token_anchor)
        v_slice_token_anchor = self.to_v(slice_token_anchor)


        dots = torch.matmul(q_slice_token, k_slice_token_anchor.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token_anchor)  # B H G C


        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights_deslice)
        out_x = einops.rearrange(out_x, 'b h n c -> b n (h c)')

        x = self.proj(x)
        return x
