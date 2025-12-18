import torch
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from modules.rope import rope

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, head_num=8, head_dim=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = head_dim * head_num
        self.head_dim = head_dim
        self.head_num = head_num
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, head_num, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(head_dim, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(head_dim, head_dim, bias=False)
        self.to_k = nn.Linear(head_dim, head_dim, bias=False)
        self.to_v = nn.Linear(head_dim, head_dim, bias=False)
        self.rope_x = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # B N C
        B, N, C = x.shape

        # add rope
        x = rearrange(
            self.rope_x(x),
            "bs seqlen (head_num head_dim) -> bs head_num seqlen head_dim",
            head_num=self.head_num,
            head_dim=self.head_dim,
        )
        x_for_v = x
        x = rope(x, freqs=freqs)
        x = rearrange(x, "bs head_num seqlen head_dim -> bs seqlen (head_num head_dim)")
        x_for_v = rearrange(x_for_v, "bs head_num seqlen head_dim -> bs seqlen (head_num head_dim)")

        ### (1) Slice for q k
        fx_mid = self.in_project_fx(x).reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim))

        ### (1) Slice for v
        fx_mid_v = self.in_project_fx(x_for_v).reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid_v = self.in_project_x(x_for_v).reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights_v = self.softmax(self.in_project_slice(x_mid_v) / self.temperature)  # B H N G
        slice_norm_v = slice_weights_v.sum(2)  # B H G
        slice_token_v = torch.einsum("bhnc,bhng->bhgc", fx_mid_v, slice_weights_v)
        slice_token_v = slice_token / ((slice_norm_v + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token_v)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

