import torch
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from modules.rope import rope

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, head_num=8, head_dim=64, dropout=0., slice_num=64):
        super().__init__()
        # inner_dim   = head_dim * head_num
        self.head_dim = head_dim
        self.head_num = head_num
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, head_num, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(self.head_dim, self.head_dim)
        self.in_project_fx = nn.Linear(self.head_dim, self.head_dim)
        self.in_project_freqs = nn.Linear(self.head_dim, self.head_dim)
       
        self.in_project_slice = nn.Linear(self.head_dim, slice_num)

        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(head_dim, head_dim, bias=False)
        self.to_v = nn.Linear(head_dim, head_dim, bias=False)
        self.rope_x = nn.Linear(dim, dim*3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # complex -> real
        freqs = torch.view_as_real(freqs)
        freqs = freqs.reshape(B, N, 2*freqs.shape[2])

        # share weights among x and freqs
        freqs = freqs.reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()
        #logging.info(f"freqs: {freqs.shape}")

        ### (1) Slice
        fx_mid = self.in_project_fx(x).contiguous()  # B H N C
        freqs_mid = self.in_project_freqs(freqs).contiguous()  # B H N C
        x_mid = self.in_project_x(x).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        
        slice_token_x = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token_x = slice_token_x / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim))
        slice_token_freqs = torch.einsum("bhnc,bhng->bhgc", freqs_mid, slice_weights)
        slice_token_freqs = slice_token_freqs / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token_x)
        k_slice_token = self.to_k(slice_token_x)
        v_slice_token = self.to_v(slice_token_x)

        q_slice_token = rope(q_slice_token, freqs=slice_token_freqs)
        k_slice_token = rope(k_slice_token, freqs=slice_token_freqs)

        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G C

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)

        x = rearrange(out_x, "bs num_head_num seqlen head_dim -> bs seqlen (num_head_num head_dim)")

        return self.to_out(x)

