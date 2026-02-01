import torch
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from modules_RT.rope import rope

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
        self.rope_x = nn.Linear(dim, dim*3)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:

        # add rope
        q, k, v = rearrange(
            self.rope_x(x),
            "bs seqlen (three head_num head_dim) -> three bs head_num seqlen head_dim",
            head_num=self.head_num,
            head_dim=self.head_dim,
        ).unbind(0)

 #       logging.info(f"q: {q.shape}")
 #       logging.info(f"freqs: {freqs.shape}")
        freqs = rearrange(
                freqs,
                "bs seqlen (head_num  head_dim) -> bs head_num seqlen head_dim",
                head_num=self.head_num,
                head_dim=self.head_dim // 2,
        )
#        logging.info(f"freqs: {freqs.shape}")
        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")

        return self.to_out(x)

