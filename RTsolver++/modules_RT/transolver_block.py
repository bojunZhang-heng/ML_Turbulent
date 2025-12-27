import torch
import torch.nn as nn
from modules_RT.physical_attention import Physics_Attention_Irregular_Mesh
from modules_RT.mlp import MLP
from typing import Any

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            head_num: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(dim=hidden_dim, head_num=head_num, head_dim=hidden_dim // head_num ,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(mlp_input=hidden_dim,
                       mlp_hidden=int(hidden_dim * mlp_ratio),
                       mlp_output=hidden_dim,
                       layer_num=0, act=act, res=False)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: torch.Tensor, attn_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        fx = self.Attn(self.ln_1(fx), **(attn_kwargs or {})) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


