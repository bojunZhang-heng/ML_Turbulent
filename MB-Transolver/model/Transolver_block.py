import torch
import torch.nn as nn
from model.Physics_Attention import Physics_Attention
from model.MLP_geometry import MLP_geometry

class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(self, args, last_layer):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(args.n_hidden)                  # C = 128
        self.Attn = Physics_Attention(args)                      # C = 4*64 = 256
        self.ln_2 = nn.LayerNorm(args.n_hidden)
        self.mlp = MLP_geometry(args)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(args.n_hidden)
            self.mlp2 = nn.Linear(args.n_hidden, args.n_dim)

    def forward(self, x):
        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        if self.last_layer:
            return self.mlp2(self.ln_3(x))
        else:
            return x

