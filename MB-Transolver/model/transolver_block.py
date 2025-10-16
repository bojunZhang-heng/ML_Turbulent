import torch
import torch.nn as nn
from model.physics_attention import PhysicsAttention
from typing import Any
from model.MLP import MLP

class TransolverBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, args, attn_ctor: type[nn.Module]=PhysicsAttention):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(args.n_hidden)                      # C = 128
        self.Attn = attn_ctor(args)                                  # C = 4*64 = 256
        self.ln_2 = nn.LayerNorm(args.n_hidden)
        self.mlp = MLP(n_input=args.n_hidden, n_hidden=args.n_hidden, n_output=args.n_hidden, n_layers=3, res=args.res)

    def forward(
            self,
            x:torch.Tensor, attn_kwargs: dict[str, Any] | None = None
        ) -> torch.Tensor :
        """Forward pass of the transformer block.

        Args:
            x: Input tensor with shape (batch_size, seqlen/num_tokens, dim).
            attn_kwargs: Dict with arguments for the attention (such as the rope frequencies). Defaults to None.

        Returns: the size is not determined
            (batch_size, num_tokens, dim)
        """
        x = x + self.Attn(self.ln_1(x), **(attn_kwargs or {}))
        x = x + self.mlp(self.ln_2(x))

        return x

