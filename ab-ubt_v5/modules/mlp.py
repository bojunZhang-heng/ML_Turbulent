import torch
from torch import nn


class Mlp(nn.Module):
    """MLP as used in transformers nn.Linear(dim, dim * 4) -> GELU -> nn.Linear(dim * 4, dim).

    Args:
        dim: Input dimension of the MLP.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.input = dim
        self.hidden = dim * 2
        self.output = dim
        self.layers = 5
        self.res = False

        self.linear_pre = nn.Sequential(nn.Linear(self.input, self.hidden), nn.GELU())
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.GELU()) for _ in range(self.layers)])
        self.linear_post = nn.Linear(self.hidden, self.output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)

        for i in range(self.layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)

        x = self.linear_post(x)

        return x




