import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
            self,
            res,
            n_input,
            n_hidden,
            n_output,
            n_layers
        ):
        super(MLP, self).__init__()

        self

        self.n_layers = n_layers
        self.res = res
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.linear_pre = nn.Sequential(nn.Linear(self.n_input, self.n_hidden), nn.GELU())
        self.linear_post = nn.Linear(self.n_hidden, self.n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.GELU()) for _ in range(self.n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x
