import torch
from torch import nn

class MLP_geometry(nn.Module):
    def __init__(self, args):
        super(MLP_geometry, self).__init__()

        self.n_layers = args.n_layers
        self.res = args.res
        self.linear_pre = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.GELU())
        self.linear_post = nn.Linear(args.n_hidden, args.n_hidden)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.GELU()) for _ in range(args.n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

