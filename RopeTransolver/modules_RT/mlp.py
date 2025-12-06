import torch.nn as nn
ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    def __init__(self, mlp_input, mlp_hidden, mlp_output, layer_num=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = mlp_input
        self.n_hidden = mlp_hidden
        self.n_output = mlp_output
        self.layer_num = layer_num
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(self.n_input, self.n_hidden), act())
        self.linear_post = nn.Linear(self.n_hidden, self.n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), act()) for _ in range(layer_num)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.layer_num):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

