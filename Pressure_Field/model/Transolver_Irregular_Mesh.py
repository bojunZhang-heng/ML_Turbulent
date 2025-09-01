import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.Embedding import timestep_embedding
import numpy as np
import logging
from model.Physics_Attention import Physics_Attention_Irregular_Mesh
from colorama import Fore, Style

#! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def knn(x, k):
    """
    k-nearest neighbors algorithm.

    Args:
        x: Input tensor of shape (batch_size, num_points, feature_dim)
        k: Number of neighbors to consider

    Returns:
        Indices of k-nearest neighbors for each point
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))             # [batch_size, num_points, num_points]
    xx = torch.sum(x ** 2, dim=2, keepdim=True)                 # [batch_size, num_points, 1]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)        # [batch_size, num_points, num_points]

    idx = pairwise_distance.topk(k=k, dim=-1)[1]                # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Construct edge features for graph convolution.

    When you create this function first, Just use 1 batch_size, 2 dims, 2 points
    to Understand this code

    Then Enlarge it meeting the physical need

    Args:
        x: Input tensor of shape (batch_size, num_points, feature_dim)
        k: Number of neighbors to use for graph construction
        idx: Optional pre-computed nearest neighbor indices
        dim9: Whether to use additional dimensional features

    Returns:
        Edge features for graph convolution
    """
    batch_size = x.size(0)
    num_points = x.size(1)
    x = x.view(batch_size, num_points, -1)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points    # idx_base.shape = [2, 1, 1]
    idx = idx + idx_base
    idx = idx.view(-1)                                                                   # [batch_size * num_points * k]

    _, _ , point_dims= x.size()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, point_dims)                        # [batch_size, num_points, k, point_dims]
    x = x.view(batch_size, num_points, 1, point_dims).repeat(1, 1, k, 1)                 # [batch_size, num_points, k, point_dims]

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()        # (batch_size, p_dim*2, num_points, k)
    return feature

class Transform_Net(nn.Module):
    #def __init__(self, args):
    def __init__(self):
        super(Transform_Net, self).__init__()
        #self.args = args
        ks=3
       # self.k = 3


        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=ks, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=ks, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=ks, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn5(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class MLP(nn.Module):
           # self.preprocess = MLP(space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True, k=40):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden                    # n_hidden = 256
        self.n_output = n_output                    # n_output = 128
        self.n_layers = n_layers
        self.res = res
        self.transform_net = Transform_Net()
        self.k = k
        ks = 3
        dropout = 0.4
       # self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
       # self.linear_post = nn.Linear(n_hidden, n_output)
       # self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
        self.bn1 = nn.BatchNorm2d(n_hidden)
        self.bn2 = nn.BatchNorm2d(n_hidden)
        self.bn3 = nn.BatchNorm2d(n_hidden)
        self.bn4 = nn.BatchNorm2d(n_hidden)
        self.bn5 = nn.BatchNorm2d(n_hidden)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(n_hidden)
        self.bn9 = nn.BatchNorm1d(n_hidden)
        self.bn10 = nn.BatchNorm1d(n_output)


        self.conv1 = nn.Sequential(nn.Conv2d(n_input, n_hidden, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(n_hidden*2, n_hidden, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(n_hidden*2, n_hidden, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(n_hidden*3, 1024, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(n_hidden*3, 1024, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(n_hidden*3+1024, n_hidden, kernel_size=ks, stride=1, padding=ks // 2, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))

        logging.info(f"{M} n_hidden: {n_hidden} {RESET}")
        logging.info(f"{M} n_output: {n_output} {RESET}")
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)

        x0 = get_graph_feature(x, k=self.k)   # (batch_size, num_points, p_dim) -> (batch_size, p_dim+d_dim, num_points, k)
        t = self.transform_net(x0)            # (batch_size, 3, 3)
        x = torch.bmm(x, t)                   # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)

        x = get_graph_feature(x, k=self.k)    # (batch_size, num_points, 3) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                     # (batch_size, p_dim+d_dim, num_points, k) -> (batch_size, n_hidden, num_points, k)
        x = self.conv2(x)                     # (batch_size, n_hidden, num_points, k) -> (batch_size, n_hidden, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]   # (batch_size, n_hidden, num_points, k) -> (batch_size, n_hidden, num_points)

        x1 = x1.transpose(2, 1).contiguous()                  # (batch_size, n_hidden, num_points) -> (batch_size, num_points, n_hidden)
        x = get_graph_feature(x1, k=self.k)    # (batch_size, num_points, n_hidden) -> (batch_size, n_hidden*2, num_points, k)
        x = self.conv3(x)                     # (batch_size, n_hidden*2, num_points, k) -> (batch_size, n_hidden, num_points, k)
        x = self.conv4(x)                     # (batch_size, n_hidden, num_points, k) -> (batch_size, n_hidden, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, n_hidden, num_points, k) -> (batch_size, n_hidden, num_points)

        x2 = x2.transpose(2, 1).contiguous()      # (batch_size, n_hidden, num_points) -> (batch_size, num_points, n_hidden)
        x = get_graph_feature(x2, k=self.k)    # (batch_size, num_points, n_hidden) -> (batch_size, n_hidden*2, num_points, k)
        x = self.conv5(x)                     # (batch_size, n_hidden*2, num_points, k) -> (batch_size, n_hidden, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, n_hidden, num_points, k) -> (batch_size, n_hidden, num_points)

        x1 = x1.transpose(2, 1).contiguous()      # (batch_size, num_points, n_hidden) -> (batch_size, n_hidden, num_points)
        x2 = x2.transpose(2, 1).contiguous()      # (batch_size, num_points, n_hidden) -> (batch_size, n_hidden, num_points)
        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, n_hidden*3, num_points)

        x = self.conv6(x)                   # (batch_size, n_hidden*3, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)

     #   logging.info(f"{M} x.shape: {x.shape} {RESET}")
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+n_hidden*3, num_points)

        x = self.conv8(x)  # (batch_size, 1024+n_hidden*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)

        x = x.permute(0, 2, 1).contiguous()   # (batch_size, n_output, num_points) -> (batch_size, num_points, n_output)

        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
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
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
      #  self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx     #[B, num_points, 128] -> [B, num_points, 128] -> [B, num_points, 128]
       # fx = self.mlp(self.ln_2(fx)) + fx      #[B, num_points, 128] -> [B, num_points, 128] -> [B, num_points, 128]
        fx = self.ln_2(fx) + fx                #[B, num_points, 128] -> [B, num_points, 128]
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=128,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 k=40,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'Transolver_1D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.k = k
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act, k=self.k)
        else:
            self.preprocess = MLP(6, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act, k=self.k)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()

        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        # x: B N 2
        # grid_ref
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda().reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, T=None):
        x = self.preprocess(x)                                   # [B, num_points, point_dim]   -> [B, num_points, 128]
        x = x + self.placeholder[None, None, :]

        for block in self.blocks:
            x = block(x)

        return x
