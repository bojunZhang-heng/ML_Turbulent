import torch
import torch.nn as nn
import logging
from timm.models.layers import trunc_normal_

from modules_RT.continuous_sincos_embed import ContinuousSincosEmbed
from modules_RT.rope_frequency import RopeFrequency
from modules_RT.mlp import MLP
from modules_RT.transolver_block import Transolver_block

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class Model(nn.Module):
    def __init__(self,
                 space_dim=3,
                 layer_num=5,
                 hidden_dim=256,
                 dropout=0,
                 head_num=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=0,
                 out_dim=1,
                 slice_num=32,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.preprocess = MLP(mlp_input=hidden_dim,
                              mlp_hidden=hidden_dim * 2,
                              mlp_output=hidden_dim,
                              layer_num=0, act=act, res=False)

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.space_dim = space_dim
        self.rope = RopeFrequency(dim=hidden_dim, ndim=space_dim)

        # pos_embed with MLP for volume
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=space_dim)
        self.volume_bias = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([Transolver_block(head_num=head_num, hidden_dim=hidden_dim,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == layer_num - 1))
                                     for _ in range(layer_num)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (hidden_dim)) * torch.rand(hidden_dim, dtype=torch.float))

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

    def forward(self, x):
        volume_decoder_attn_kwargs = {}

        # rope frequencies batch size only for 1
        volume_rope = self.rope(x)
        volume_decoder_attn_kwargs["freqs"] = volume_rope

        fx = self.pos_embed(x)
        fx = self.preprocess(fx)

        #fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]


        for block in self.blocks:
            fx = block(fx,
                       attn_kwargs=dict(**volume_decoder_attn_kwargs)
                       )

        return fx
