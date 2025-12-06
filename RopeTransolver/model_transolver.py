import torch
import torch.nn as nn
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
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 head_num=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=0,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.preprocess = MLP(mlp_input=space_dim,
                              mlp_hidden=n_hidden * 2,
                              mlp_output=n_hidden,
                              n_layers=0, act=act, res=False
                              )

        self.n_hidden = n_hidden
        self.head_num = head_num
        self.space_dim = space_dim
        self.rope = RopeFrequency(dim=n_hidden // head_num, ndim=space_dim)

        # pos_embed with MLP for volume
        self.pos_embed = ContinuousSincosEmbed(dim=n_hidden, ndim=space_dim)
        self.volume_bias = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.blocks = nn.ModuleList([Transolver_block(head_num=head_num, hidden_dim=n_hidden,
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

    def forward(self, x):
        volume_decoder_attn_kwargs = {}

        # rope frequencies batch size only for 1
        volume_rope = self.rope(x)
        volume_decoder_attn_kwargs["freqs"] = volume_rope

        fx = self.volume_bias(self.pos_embed(x))
       # fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]


        for block in self.blocks:
            fx = block(fx,
                       attn_kwargs=dict(**volume_decoder_attn_kwargs)
                       )

        return fx
