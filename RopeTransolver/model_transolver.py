import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat, rearrange

from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency
from modules.rope import rope
from typing import Any

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, head_num=8, head_dim=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = head_dim * head_num
        self.head_dim = head_dim
        self.head_num = head_num
        self.scale = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, head_num, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(head_dim, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(head_dim, head_dim, bias=False)
        self.to_k = nn.Linear(head_dim, head_dim, bias=False)
        self.to_v = nn.Linear(head_dim, head_dim, bias=False)
        self.rope_x = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # B N C
        B, N, C = x.shape

        # add rope
        x = rearrange(
            self.rope_x(x),
            "bs seqlen (head_num head_dim) -> bs head_num seqlen head_dim",
            head_num=self.head_num,
            head_dim=self.head_dim,
        )
        x = rope(x, freqs=freqs)
        x = rearrange(x, "bs head_num seqlen head_dim -> bs seqlen (head_num head_dim)")

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.head_num, self.head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.head_dim))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, mlp_input, mlp_hidden, mlp_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = mlp_input
        self.n_hidden = mlp_hidden
        self.n_output = mlp_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(self.n_input, self.n_hidden), act())
        self.linear_post = nn.Linear(self.n_hidden, self.n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


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
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, head_num=head_num, head_dim=hidden_dim // head_num ,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(mlp_input=hidden_dim,
                       mlp_hidden=hidden_dim * mlp_ratio,
                       mlp_output=hidden_dim,
                       n_layers=0, act=act, res=False
                       )
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
