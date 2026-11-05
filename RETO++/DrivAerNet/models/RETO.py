"""RETO: continuous-coordinate rotary transformer for irregular meshes.

Migrated from RETO/modules_RT/model/model_transolver.py.  The model keeps
continuous coordinate encoding, complex rotary frequencies and full
coordinate-to-coordinate scaled dot-product attention.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=1, act="gelu", res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(act)
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(input_dim, hidden_dim), ACTIVATION[act]())
        self.linears = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), ACTIVATION[act]())
            for _ in range(layer_num)
        )
        self.linear_post = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = x + layer(x) if self.res else layer(x)
        return self.linear_post(x)


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, dim, ndim, max_wavelength=10000.0):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        ndim_padding = dim % ndim
        dim_per_ndim = (dim - ndim_padding) // ndim
        sincos_padding = dim_per_ndim % 2
        self.padding = ndim_padding + sincos_padding * ndim
        effective_dim = (dim - self.padding) // ndim
        if effective_dim <= 0 or effective_dim % 2:
            raise ValueError("dim must provide an even positive frequency dimension per axis")
        omega = torch.arange(0, effective_dim, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (omega / effective_dim))

    def forward(self, coords):
        if coords.shape[-1] != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} coordinate channels, got {coords.shape[-1]}"
            )
        with torch.autocast(device_type=coords.device.type, enabled=False):
            out = coords.float().unsqueeze(-1) @ self.omega.unsqueeze(0)
            embedding = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
            embedding = rearrange(embedding, "... ndim dim -> ... (ndim dim)")
        if self.padding:
            padding = torch.zeros(
                *embedding.shape[:-1], self.padding,
                device=embedding.device, dtype=embedding.dtype
            )
            embedding = torch.cat([embedding, padding], dim=-1)
        return embedding


class RopeFrequency(nn.Module):
    def __init__(self, dim, ndim, max_wavelength=10000.0):
        super().__init__()
        self.ndim = ndim
        ndim_padding = dim % ndim
        dim_per_ndim = (dim - ndim_padding) // ndim
        sincos_padding = dim_per_ndim % 2
        self.padding = ndim_padding + sincos_padding * ndim
        effective_dim = (dim - self.padding) // ndim
        if effective_dim <= 0 or effective_dim % 2:
            raise ValueError("dim must provide an even positive RoPE dimension per axis")
        omega = torch.arange(0, effective_dim, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (omega / effective_dim))

    def forward(self, coords):
        if coords.shape[-1] != self.ndim:
            raise ValueError("Coordinate dimension does not match RoPE dimension")
        with torch.autocast(device_type=coords.device.type, enabled=False):
            phases = coords.float().unsqueeze(-1) @ self.omega.unsqueeze(0)
            phases = rearrange(phases, "... ndim dim -> ... (ndim dim)")
            if self.padding:
                phases = torch.cat(
                    [
                        phases,
                        torch.zeros(
                            *phases.shape[:-1], self.padding // 2,
                            device=phases.device, dtype=phases.dtype
                        ),
                    ],
                    dim=-1,
                )
            return torch.polar(torch.ones_like(phases), phases)


def apply_rope(x, freqs):
    if not torch.is_complex(freqs):
        raise TypeError("RoPE frequencies must be complex")
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(x_complex * freqs).flatten(start_dim=3)
    return rotated.to(dtype=x.dtype)


class RETOAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        if dim % heads != 0 or (dim // heads) % 2:
            raise ValueError("hidden dimension must yield an even head dimension")
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_x = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def forward(self, x, freqs):
        q, k, v = rearrange(
            self.rope_x(x),
            "b n (three h d) -> three b h n d",
            three=3, h=self.heads, d=self.head_dim,
        ).unbind(0)
        freqs = rearrange(
            freqs,
            "b n (h d) -> b h n d",
            h=self.heads, d=self.head_dim // 2,
        )
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class RETOBlock(nn.Module):
    def __init__(self, heads, hidden_dim, dropout, act, mlp_ratio, last_layer, out_dim):
        super().__init__()
        self.last_layer = last_layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RETOAttention(hidden_dim, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, 0, act, False)
        if last_layer:
            self.norm3 = nn.LayerNorm(hidden_dim)
            self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs)
        x = x + self.mlp(self.norm2(x))
        return self.decoder(self.norm3(x)) if self.last_layer else x


class Model(nn.Module):
    """RETO model with coordinates as the continuous positional domain."""

    def __init__(
        self,
        space_dim=3,
        n_layers=8,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=2.0,
        out_dim=1,
        sincos_max_wavelength=10000,
        rope_max_wavelength=10000,
        **_: Any,
    ):
        super().__init__()
        if space_dim < 1:
            raise ValueError("space_dim must be positive")
        if n_layers < 1 or n_hidden % n_head:
            raise ValueError("Invalid layer count or hidden/head dimensions")
        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.pos_embed = ContinuousSincosEmbed(n_hidden, space_dim, sincos_max_wavelength)
        self.preprocess = MLP(n_hidden, n_hidden * 2, n_hidden, 0, act, False)
        self.rope = RopeFrequency(n_hidden, space_dim, rope_max_wavelength)
        self.blocks = nn.ModuleList(
            RETOBlock(
                n_head, n_hidden, dropout, act, mlp_ratio,
                index == n_layers - 1, out_dim,
            )
            for index in range(n_layers)
        )
        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.space_dim:
            raise ValueError(
                f"Expected input shape (B, N, {self.space_dim}), got {tuple(x.shape)}"
            )
        freqs = self.rope(x)
        features = self.preprocess(self.pos_embed(x))
        features = features + self.placeholder.to(dtype=features.dtype)[None, None, :]
        for block in self.blocks:
            features = block(features, freqs)
        return features


LinearAttentionNeuralOperator = Model

