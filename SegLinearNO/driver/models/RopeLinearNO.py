from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.weight_init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from models.continuous_sincos_embed import ContinuousSincosEmbed
from models.rope import rope
from models.rope_frequency import RopeFrequency


class LinearNO(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, key_ratio=4):
        super().__init__()
        inner_dim = dim_head * heads
        key_dim = key_ratio * dim_head
        if key_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even key dimension, got {key_dim}")

        self.dim_head = dim_head
        self.key_dim = key_dim
        self.heads = heads
        self.tempreature_q = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.tempreature_k = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim_head, key_dim, bias=False)
        self.to_k = nn.Linear(dim_head, key_dim, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x_mid = rearrange(
            self.in_project_x(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
        )
        freqs = rearrange(
            freqs,
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.key_dim // 2,
        )

        q = self.to_q(x_mid)
        k = self.to_k(x_mid)
        v = self.to_v(x_mid)

        q = F.softmax(
            q / torch.clamp(self.tempreature_q, max=2.0, min=0.1), dim=-1
        )
        k = F.softmax(
            k / torch.clamp(self.tempreature_k, max=2.0, min=0.1), dim=-2
        )
        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs)

        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        qkv = torch.einsum("bhnd,bhdc->bhnc", q, kv)
        qkv = rearrange(qkv, "b h n d -> b n (h d)")
        return self.to_out(qkv)


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
    def __init__(
        self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True
    ):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Unsupported activation: {act}")

        activation = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class LinearNO_block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        key_ratio=4,
        last_layer=False,
        out_dim=1,
        **_,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = LinearNO(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            key_ratio=key_ratio,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            int(hidden_dim * mlp_ratio),
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, fx: torch.Tensor, attn_kwargs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        fx = self.Attn(self.ln_1(fx), **(attn_kwargs or {})) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class LinearAttentionNeuralOperator(nn.Module):
    """LinearNO with continuous 3D RoPE.

    The model uses only XYZ coordinates. Inputs shaped ``(B, N, 3)`` are used
    directly. Existing ``(B, N, 6)`` coordinate-plus-normal inputs remain
    compatible, but only their first three channels are used.
    """

    def __init__(
        self,
        space_dim=3,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=1,
        key_ratio=4,
        ref=8,
        unified_pos=False,
        H=85,
        W=85,
        isregular=False,
    ):
        super().__init__()
        if space_dim != 3:
            raise ValueError(f"RopeLinearNO expects space_dim=3, got {space_dim}")
        if fun_dim != 0:
            raise ValueError("RopeLinearNO is coordinate-only, so fun_dim must be 0")
        if unified_pos:
            raise ValueError("unified_pos is not supported for irregular 3D point clouds")
        if Time_Input:
            raise ValueError("Time_Input is not supported by the coordinate-only interface")
        if n_hidden % n_head != 0:
            raise ValueError(
                f"n_hidden ({n_hidden}) must be divisible by n_head ({n_head})"
            )

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        rotary_dim = key_ratio * n_hidden
        self.rope = RopeFrequency(dim=rotary_dim, ndim=space_dim)
        self.pos_embed = ContinuousSincosEmbed(dim=n_hidden, ndim=space_dim)
        self.preprocess = MLP(
            n_hidden,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.blocks = nn.ModuleList(
            [
                LinearNO_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    key_ratio=key_ratio,
                    H=H,
                    W=W,
                    isregular=isregular,
                    last_layer=layer_idx == n_layers - 1,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            torch.rand(n_hidden, dtype=torch.float) / n_hidden
        )

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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(data) or data.ndim != 3:
            raise ValueError("Input must be a tensor shaped (batch, points, channels)")
        if data.shape[-1] < self.space_dim:
            raise ValueError(
                f"Input needs at least {self.space_dim} XYZ channels, got {data.shape[-1]}"
            )

        coords = data[..., : self.space_dim]
        attn_kwargs = {"freqs": self.rope(coords)}
        fx = self.preprocess(self.pos_embed(coords))
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            if self.training:
                fx = checkpoint(
                    block,
                    fx,
                    attn_kwargs=attn_kwargs,
                    use_reentrant=False,
                )
            else:
                fx = block(fx, attn_kwargs=attn_kwargs)
        return fx
