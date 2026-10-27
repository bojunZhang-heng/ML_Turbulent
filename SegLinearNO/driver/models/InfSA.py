"""Linear Infinite Self-Attention for irregular automotive surface meshes.

This module adapts Linear-InfSA from "Self-Attention And Beyond the
Infinite" to coordinate-only, per-node CFD regression.  It deliberately does
not implement quadratic Pure InfSA because DrivAerNet surfaces can contain
hundreds of thousands of nodes.
"""

import math

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
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Unsupported activation: {act}")

        activation = ACTIVATION[act]
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linears = nn.ModuleList(
            nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
            for _ in range(n_layers)
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class LinearInfSAAttention(nn.Module):
    """O(N) principal-centrality approximation from Linear-InfSA.

    The query and key projections are tied exactly as required by the paper:
    ``Q = K = X W_Q``.  Each head produces one weighted global context vector
    and broadcasts it to all nodes.  Token-specific information is retained by
    the residual path in :class:`InfSABlock`.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 64,
        dropout: float = 0.0,
        gamma_init: float = 0.7,
        gamma_trainable: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if not 0.0 < gamma_init < 1.0:
            raise ValueError("gamma_init must be strictly between 0 and 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5
        self.eps = eps

        # Q and K intentionally share this single projection.  Separate Q/K
        # projections would violate the symmetric operator used by InfSA's
        # Perron-Frobenius interpretation.
        self.qk_project = nn.Linear(dim, dim, bias=True)
        self.v_project = nn.Linear(dim, dim, bias=True)
        self.out_project = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        gamma_logit = math.log(gamma_init / (1.0 - gamma_init))
        gamma_tensor = torch.full((1, heads, 1), gamma_logit)
        if gamma_trainable:
            self.gamma_logit = nn.Parameter(gamma_tensor)
        else:
            self.register_buffer("gamma_logit", gamma_tensor)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected attention input shaped (B, N, {self.dim}), "
                f"got {tuple(x.shape)}"
            )

        q = rearrange(
            self.qk_project(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
        )
        v = rearrange(
            self.v_project(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
        )

        # Q=K. Scaling is retained to match the implementation used for the
        # paper's eigenvector-alignment study. Both L1 normalizations make the
        # global scale cancel, apart from the numerical epsilon.
        q = q * self.scale

        # Equations (16)-(18): energy-normalized soft central query.
        energy = torch.linalg.vector_norm(q, ord=2, dim=-1)
        alpha = energy / energy.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        q_bar = torch.einsum("bhn,bhnd->bhd", alpha, q)

        # Equations (19)-(20): positive similarity and L1-normalized token
        # centrality. K equals Q by construction.
        scores = F.relu(torch.einsum("bhd,bhnd->bhn", q_bar, q))
        attention = scores / scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # Equation (21). Keep gamma as a Tensor -- using .item() here would
        # silently detach it and make a nominally trainable gamma ineffective.
        gamma = torch.sigmoid(self.gamma_logit).to(dtype=attention.dtype)
        weights = gamma * attention
        context = torch.einsum("bhn,bhnd->bhd", weights, v)

        # Each head is rank one across nodes. Because out_project is pointwise,
        # projecting the single concatenated context before broadcasting is
        # exactly equivalent and avoids repeating the same matrix multiply N
        # times. expand() is a view; the residual addition materializes the
        # required per-node representation in the enclosing block.
        output = rearrange(context, "b h d -> b 1 (h d)")
        output = self.out_project(output)
        output = output.expand(-1, x.shape[1], -1)

        if return_attention:
            return output, attention
        return output

    def gamma_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_logit).detach().flatten()


class InfSABlock(nn.Module):
    """Pre-LN Linear-InfSA block with residual MLP."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 2.0,
        gamma_init: float = 0.7,
        gamma_trainable: bool = True,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = LinearInfSAAttention(
            dim=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            gamma_init=gamma_init,
            gamma_trainable=gamma_trainable,
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

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        return self.mlp(self.ln_2(fx)) + fx


class Model(nn.Module):
    """Coordinate-only Linear-InfSA neural operator for surface regression."""

    def __init__(
        self,
        space_dim: int = 3,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 64,
        act: str = "gelu",
        mlp_ratio: float = 2.0,
        fun_dim: int = 0,
        out_dim: int = 1,
        gamma_init: float = 0.7,
        gamma_trainable: bool = True,
        unified_pos: bool = False,
        **_,
    ):
        super().__init__()
        if space_dim != 3:
            raise ValueError(f"InfSA expects space_dim=3, got {space_dim}")
        if fun_dim != 0:
            raise ValueError("Coordinate-only InfSA requires fun_dim=0")
        if unified_pos:
            raise ValueError("InfSA does not support unified_pos")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.preprocess = MLP(
            space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.blocks = nn.ModuleList(
            InfSABlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                gamma_init=gamma_init,
                gamma_trainable=gamma_trainable,
            )
            for _ in range(n_layers)
        )
        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)
        self.output_norm = nn.LayerNorm(n_hidden)
        self.output_decoder = nn.Linear(n_hidden, out_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError("Input must be a tensor shaped (B, N, channels)")
        if x.shape[-1] < self.space_dim:
            raise ValueError(
                f"Input needs at least {self.space_dim} XYZ channels, "
                f"got {x.shape[-1]}"
            )

        coords = x[..., : self.space_dim]
        fx = self.preprocess(coords)
        fx = fx + self.placeholder.to(dtype=fx.dtype)[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return self.output_decoder(self.output_norm(fx))

    def gamma_values(self) -> torch.Tensor:
        """Return per-layer, per-head decay factors for diagnostics."""
        return torch.stack([block.attn.gamma_values() for block in self.blocks])


LinearAttentionNeuralOperator = Model

