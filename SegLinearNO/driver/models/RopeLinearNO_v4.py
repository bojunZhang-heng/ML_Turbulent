"""Pure relative-position-kernel Linear Neural Operator.

Unlike RopeLinearNO_v3, this model has no ordinary LinearNO output branch and
no learnable gate between content-only and relative-position attention. Every
attention block is driven directly by a learnable, non-negative Fourier kernel
of the relative coordinates.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.weight_init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from .RopeLinearNO_v3 import MLP, PositiveRelativePositionKernel


class PureRopeLinearAttention(nn.Module):
    """Linear attention whose only output is the relative-kernel attention.

    For a non-negative Fourier kernel ``kappa`` the output is

        sum_j [(q_i^T k_j) kappa(p_i-p_j) v_j]
        -------------------------------------------------- .
             sum_j [(q_i^T k_j) kappa(p_i-p_j)]

    The separable sine/cosine expansion evaluates this expression without an
    explicit N x N attention matrix.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        key_ratio: int = 4,
        num_position_frequencies: int = 6,
        checkpoint_relative_terms: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if key_ratio < 1:
            raise ValueError("key_ratio must be at least 1")
        if num_position_frequencies < 1:
            raise ValueError("num_position_frequencies must be at least 1")

        inner_dim = dim_head * heads
        key_dim = key_ratio * dim_head
        self.dim_head = dim_head
        self.key_dim = key_dim
        self.heads = heads
        self.num_position_frequencies = num_position_frequencies
        self.checkpoint_relative_terms = checkpoint_relative_terms
        self.eps = eps

        self.tempreature_q = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.tempreature_k = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
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

    @staticmethod
    def _constant_kernel_term(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute the constant 1/2 term required by the positive kernel."""
        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        return 0.5 * torch.einsum("bhnd,bhdc->bhnc", q, kv)

    @staticmethod
    def _relative_phase_term(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phase: torch.Tensor,
        *,
        use_sine: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one exact sine/cosine low-rank kernel term."""
        position_feature = torch.sin(phase) if use_sine else torch.cos(phase)
        weighted_k = k * position_feature.unsqueeze(-1)
        weighted_kv = torch.einsum("bhnd,bhnc->bhdc", weighted_k, v)
        query_contribution = torch.einsum("bhnd,bhdc->bhnc", q, weighted_kv)
        numerator_contribution = position_feature.unsqueeze(-1) * query_contribution

        key_mass = weighted_k.sum(dim=-2)
        mass_contribution = torch.einsum("bhnd,bhd->bhn", q, key_mass)
        denominator_contribution = position_feature * mass_contribution
        return numerator_contribution, denominator_contribution

    def _pure_relative_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            self.num_position_frequencies,
        )
        if phases.shape != expected_shape:
            raise ValueError(
                "Unexpected relative-position phase shape: "
                f"expected {expected_shape}, got {tuple(phases.shape)}"
            )

        phases = phases.to(dtype=q.dtype)
        coefficient = 0.5 / self.num_position_frequencies

        # The 1/2 constant is part of kappa itself. It guarantees
        # 0 <= kappa <= 1; it is not a parallel ordinary-LinearNO branch.
        numerator = self._constant_kernel_term(q, k, v)
        denominator = torch.full_like(q[..., 0], 0.5)

        for frequency_index in range(self.num_position_frequencies):
            phase = phases[..., frequency_index]
            for use_sine in (False, True):
                if (
                    self.checkpoint_relative_terms
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    def run_term(q_, k_, v_, phase_, use_sine=use_sine):
                        return self._relative_phase_term(
                            q_, k_, v_, phase_, use_sine=use_sine
                        )

                    numerator_term, denominator_term = checkpoint(
                        run_term,
                        q,
                        k,
                        v,
                        phase,
                        use_reentrant=False,
                    )
                else:
                    numerator_term, denominator_term = self._relative_phase_term(
                        q, k, v, phase, use_sine=use_sine
                    )

                numerator = numerator + coefficient * numerator_term
                denominator = denominator + coefficient * denominator_term

        return numerator / denominator.clamp_min(self.eps).unsqueeze(-1)

    def forward(self, x: torch.Tensor, position_phases: torch.Tensor) -> torch.Tensor:
        x_mid = rearrange(
            self.in_project_x(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
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

        output = self._pure_relative_attention(q, k, v, position_phases)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


class PureRopeLinearNOBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 4,
        key_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        num_position_frequencies: int = 6,
        checkpoint_relative_terms: bool = True,
        **_,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PureRopeLinearAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            key_ratio=key_ratio,
            num_position_frequencies=num_position_frequencies,
            checkpoint_relative_terms=checkpoint_relative_terms,
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
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, fx: torch.Tensor, attn_kwargs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx), **(attn_kwargs or {})) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class LinearAttentionNeuralOperator(nn.Module):
    """Coordinate-only neural operator using pure relative-kernel attention."""

    def __init__(
        self,
        space_dim: int = 3,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        Time_Input: bool = False,
        act: str = "gelu",
        mlp_ratio: float = 1,
        fun_dim: int = 0,
        out_dim: int = 1,
        key_ratio: int = 4,
        ref: int = 8,
        unified_pos: bool = False,
        H: int = 85,
        W: int = 85,
        isregular: bool = False,
        num_position_frequencies: int = 6,
        checkpoint_segment_size: int = 1,
        checkpoint_relative_terms: bool = True,
    ):
        super().__init__()
        if space_dim != 3:
            raise ValueError(f"RopeLinearNO_v4 expects space_dim=3, got {space_dim}")
        if fun_dim != 0:
            raise ValueError("RopeLinearNO_v4 is coordinate-only, so fun_dim must be 0")
        if unified_pos or Time_Input:
            raise ValueError("unified_pos and Time_Input are not supported")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if checkpoint_segment_size < 1:
            raise ValueError("checkpoint_segment_size must be at least 1")

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.checkpoint_segment_size = checkpoint_segment_size
        self.position_kernel = PositiveRelativePositionKernel(
            heads=n_head,
            num_frequencies=num_position_frequencies,
            space_dim=space_dim,
        )

        # Normalized coordinates encode node content; raw coordinates are used
        # separately by position_kernel after per-sample nondimensionalization.
        self.preprocess = MLP(
            space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.blocks = nn.ModuleList(
            PureRopeLinearNOBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                key_ratio=key_ratio,
                out_dim=out_dim,
                num_position_frequencies=num_position_frequencies,
                checkpoint_relative_terms=checkpoint_relative_terms,
                last_layer=layer_index == n_layers - 1,
                H=H,
                W=W,
                isregular=isregular,
            )
            for layer_index in range(n_layers)
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)

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

    def forward(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not torch.is_tensor(data) or data.ndim != 3:
            raise ValueError("Input must be a tensor shaped (B, N, 3)")
        if data.shape[-1] != self.space_dim:
            raise ValueError(f"Expected {self.space_dim} input channels")

        position_coords = data if rope_coords is None else rope_coords
        if position_coords.shape != data.shape:
            raise ValueError("rope_coords must have the same (B, N, 3) shape as data")

        position_phases = self.position_kernel(position_coords)
        fx = self.preprocess(data)
        fx = fx + self.placeholder.to(dtype=fx.dtype)[None, None, :]

        if self.training and torch.is_grad_enabled():
            for start in range(0, len(self.blocks), self.checkpoint_segment_size):
                end = min(start + self.checkpoint_segment_size, len(self.blocks))

                def run_segment(segment_fx, phases, start=start, end=end):
                    return self._forward_block_range(segment_fx, phases, start, end)

                fx = checkpoint(
                    run_segment,
                    fx,
                    position_phases,
                    use_reentrant=False,
                )
            return fx

        return self._forward_blocks(fx, position_phases)

    def _forward_block_range(
        self,
        fx: torch.Tensor,
        position_phases: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        attn_kwargs = {"position_phases": position_phases}
        for block_index in range(start, end):
            fx = self.blocks[block_index](fx, attn_kwargs=attn_kwargs)
        return fx

    def _forward_blocks(
        self, fx: torch.Tensor, position_phases: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_block_range(fx, position_phases, 0, len(self.blocks))


# Keep the model-construction name used by main.py and existing experiment code.
RopeLinearNO_v4 = LinearAttentionNeuralOperator

