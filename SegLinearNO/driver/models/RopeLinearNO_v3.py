from typing import Any

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.weight_init import trunc_normal_
from torch.utils.checkpoint import checkpoint


class PositiveRelativePositionKernel(nn.Module):
    """Learnable, non-negative and translation-invariant Fourier kernel.

    For every attention head the kernel is

        kappa(p_i, p_j) = (R + sum_r cos(w_r^T (p_i - p_j))) / (2R),

    which is always in [0, 1]. Coordinates are made dimensionless with the
    per-sample bounding-box diagonal before the phases are evaluated.
    """

    def __init__(self, heads: int, num_frequencies: int = 6, space_dim: int = 3):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be at least 1")
        if space_dim != 3:
            raise ValueError("The current relative-position kernel expects XYZ coordinates")

        self.heads = heads
        self.num_frequencies = num_frequencies
        self.space_dim = space_dim

        # Deterministic multi-scale initialization. The frequencies remain
        # learnable and can rotate away from the coordinate axes during training.
        frequencies = torch.zeros(heads, num_frequencies, space_dim)
        for head in range(heads):
            for index in range(num_frequencies):
                axis = (head + index) % space_dim
                band = index // space_dim
                frequencies[head, index, axis] = 2.0 * math.pi * (2.0**band)
        self.frequencies = nn.Parameter(frequencies)

    @staticmethod
    def normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
        coord_min = coords.amin(dim=-2, keepdim=True)
        coord_max = coords.amax(dim=-2, keepdim=True)
        center = 0.5 * (coord_min + coord_max)
        length = torch.linalg.vector_norm(coord_max - coord_min, dim=-1, keepdim=True)
        length = length.clamp_min(torch.finfo(coords.dtype).eps)
        return (coords - center) / length

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != self.space_dim:
            raise ValueError(
                f"Expected position coordinates shaped (B, N, {self.space_dim}), "
                f"got {tuple(coords.shape)}"
            )

        # Trigonometric functions are deliberately evaluated in fp32 even when
        # the rest of the network is under mixed precision.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            normalized = self.normalize_coordinates(coords.float())
            phases = torch.einsum(
                "bnc,hrc->bhnr", normalized, self.frequencies.float()
            )
        return phases


class LinearNO(nn.Module):
    """Ordinary LinearNO plus a normalized positive relative-position branch."""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        key_ratio=4,
        num_position_frequencies=6,
        initial_relative_gate=0.05,
        eps=1e-6,
    ):
        super().__init__()
        if not 0.0 < initial_relative_gate < 1.0:
            raise ValueError("initial_relative_gate must be between 0 and 1")

        inner_dim = dim_head * heads
        key_dim = key_ratio * dim_head
        self.dim_head = dim_head
        self.key_dim = key_dim
        self.heads = heads
        self.num_position_frequencies = num_position_frequencies
        self.eps = eps

        self.tempreature_q = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.tempreature_k = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim_head, key_dim, bias=False)
        self.to_k = nn.Linear(dim_head, key_dim, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        gate_logit = math.log(initial_relative_gate / (1.0 - initial_relative_gate))
        self.relative_gate_logit = nn.Parameter(
            torch.full((1, heads, 1, 1), gate_logit)
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _content_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        return torch.einsum("bhnd,bhdc->bhnc", q, kv)

    def _positive_relative_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phases: torch.Tensor,
        base_out: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate content attention multiplied by the Fourier kernel.

        The Fourier sum is evaluated term by term. This avoids materializing
        either an N x N attention matrix or an N x key_dim x position_dim
        tensor, so memory remains linear in the number of points.
        """

        if phases.shape != (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            self.num_position_frequencies,
        ):
            raise ValueError(
                "Unexpected relative-position phase shape: "
                f"expected {(q.shape[0], q.shape[1], q.shape[2], self.num_position_frequencies)}, "
                f"got {tuple(phases.shape)}"
            )

        phases = phases.to(dtype=q.dtype)
        coefficient = 0.5 / self.num_position_frequencies

        # The constant R term in the kernel contributes 0.5 times ordinary
        # content attention. Ordinary LinearNO has unit row mass, hence 0.5 in
        # the corresponding denominator.
        numerator = 0.5 * base_out
        denominator = torch.full_like(q[..., 0], 0.5)

        for index in range(self.num_position_frequencies):
            phase = phases[..., index]
            for position_feature in (torch.cos(phase), torch.sin(phase)):
                weighted_k = k * position_feature.unsqueeze(-1)
                weighted_kv = torch.einsum("bhnd,bhnc->bhdc", weighted_k, v)
                contribution = torch.einsum("bhnd,bhdc->bhnc", q, weighted_kv)
                numerator = numerator + coefficient * (
                    position_feature.unsqueeze(-1) * contribution
                )

                key_mass = weighted_k.sum(dim=-2)
                mass_contribution = torch.einsum("bhnd,bhd->bhn", q, key_mass)
                denominator = denominator + coefficient * (
                    position_feature * mass_contribution
                )

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

        # This branch is exactly the ordinary LinearNO normalization.
        q = F.softmax(
            q / torch.clamp(self.tempreature_q, max=2.0, min=0.1), dim=-1
        )
        k = F.softmax(
            k / torch.clamp(self.tempreature_k, max=2.0, min=0.1), dim=-2
        )
        base_out = self._content_attention(q, k, v)
        relative_out = self._positive_relative_attention(
            q, k, v, position_phases, base_out
        )

        # Sigmoid makes this a convex interpolation. A small non-zero initial
        # value lets the position frequencies receive gradients immediately.
        gate = torch.sigmoid(self.relative_gate_logit).to(dtype=base_out.dtype)
        output = base_out + gate * (relative_out - base_out)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


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
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
            for _ in range(n_layers)
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class LinearNOBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=4,
        key_ratio=4,
        last_layer=False,
        out_dim=1,
        num_position_frequencies=6,
        initial_relative_gate=0.05,
        **_,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = LinearNO(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            key_ratio=key_ratio,
            num_position_frequencies=num_position_frequencies,
            initial_relative_gate=initial_relative_gate,
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
    """LinearNO with a gated, normalized positive relative-position kernel."""

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
        num_position_frequencies=6,
        initial_relative_gate=0.05,
    ):
        super().__init__()
        if space_dim != 3:
            raise ValueError(f"RopeLinearNO_v3 expects space_dim=3, got {space_dim}")
        if fun_dim != 0:
            raise ValueError("RopeLinearNO_v3 is coordinate-only, so fun_dim must be 0")
        if unified_pos or Time_Input:
            raise ValueError("unified_pos and Time_Input are not supported")

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.checkpoint_segment_size = 2
        self.position_kernel = PositiveRelativePositionKernel(
            heads=n_head,
            num_frequencies=num_position_frequencies,
            space_dim=space_dim,
        )

        # Deliberately identical to the ordinary coordinate-only LinearNO stem.
        self.preprocess = MLP(
            space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.blocks = nn.ModuleList(
            LinearNOBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                key_ratio=key_ratio,
                out_dim=out_dim,
                num_position_frequencies=num_position_frequencies,
                initial_relative_gate=initial_relative_gate,
                last_layer=layer_index == n_layers - 1,
                H=H,
                W=W,
                isregular=isregular,
            )
            for layer_index in range(n_layers)
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)

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

    def forward(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not torch.is_tensor(data) or data.ndim != 3:
            raise ValueError("Input must be a tensor shaped (B, N, 3)")
        if data.shape[-1] != self.space_dim:
            raise ValueError(f"Expected {self.space_dim} input channels")

        coords = data
        position_coords = coords if rope_coords is None else rope_coords
        if position_coords.shape != coords.shape:
            raise ValueError("rope_coords must have the same (B, N, 3) shape as data")

        position_phases = self.position_kernel(position_coords)
        attn_kwargs = {"position_phases": position_phases}

        # Keep the ordinary LinearNO encoder; relative position only modifies
        # the attention kernel.
        fx = self.preprocess(coords)
        fx = fx + self.placeholder.to(dtype=fx.dtype)[None, None, :]

        if self.training:
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
        return self._forward_blocks(fx, attn_kwargs)

    def _forward_block_range(self, fx, position_phases, start, end):
        attn_kwargs = {"position_phases": position_phases}
        for block_index in range(start, end):
            fx = self.blocks[block_index](fx, attn_kwargs=attn_kwargs)
        return fx

    def _forward_blocks(self, fx, attn_kwargs):
        for block in self.blocks:
            fx = block(fx, attn_kwargs=attn_kwargs)
        return fx

    def relative_gate_values(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.sigmoid(block.attn.relative_gate_logit).flatten()
                for block in self.blocks
            ]
        )
