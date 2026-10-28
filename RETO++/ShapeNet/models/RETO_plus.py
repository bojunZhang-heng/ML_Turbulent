"""ShapeNet-Car RETO+ with continuous 3D RoPE.

This file intentionally contains the complete model implementation.  It does
not import ClusterAttention or the neural-operator backbone from CANO.py.  The
seven input channels are [x, y, z, sdf, normal_x, normal_y, normal_z], while
the four outputs are [velocity_x, velocity_y, velocity_z, pressure].
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.weight_init import trunc_normal_
from torch.utils.checkpoint import checkpoint


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
    """Pointwise MLP used by the input encoder and RETO+ blocks."""

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
            x = x + layer(x) if self.res else layer(x)
        return self.linear_post(x)


class ContinuousRotaryEmbedding3D(nn.Module):
    """Learnable continuous rotary phases for irregular three-dimensional meshes."""

    def __init__(
        self,
        heads: int,
        rotary_dim: int,
        num_position_frequencies: int = 6,
    ):
        super().__init__()
        if rotary_dim % 2 != 0:
            raise ValueError("The RoPE dimension must be even")
        if num_position_frequencies < 1:
            raise ValueError("num_position_frequencies must be at least 1")

        self.heads = heads
        self.rotary_pairs = rotary_dim // 2
        frequencies = torch.zeros(heads, self.rotary_pairs, 3)
        pairs_per_axis = max(1, math.ceil(self.rotary_pairs / 3))
        max_band = max(0, num_position_frequencies - 1)
        for head in range(heads):
            for pair_index in range(self.rotary_pairs):
                axis = (head + pair_index) % 3
                level = pair_index // 3
                exponent = max_band * level / max(1, pairs_per_axis - 1)
                frequencies[head, pair_index, axis] = (
                    2.0 * math.pi * (2.0**exponent)
                )
        self.frequencies = nn.Parameter(frequencies)

    @staticmethod
    def normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
        """Center each geometry and divide by its bounding-box diagonal."""
        coord_min = coords.amin(dim=-2, keepdim=True)
        coord_max = coords.amax(dim=-2, keepdim=True)
        center = 0.5 * (coord_min + coord_max)
        length = torch.linalg.vector_norm(
            coord_max - coord_min, dim=-1, keepdim=True
        )
        length = length.clamp_min(torch.finfo(coords.dtype).eps)
        return (coords - center) / length

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"RoPE coordinates must have shape (B, N, 3), got {tuple(coords.shape)}"
            )
        # Trigonometric phases are evaluated in float32 even under AMP.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            normalized = self.normalize_coordinates(coords.float())
            return torch.einsum(
                "bnc,hrc->bhnr", normalized, self.frequencies.float()
            )


def apply_rotary_embedding(
    features: torch.Tensor, phases: torch.Tensor
) -> torch.Tensor:
    """Rotate adjacent feature pairs using continuous position-dependent phases."""
    if features.shape[-1] % 2 != 0:
        raise ValueError("RoPE features must have an even final dimension")
    expected = (*features.shape[:-1], features.shape[-1] // 2)
    if tuple(phases.shape) != expected:
        raise ValueError(
            f"Expected phases {expected} for features {tuple(features.shape)}, "
            f"got {tuple(phases.shape)}"
        )

    phases = phases.to(dtype=features.dtype)
    cosine = torch.cos(phases)
    sine = torch.sin(phases)
    even = features[..., 0::2]
    odd = features[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RETOPlusAttention(nn.Module):
    """Soft cluster attention followed by RoPE cross-attention.

    Q, K and V use independent projections.  Content queries determine the
    node-to-cluster assignment.  RoPE is then applied to node Q/K before the
    cluster queries attend to all node keys and values.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        key_ratio: int = 1,
        cluster_num: int = 64,
        cluster_temperature_init: float = 1.0,
        cluster_temperature_min: float = 0.25,
        cluster_temperature_max: float = 4.0,
        cluster_eps: float = 1e-6,
    ):
        super().__init__()
        if cluster_num < 1:
            raise ValueError("cluster_num must be at least 1")
        if key_ratio < 1:
            raise ValueError("key_ratio must be at least 1")
        if cluster_temperature_init <= 0:
            raise ValueError("cluster_temperature_init must be positive")
        if not 0 < cluster_temperature_min <= cluster_temperature_init:
            raise ValueError("Invalid cluster_temperature_min")
        if cluster_temperature_max < cluster_temperature_init:
            raise ValueError("Invalid cluster_temperature_max")

        self.heads = heads
        self.dim_head = dim_head
        self.key_dim = key_ratio * dim_head
        self.cluster_num = cluster_num
        self.scale = self.key_dim**-0.5
        self.attention_dropout = dropout
        self.cluster_temperature_min = cluster_temperature_min
        self.cluster_temperature_max = cluster_temperature_max
        self.cluster_eps = cluster_eps

        # Independent projections preserve the full Q/K/V capacity.
        self.to_q = nn.Linear(dim, heads * self.key_dim, bias=False)
        self.to_k = nn.Linear(dim, heads * self.key_dim, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
        self.cluster_projection = nn.Parameter(
            torch.empty(heads, self.key_dim, cluster_num)
        )
        trunc_normal_(self.cluster_projection, std=0.02)
        self.cluster_log_temperature = nn.Parameter(
            torch.full((1, heads, 1, 1), math.log(cluster_temperature_init))
        )
        self.to_out = nn.Sequential(
            nn.Linear(heads * dim_head, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = rearrange(
            self.to_q(x), "b n (h d) -> b h n d", h=self.heads, d=self.key_dim
        )
        k = rearrange(
            self.to_k(x), "b n (h d) -> b h n d", h=self.heads, d=self.key_dim
        )
        v = rearrange(
            self.to_v(x), "b n (h d) -> b h n d", h=self.heads, d=self.dim_head
        )
        return q, k, v

    def _cluster_assignment(self, q: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bhnd,hdg->bhng", q, self.cluster_projection)
        temperature = self.cluster_temperature()
        return F.softmax(logits / temperature, dim=-1)

    def _aggregate_cluster_queries(
        self, assignment: torch.Tensor, q_rope: torch.Tensor
    ) -> torch.Tensor:
        cluster_sum = torch.einsum("bhng,bhnd->bhgd", assignment, q_rope)
        cluster_mass = assignment.sum(dim=-2).unsqueeze(-1)
        cluster_mean = cluster_sum / cluster_mass.clamp_min(self.cluster_eps)
        return F.normalize(
            cluster_mean, p=2.0, dim=-1, eps=self.cluster_eps
        ) * math.sqrt(self.key_dim)

    def cluster_temperature(self) -> torch.Tensor:
        return self.cluster_log_temperature.exp().clamp(
            min=self.cluster_temperature_min,
            max=self.cluster_temperature_max,
        )

    def forward(
        self, x: torch.Tensor, position_phases: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(x)

        # Clustering remains content-driven; RoPE modifies spatial similarity
        # in the following cluster-to-node cross-attention.
        assignment = self._cluster_assignment(q)
        q_rope = apply_rotary_embedding(q, position_phases)
        k_rope = apply_rotary_embedding(k, position_phases)
        cluster_queries = self._aggregate_cluster_queries(assignment, q_rope)

        cluster_features = F.scaled_dot_product_attention(
            cluster_queries,
            k_rope,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scale,
        )
        node_features = torch.einsum(
            "bhng,bhgd->bhnd", assignment, cluster_features
        )
        node_features = rearrange(node_features, "b h n d -> b n (h d)")
        return self.to_out(node_features)


class RETOPlusBlock(nn.Module):
    """Pre-LN RETO+ block with LayerScale residual stabilization."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: float = 2.0,
        key_ratio: int = 1,
        cluster_num: int = 64,
        layer_scale_init: float = 0.1,
        cluster_temperature_init: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if layer_scale_init <= 0:
            raise ValueError("layer_scale_init must be positive")

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = RETOPlusAttention(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            key_ratio=key_ratio,
            cluster_num=cluster_num,
            cluster_temperature_init=cluster_temperature_init,
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
        self.attn_layer_scale = nn.Parameter(
            torch.full((hidden_dim,), layer_scale_init)
        )
        self.mlp_layer_scale = nn.Parameter(
            torch.full((hidden_dim,), layer_scale_init)
        )

    def forward(
        self, fx: torch.Tensor, position_phases: torch.Tensor
    ) -> torch.Tensor:
        attn_update = self.attn(self.ln_1(fx), position_phases)
        fx = fx + self.attn_layer_scale.to(dtype=fx.dtype) * attn_update
        mlp_update = self.mlp(self.ln_2(fx))
        return fx + self.mlp_layer_scale.to(dtype=fx.dtype) * mlp_update


class RETOPlusNeuralOperator(nn.Module):
    """Standalone RETO+ model for node-wise aerodynamic regression.

    ``space_dim`` is the complete number of input channels.  The first three
    channels are always interpreted as XYZ for RoPE.  ShapeNet-Car training
    uses ``space_dim=7, out_dim=4``.
    """

    def __init__(
        self,
        space_dim: int = 7,
        n_layers: int = 8,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        Time_Input: bool = False,
        act: str = "gelu",
        mlp_ratio: float = 2.0,
        fun_dim: int = 0,
        out_dim: int = 4,
        key_ratio: int = 1,
        unified_pos: bool = False,
        cluster_num: int = 64,
        num_position_frequencies: int = 6,
        checkpoint_segment_size: int = 2,
        layer_scale_init: float = 0.1,
        cluster_temperature_init: float = 1.0,
        **_: Any,
    ):
        super().__init__()
        if space_dim < 3:
            raise ValueError("space_dim must include at least XYZ")
        if fun_dim != 0:
            raise ValueError("Pass all input channels through space_dim; fun_dim must be 0")
        if Time_Input or unified_pos:
            raise ValueError("Time_Input and unified_pos are not supported")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")
        if (key_ratio * (n_hidden // n_head)) % 2 != 0:
            raise ValueError("key_ratio * head_dim must be even for RoPE")
        if checkpoint_segment_size < 1:
            raise ValueError("checkpoint_segment_size must be at least 1")

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.cluster_num = cluster_num
        self.checkpoint_segment_size = checkpoint_segment_size
        self.preprocess = MLP(
            space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.rotary_embedding = ContinuousRotaryEmbedding3D(
            heads=n_head,
            rotary_dim=key_ratio * (n_hidden // n_head),
            num_position_frequencies=num_position_frequencies,
        )
        self.blocks = nn.ModuleList(
            RETOPlusBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                key_ratio=key_ratio,
                cluster_num=cluster_num,
                layer_scale_init=layer_scale_init,
                cluster_temperature_init=cluster_temperature_init,
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

    def _validate_input(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None
    ) -> torch.Tensor:
        if not torch.is_tensor(data) or data.ndim != 3:
            raise ValueError("Input must have shape (B, N, C)")
        if data.shape[-1] != self.space_dim:
            raise ValueError(
                f"Expected {self.space_dim} input channels, got {data.shape[-1]}"
            )
        coords = data[..., :3] if rope_coords is None else rope_coords
        if coords.ndim != 3 or coords.shape[:2] != data.shape[:2] or coords.shape[-1] != 3:
            raise ValueError("rope_coords must have shape (B, N, 3)")
        return coords

    def _forward_block_range(
        self,
        fx: torch.Tensor,
        phases: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        for block_index in range(start, end):
            fx = self.blocks[block_index](fx, phases)
        return fx

    def _forward_blocks(
        self, fx: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_block_range(fx, phases, 0, len(self.blocks))

    def _forward_with_checkpoint(
        self, fx: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        for start in range(0, len(self.blocks), self.checkpoint_segment_size):
            end = min(start + self.checkpoint_segment_size, len(self.blocks))

            def run_segment(
                segment_fx: torch.Tensor,
                segment_phases: torch.Tensor,
                start: int = start,
                end: int = end,
            ) -> torch.Tensor:
                return self._forward_block_range(
                    segment_fx, segment_phases, start, end
                )

            fx = checkpoint(run_segment, fx, phases, use_reentrant=False)
        return fx

    def forward(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        coords = self._validate_input(data, rope_coords)
        phases = self.rotary_embedding(coords)
        fx = self.preprocess(data)
        fx = fx + self.placeholder.to(dtype=fx.dtype)[None, None, :]
        if self.training:
            fx = self._forward_with_checkpoint(fx, phases)
        else:
            fx = self._forward_blocks(fx, phases)
        return self.output_decoder(self.output_norm(fx))

    def stability_values(self) -> dict[str, torch.Tensor]:
        """Return small CPU tensors for monitoring training stability."""
        return {
            "cluster_temperature": torch.stack(
                [
                    block.attn.cluster_temperature().detach().flatten()
                    for block in self.blocks
                ]
            ).cpu(),
            "attention_layer_scale": torch.stack(
                [block.attn_layer_scale.detach() for block in self.blocks]
            ).cpu(),
            "mlp_layer_scale": torch.stack(
                [block.mlp_layer_scale.detach() for block in self.blocks]
            ).cpu(),
        }


# Both aliases match the import conventions used elsewhere in this repository.
Model = RETOPlusNeuralOperator
LinearAttentionNeuralOperator = RETOPlusNeuralOperator
