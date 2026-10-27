import math

import torch

from models.CANO import ClusterAttention, ClusterAttentionNeuralOperator


class ContinuousRotaryEmbedding3D(torch.nn.Module):
    """Learnable continuous 3D rotary phases for irregular point clouds."""

    def __init__(
        self,
        heads: int,
        rotary_dim: int,
        num_position_frequencies: int = 6,
        space_dim: int = 3,
    ):
        super().__init__()
        if rotary_dim % 2 != 0:
            raise ValueError("The RoPE dimension must be even")
        if num_position_frequencies < 1:
            raise ValueError("num_position_frequencies must be at least 1")
        if space_dim != 3:
            raise ValueError("ContinuousRotaryEmbedding3D expects XYZ coordinates")

        self.heads = heads
        self.rotary_pairs = rotary_dim // 2
        self.space_dim = space_dim

        frequencies = torch.zeros(heads, self.rotary_pairs, space_dim)
        pairs_per_axis = max(1, math.ceil(self.rotary_pairs / space_dim))
        max_band = max(0, num_position_frequencies - 1)
        for head in range(heads):
            for pair_index in range(self.rotary_pairs):
                axis = (head + pair_index) % space_dim
                level = pair_index // space_dim
                exponent = max_band * level / max(1, pairs_per_axis - 1)
                frequencies[head, pair_index, axis] = (
                    2.0 * math.pi * (2.0**exponent)
                )
        self.frequencies = torch.nn.Parameter(frequencies)

    @staticmethod
    def normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
        coord_min = coords.amin(dim=-2, keepdim=True)
        coord_max = coords.amax(dim=-2, keepdim=True)
        center = 0.5 * (coord_min + coord_max)
        length = torch.linalg.vector_norm(
            coord_max - coord_min, dim=-1, keepdim=True
        )
        length = length.clamp_min(torch.finfo(coords.dtype).eps)
        return (coords - center) / length

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != self.space_dim:
            raise ValueError(
                f"Expected coordinates shaped (B, N, {self.space_dim}), "
                f"got {tuple(coords.shape)}"
            )
        with torch.autocast(device_type=coords.device.type, enabled=False):
            normalized = self.normalize_coordinates(coords.float())
            return torch.einsum(
                "bnc,hrc->bhnr", normalized, self.frequencies.float()
            )


def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    expected_shape = (*x.shape[:-1], x.shape[-1] // 2)
    if x.shape[-1] % 2 != 0 or phases.shape != expected_shape:
        raise ValueError(
            f"Expected phases shaped {expected_shape} for features {tuple(x.shape)}, "
            f"got {tuple(phases.shape)}"
        )

    phases = phases.to(dtype=x.dtype)
    cos_phase = torch.cos(phases)
    sin_phase = torch.sin(phases)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos_phase - odd * sin_phase
    rotated_odd = even * sin_phase + odd * cos_phase
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RopeClusterAttention(ClusterAttention):
    """CANO cross-attention using node-level RoPE on Q and K."""

    def forward(self, x, position_phases):
        q, k, v = self._project_qkv(x)
        cluster_assignment = self._cluster_assignment(q)

        q_rope = apply_rotary_embedding(q, position_phases)
        k_rope = apply_rotary_embedding(k, position_phases)
        clustered_q_rope = torch.einsum(
            "bhnk,bhnd->bhkd", cluster_assignment, q_rope
        )

        cluster_features = self._cross_attention(clustered_q_rope, k_rope, v)
        output = self._broadcast_clusters(cluster_assignment, cluster_features)
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(output)


class RopeClusterAttentionNeuralOperator(ClusterAttentionNeuralOperator):
    """CANO with continuous three-dimensional rotary cross-attention."""

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
        key_ratio=1,
        ref=8,
        unified_pos=False,
        H=85,
        W=85,
        isregular=False,
        cluster_num=64,
        num_position_frequencies=6,
        checkpoint_segment_size=2,
        _attention_cls=RopeClusterAttention,
        _attention_kwargs=None,
        **_,
    ):
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")
        key_dim = key_ratio * (n_hidden // n_head)
        if key_dim % 2 != 0:
            raise ValueError("key_ratio * head_dim must be even for RoPE")

        super().__init__(
            space_dim=space_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            Time_Input=Time_Input,
            act=act,
            mlp_ratio=mlp_ratio,
            fun_dim=fun_dim,
            out_dim=out_dim,
            key_ratio=key_ratio,
            ref=ref,
            unified_pos=unified_pos,
            H=H,
            W=W,
            isregular=isregular,
            cluster_num=cluster_num,
            checkpoint_segment_size=checkpoint_segment_size,
            _attention_cls=_attention_cls,
            _attention_kwargs=_attention_kwargs,
        )
        self.rotary_embedding = ContinuousRotaryEmbedding3D(
            heads=n_head,
            rotary_dim=key_dim,
            num_position_frequencies=num_position_frequencies,
            space_dim=space_dim,
        )

    @staticmethod
    def _phase_kwargs(position_phases):
        return {"position_phases": position_phases}

    def forward(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._validate_input(data)
        position_coords = data if rope_coords is None else rope_coords
        if position_coords.shape != data.shape:
            raise ValueError("rope_coords must have the same (B, N, 3) shape as data")

        position_phases = self.rotary_embedding(position_coords)
        fx = self._encode(data)
        if self.training:
            return self._forward_with_checkpoint(
                fx,
                extra_tensor=position_phases,
                kwargs_factory=self._phase_kwargs,
            )
        return self._forward_blocks(
            fx, attn_kwargs=self._phase_kwargs(position_phases)
        )


LinearAttentionNeuralOperator = RopeClusterAttentionNeuralOperator

