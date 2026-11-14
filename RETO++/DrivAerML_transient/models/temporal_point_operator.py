"""Linear-attention point operator for one-step transient prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FourierCoordinates(nn.Module):
    def __init__(self, frequencies: int) -> None:
        super().__init__()
        if frequencies < 0:
            raise ValueError("frequencies cannot be negative")
        self.frequencies = frequencies
        values = 2.0 ** torch.arange(frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("bands", values, persistent=False)

    @property
    def output_dim(self) -> int:
        return 6 * self.frequencies

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.frequencies == 0:
            return coordinates.new_empty(*coordinates.shape[:-1], 0)
        angles = coordinates.unsqueeze(-1) * self.bands
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class TimeConditioner(nn.Module):
    def __init__(self, hidden_dim: int, frequencies: int = 8) -> None:
        super().__init__()
        self.frequencies = frequencies
        bands = 2.0 ** torch.arange(frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("bands", bands, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(2 + 4 * frequencies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        values = torch.stack((time, delta_t), dim=-1)
        angles = values.unsqueeze(-1) * self.bands
        encoded = torch.cat(
            (values, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1
        )
        return self.net(encoded)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, points, _ = values.shape
        return values.reshape(batch, points, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        query = F.softmax(self._heads(self.q_proj(values)), dim=-1)
        key = F.softmax(self._heads(self.k_proj(values)), dim=-2)
        value = self._heads(self.v_proj(values))
        context = torch.matmul(key.transpose(-2, -1), value)
        output = torch.matmul(query, context).transpose(1, 2).contiguous()
        output = output.reshape(values.shape)
        return self.dropout(self.out_proj(output))


class OperatorBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        inner = max(dim, int(dim * mlp_ratio))
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = LinearAttention(dim, heads, dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )
        self.attention_scale = nn.Parameter(torch.full((dim,), 1.0e-2))
        self.mlp_scale = nn.Parameter(torch.full((dim,), 1.0e-2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention_scale * self.attention(
            self.norm_attention(values)
        )
        values = values + self.mlp_scale * self.mlp(self.norm_mlp(values))
        return values


class TemporalPointOperator(nn.Module):
    """Predict the next normalized field on a fixed unstructured point cloud."""

    def __init__(
        self,
        state_channels: int,
        context_steps: int = 1,
        hidden_dim: int = 128,
        layers: int = 6,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        position_frequencies: int = 6,
        time_frequencies: int = 8,
        checkpoint_blocks: bool = True,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if state_channels < 1 or context_steps < 1:
            raise ValueError("state_channels and context_steps must be positive")
        self.state_channels = state_channels
        self.context_steps = context_steps
        self.checkpoint_blocks = checkpoint_blocks
        self.residual_prediction = residual_prediction
        self.position = FourierCoordinates(position_frequencies)
        input_dim = 3 + state_channels * context_steps + self.position.output_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_conditioner = TimeConditioner(hidden_dim, time_frequencies)
        self.blocks = nn.ModuleList(
            [OperatorBlock(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, state_channels),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        current_state: torch.Tensor,
        time: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.state_channels * self.context_steps
        if current_state.shape[-1] != expected:
            raise ValueError(
                f"current_state has {current_state.shape[-1]} channels, expected {expected}"
            )
        features = torch.cat(
            (coordinates, self.position(coordinates), current_state), dim=-1
        )
        features = self.input_projection(features)
        features = features + self.time_conditioner(time, delta_t).unsqueeze(1)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and features.requires_grad:
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        update = self.output(features)
        if self.residual_prediction:
            return current_state[..., -self.state_channels :] + update
        return update
