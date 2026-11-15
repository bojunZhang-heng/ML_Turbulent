"""RETO+ v2 with continuous sine/cosine coordinate preprocessing.

The RETO+ attention, continuous rotary attention, cluster assignment, layer
scales and activation-checkpointing are inherited unchanged from RETO_plus.
Only the pointwise input encoder is replaced: normalized XYZ coordinates are
mapped directly to ``n_hidden`` channels by a continuous 3-D sin/cos basis.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.RETO_plus import Model as RETOPlusModel
from models.continuous_sincos_embed import ContinuousSincosEmbed


class TransientSincosPreprocess(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_wavelength: int):
        super().__init__()
        self.input_dim = input_dim
        self.position = ContinuousSincosEmbed(
            dim=hidden_dim,
            ndim=3,
            max_wavelength=max_wavelength,
            assert_positive=False,
        )
        condition_dim = input_dim - 3
        self.condition = (
            nn.Sequential(
                nn.Linear(hidden_dim + condition_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            if condition_dim
            else nn.Identity()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        position = self.position(data[..., :3])
        if self.input_dim > 3:
            position = torch.cat((position, data[..., 3:]), dim=-1)
        return self.condition(position)


class Model(RETOPlusModel):
    """RETO+ whose input preprocessing is continuous Fourier coordinates."""

    def __init__(
        self,
        space_dim: int = 3,
        sincos_max_wavelength: int = 10000,
        **kwargs: Any,
    ):
        if space_dim < 3:
            raise ValueError("Transient RETO_plus_v2 input must include XYZ")
        if sincos_max_wavelength <= 0:
            raise ValueError("sincos_max_wavelength must be positive")

        super().__init__(space_dim=space_dim, **kwargs)
        self.sincos_max_wavelength = int(sincos_max_wavelength)
        self.preprocess = TransientSincosPreprocess(
            input_dim=space_dim,
            hidden_dim=self.n_hidden,
            max_wavelength=self.sincos_max_wavelength,
        )

    def forward(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        if data.ndim != 3 or data.shape[-1] != self.space_dim:
            raise ValueError(
                f"Expected input shape (B, N, {self.space_dim}), got {tuple(data.shape)}"
            )
        return super().forward(data, rope_coords=rope_coords)


LinearAttentionNeuralOperator = Model
