"""RETO+ v2 adapted to ShapeNet-Car seven-channel inputs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.RETO_plus import Model as RETOPlusModel
from models.continuous_sincos_embed import ContinuousSincosEmbed


class ShapeNetPreprocess(nn.Module):
    """Encode XYZ continuously and fuse SDF/normals through a linear branch."""

    def __init__(self, input_dim: int, hidden_dim: int, max_wavelength: int):
        super().__init__()
        if input_dim < 3:
            raise ValueError("ShapeNet preprocessing requires at least XYZ")
        self.input_dim = input_dim
        self.coordinate_encoder = ContinuousSincosEmbed(
            dim=hidden_dim,
            ndim=3,
            max_wavelength=max_wavelength,
            assert_positive=False,
        )
        self.extra_encoder = (
            nn.Linear(input_dim - 3, hidden_dim, bias=False)
            if input_dim > 3 else None
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim != 3 or data.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (B, N, {self.input_dim}), got {tuple(data.shape)}"
            )
        encoded = self.coordinate_encoder(data[..., :3])
        if self.extra_encoder is not None:
            encoded = encoded + self.extra_encoder(data[..., 3:])
        return encoded


class Model(RETOPlusModel):
    """RETO+ v2 for ShapeNet-Car ``[XYZ, SDF, normal_xyz]`` inputs."""

    def __init__(self, space_dim: int = 7, sincos_max_wavelength: int = 10000, **kwargs: Any):
        if space_dim < 3:
            raise ValueError("ShapeNet RETO_plus_v2 expects at least XYZ")
        if sincos_max_wavelength <= 0:
            raise ValueError("sincos_max_wavelength must be positive")
        super().__init__(space_dim=space_dim, **kwargs)
        self.sincos_max_wavelength = int(sincos_max_wavelength)
        self.preprocess = ShapeNetPreprocess(space_dim, self.n_hidden, self.sincos_max_wavelength)


LinearAttentionNeuralOperator = Model
