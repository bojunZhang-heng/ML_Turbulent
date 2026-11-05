"""RETO+ v2 with continuous sine/cosine coordinate preprocessing.

The RETO+ attention, continuous rotary attention, cluster assignment, layer
scales and activation-checkpointing are inherited unchanged from RETO_plus.
Only the pointwise input encoder is replaced: normalized XYZ coordinates are
mapped directly to ``n_hidden`` channels by a continuous 3-D sin/cos basis.
"""

from __future__ import annotations

from typing import Any

import torch

from models.RETO_plus import Model as RETOPlusModel
from models.continuous_sincos_embed import ContinuousSincosEmbed


class Model(RETOPlusModel):
    """RETO+ whose input preprocessing is continuous Fourier coordinates."""

    def __init__(
        self,
        space_dim: int = 3,
        sincos_max_wavelength: int = 10000,
        **kwargs: Any,
    ):
        if space_dim != 3:
            raise ValueError(
                "RETO_plus_v2 expects space_dim=3 because its continuous "
                "sine/cosine encoder consumes XYZ coordinates."
            )
        if sincos_max_wavelength <= 0:
            raise ValueError("sincos_max_wavelength must be positive")

        super().__init__(space_dim=space_dim, **kwargs)
        self.sincos_max_wavelength = int(sincos_max_wavelength)
        self.preprocess = ContinuousSincosEmbed(
            dim=self.n_hidden,
            ndim=3,
            max_wavelength=self.sincos_max_wavelength,
            assert_positive=False,
        )

    def forward(
        self, data: torch.Tensor, rope_coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(
                f"RETO_plus_v2 expects input shape (B, N, 3), got {tuple(data.shape)}"
            )
        return super().forward(data, rope_coords=rope_coords)


LinearAttentionNeuralOperator = Model

