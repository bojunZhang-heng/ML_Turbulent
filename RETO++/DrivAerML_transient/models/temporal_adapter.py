"""Shared transient input adapter and model registry."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from SATO_dataset.Dataset import pos_to_order_inverse_index
from models.CANO import LinearAttentionNeuralOperator as CANO
from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator as LinearNO
from models.RETO import Model as RETO
from models.RETO_plus import Model as RETOPlus
from models.RETO_plus_v2 import Model as RETOPlusV2
from models.SATO import Model as SATO
from models.transolver_model import Model as Transolver


SUPPORTED_MODELS = (
    "Transolver",
    "LinearNO",
    "CANO",
    "RETO",
    "RETO_plus",
    "RETO_plus_v2",
    "SATO",
)


class TemporalModelAdapter(nn.Module):
    """Give every spatial backbone the same next-frame interface.

    Per-point input layout is ``XYZ | context fields | time | delta_t``.  XYZ
    remains the first three channels, so coordinate-aware backbones apply their
    positional operations only to real spatial coordinates.
    """

    def __init__(
        self,
        model_name: str,
        backbone: nn.Module,
        state_channels: int,
        context_steps: int,
        residual_prediction: bool,
        coordinates: torch.Tensor,
        sato_grid_size: float = 0.05,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone
        self.state_channels = int(state_channels)
        self.context_steps = int(context_steps)
        self.residual_prediction = bool(residual_prediction)
        if model_name == "SATO":
            order, inverse = pos_to_order_inverse_index(
                coordinates,
                device=coordinates.device,
                tensor=True,
                grid_size=sato_grid_size,
            )
        else:
            order = torch.empty(0, dtype=torch.long)
            inverse = torch.empty(0, dtype=torch.long)
        self.register_buffer("serialization_order", order, persistent=True)
        self.register_buffer("serialization_inverse", inverse, persistent=True)

    @property
    def input_channels(self) -> int:
        return 3 + self.state_channels * self.context_steps + 2

    def _assemble_input(
        self,
        coordinates: torch.Tensor,
        current_state: torch.Tensor,
        time: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape (B,N,3)")
        expected = self.state_channels * self.context_steps
        if current_state.shape[:2] != coordinates.shape[:2] or current_state.shape[-1] != expected:
            raise ValueError(
                f"current_state must have shape (B,N,{expected}), got "
                f"{tuple(current_state.shape)}"
            )
        batch, points, _ = coordinates.shape
        time_features = torch.stack((time, delta_t), dim=-1)
        time_features = time_features[:, None, :].expand(batch, points, 2)
        return torch.cat((coordinates, current_state, time_features), dim=-1)

    def forward(
        self,
        coordinates: torch.Tensor,
        current_state: torch.Tensor,
        time: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        model_input = self._assemble_input(coordinates, current_state, time, delta_t)
        if self.model_name == "SATO":
            batch = model_input.shape[0]
            order = self.serialization_order.expand(batch, -1, -1)
            inverse = self.serialization_inverse.expand(batch, -1, -1)
            update = self.backbone(model_input, order, inverse)
        elif self.model_name in {"RETO_plus", "RETO_plus_v2"}:
            update = self.backbone(model_input, rope_coords=coordinates)
        else:
            update = self.backbone(model_input)
        if self.residual_prediction:
            update = current_state[..., -self.state_channels :] + update
        return update


def value(config: dict[str, Any], name: str, default: Any) -> Any:
    result = config.get(name, default)
    return default if result is None else result


def build_temporal_model(
    model_name: str,
    state_channels: int,
    context_steps: int,
    coordinates: torch.Tensor,
    config: dict[str, Any],
) -> TemporalModelAdapter:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; choose one of {SUPPORTED_MODELS}"
        )
    input_channels = 3 + state_channels * context_steps + 2
    common = dict(
        space_dim=input_channels,
        n_layers=int(value(config, "n_layers", 8)),
        n_hidden=int(value(config, "n_hidden", 256)),
        dropout=float(value(config, "dropout", 0.0)),
        n_head=int(value(config, "n_head", 8)),
        act=str(value(config, "act", "gelu")),
        mlp_ratio=int(value(config, "mlp_ratio", 2)),
        out_dim=state_channels,
    )

    if model_name == "Transolver":
        backbone = Transolver(
            **common,
            fun_dim=0,
            slice_num=int(value(config, "slice_num", 64)),
            ref=int(value(config, "ref", 8)),
            unified_pos=False,
        )
    elif model_name == "LinearNO":
        backbone = LinearNO(
            **common,
            Time_Input=False,
            fun_dim=0,
            key_ratio=int(value(config, "key_ratio", 1)),
            unified_pos=False,
            isregular=False,
        )
    elif model_name == "CANO":
        backbone = CANO(
            **common,
            Time_Input=False,
            fun_dim=0,
            key_ratio=int(value(config, "key_ratio", 1)),
            unified_pos=False,
            cluster_num=int(value(config, "cluster_num", 64)),
            checkpoint_segment_size=int(value(config, "checkpoint_segment_size", 2)),
            layer_scale_init=float(value(config, "layer_scale_init", 0.1)),
            cluster_temperature_init=float(
                value(config, "cluster_temperature_init", 1.0)
            ),
        )
    elif model_name == "RETO":
        backbone = RETO(
            **common,
            sincos_max_wavelength=int(value(config, "sincos_max_wavelength", 10000)),
            rope_max_wavelength=int(value(config, "rope_max_wavelength", 10000)),
            checkpoint_segment_size=int(value(config, "checkpoint_segment_size", 2)),
            checkpoint_num_blocks=value(config, "checkpoint_num_blocks", None),
        )
    elif model_name in {"RETO_plus", "RETO_plus_v2"}:
        model_class = RETOPlus if model_name == "RETO_plus" else RETOPlusV2
        backbone = model_class(
            **common,
            Time_Input=False,
            fun_dim=0,
            key_ratio=int(value(config, "key_ratio", 1)),
            unified_pos=False,
            cluster_num=int(value(config, "cluster_num", 64)),
            num_position_frequencies=int(
                value(config, "num_position_frequencies", 6)
            ),
            checkpoint_segment_size=int(value(config, "checkpoint_segment_size", 2)),
            checkpoint_num_blocks=value(config, "checkpoint_num_blocks", None),
            layer_scale_init=float(value(config, "layer_scale_init", 0.1)),
            cluster_temperature_init=float(
                value(config, "cluster_temperature_init", 1.0)
            ),
            sincos_max_wavelength=int(value(config, "sincos_max_wavelength", 10000)),
        )
    else:
        backbone = SATO(
            **common,
            fun_dim=0,
            slice_num=int(value(config, "slice_num", 64)),
            patch_size=int(value(config, "patch_size", 20)),
            shift=int(value(config, "shift", 2)),
            n_iter=int(value(config, "n_iter", 1)),
        )

    return TemporalModelAdapter(
        model_name=model_name,
        backbone=backbone,
        state_channels=state_channels,
        context_steps=context_steps,
        residual_prediction=bool(value(config, "residual_prediction", True)),
        coordinates=coordinates,
        sato_grid_size=float(value(config, "sato_grid_size", 0.05)),
    )
