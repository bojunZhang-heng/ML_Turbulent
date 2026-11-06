"""Construct DrivAerStar models from the same YAML keys as training."""

from __future__ import annotations

from pathlib import Path

import torch

from models.CANO import LinearAttentionNeuralOperator as CANO
from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator as LinearNO
from models.RETO import Model as RETO
from models.RETO_plus import Model as RETOPlus
from models.RETO_plus_v2 import Model as RETOPlusV2
from models.SATO import Model as SATO
from models.transolver_model import Model as Transolver

SUPPORTED_MODELS = {"Transolver", "LinearNO", "CANO", "RETO", "RETO_plus", "RETO_plus_v2", "SATO"}


def checkpoint_model_hint(state_dict: dict[str, torch.Tensor]) -> str | None:
    """Infer only architectures with an unambiguous state-dict signature."""

    keys = set(state_dict)
    if "preprocess.omega" in keys:
        return "RETO_plus_v2"
    if {
        "preprocess.linear_pre.0.weight",
        "preprocess.linear_post.weight",
        "rotary_embedding.frequencies",
    }.issubset(keys):
        return "RETO_plus"
    return None


def value(config, name, default):
    result = config.get(name, default)
    return default if result is None else result


def build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    common = dict(
        space_dim=int(value(cfg, "space_dim", 3)),
        n_layers=int(value(cfg, "n_layers", 8)),
        n_hidden=int(value(cfg, "n_hidden", 256)),
        dropout=float(value(cfg, "dropout", 0.0)),
        n_head=int(value(cfg, "n_head", 8)),
        act=str(value(cfg, "act", "gelu")),
        mlp_ratio=value(cfg, "mlp_ratio", 2),
        out_dim=int(value(cfg, "out_dim", 1)),
    )
    if model_name == "Transolver":
        return Transolver(**common, fun_dim=int(value(cfg, "fun_dim", 0)),
                          slice_num=int(value(cfg, "slice_num", 64)),
                          ref=int(value(cfg, "ref", 8)),
                          unified_pos=bool(value(cfg, "unified_pos", False)))
    if model_name == "LinearNO":
        return LinearNO(**common, Time_Input=bool(value(cfg, "time_input", False)),
                        fun_dim=int(value(cfg, "fun_dim", 0)),
                        key_ratio=int(value(cfg, "key_ratio", 1)),
                        ref=int(value(cfg, "ref", 8)),
                        unified_pos=bool(value(cfg, "unified_pos", False)),
                        H=int(value(cfg, "H", 85)), W=int(value(cfg, "W", 85)),
                        isregular=bool(value(cfg, "isregular", False)))
    if model_name == "CANO":
        return CANO(**common, Time_Input=bool(value(cfg, "time_input", False)),
                    fun_dim=int(value(cfg, "fun_dim", 0)),
                    key_ratio=int(value(cfg, "key_ratio", 1)),
                    ref=int(value(cfg, "ref", 8)),
                    unified_pos=bool(value(cfg, "unified_pos", False)),
                    H=int(value(cfg, "H", 85)), W=int(value(cfg, "W", 85)),
                    isregular=bool(value(cfg, "isregular", False)),
                    cluster_num=int(value(cfg, "cluster_num", 64)),
                    checkpoint_segment_size=int(value(cfg, "checkpoint_segment_size", 2)),
                    checkpoint_num_blocks=value(cfg, "checkpoint_num_blocks", None),
                    layer_scale_init=float(value(cfg, "layer_scale_init", 0.1)),
                    cluster_temperature_init=float(value(cfg, "cluster_temperature_init", 1.0)))
    if model_name == "RETO_plus":
        return RETOPlus(
            **common,
            Time_Input=bool(value(cfg, "time_input", False)),
            fun_dim=int(value(cfg, "fun_dim", 0)),
            key_ratio=int(value(cfg, "key_ratio", 1)),
            unified_pos=bool(value(cfg, "unified_pos", False)),
            cluster_num=int(value(cfg, "cluster_num", 64)),
            num_position_frequencies=int(value(cfg, "num_position_frequencies", 6)),
            checkpoint_segment_size=int(value(cfg, "checkpoint_segment_size", 2)),
            checkpoint_num_blocks=value(cfg, "checkpoint_num_blocks", None),
            layer_scale_init=float(value(cfg, "layer_scale_init", 0.1)),
            cluster_temperature_init=float(value(cfg, "cluster_temperature_init", 1.0)),
        )
    if model_name == "RETO_plus_v2":
        return RETOPlusV2(
            **common,
            Time_Input=bool(value(cfg, "time_input", False)),
            fun_dim=int(value(cfg, "fun_dim", 0)),
            key_ratio=int(value(cfg, "key_ratio", 1)),
            unified_pos=bool(value(cfg, "unified_pos", False)),
            cluster_num=int(value(cfg, "cluster_num", 64)),
            num_position_frequencies=int(value(cfg, "num_position_frequencies", 6)),
            checkpoint_segment_size=int(value(cfg, "checkpoint_segment_size", 2)),
            checkpoint_num_blocks=value(cfg, "checkpoint_num_blocks", None),
            layer_scale_init=float(value(cfg, "layer_scale_init", 0.1)),
            cluster_temperature_init=float(value(cfg, "cluster_temperature_init", 1.0)),
            sincos_max_wavelength=int(
                value(cfg, "sincos_max_wavelength", 10000)
            ),
        )
    if model_name == "RETO":
        return RETO(**common,
                    sincos_max_wavelength=int(value(cfg, "sincos_max_wavelength", 10000)),
                    rope_max_wavelength=int(value(cfg, "rope_max_wavelength", 10000)),
                    checkpoint_segment_size=int(value(cfg, "checkpoint_segment_size", 2)),
                    checkpoint_num_blocks=value(cfg, "checkpoint_num_blocks", None))
    if model_name == "SATO":
        return SATO(**common, fun_dim=int(value(cfg, "fun_dim", 0)),
                    slice_num=int(value(cfg, "slice_num", 64)),
                    patch_size=int(value(cfg, "patch_size", 20)),
                    shift=int(value(cfg, "shift", 2)), n_iter=int(value(cfg, "n_iter", 1)))
    raise ValueError(f"Unsupported model {model_name!r}; choose from {sorted(SUPPORTED_MODELS)}")


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    expected_model_name: str | None = None,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a state_dict")
    if payload and all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    checkpoint_hint = checkpoint_model_hint(payload)
    if (
        expected_model_name in {"RETO_plus", "RETO_plus_v2"}
        and checkpoint_hint is not None
        and checkpoint_hint != expected_model_name
    ):
        signatures = {
            "RETO_plus": "MLP preprocess (preprocess.linear_pre/linear_post)",
            "RETO_plus_v2": "continuous sin/cos preprocess (preprocess.omega)",
        }
        raise ValueError(
            f"Checkpoint/model mismatch: YAML builds {expected_model_name} with "
            f"{signatures[expected_model_name]}, but {path.name} contains "
            f"{checkpoint_hint} with {signatures[checkpoint_hint]}. Use the "
            f"matching config_{checkpoint_hint}.yml file."
        )
    try:
        model.load_state_dict(payload, strict=True)
    except RuntimeError as error:
        hint = (
            f" The checkpoint signature looks like {checkpoint_hint}; verify "
            f"that the YAML model_name and all model hyperparameters match."
            if checkpoint_hint is not None
            else " Verify that the checkpoint and YAML use the same model and hyperparameters."
        )
        raise RuntimeError(f"Unable to load checkpoint {path}.{hint}\n{error}") from error
