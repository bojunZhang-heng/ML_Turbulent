"""Construct dataset-local models from the same YAML keys as training."""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import torch

from models.CANO import LinearAttentionNeuralOperator as CANO
from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator as LinearNO
from models.RETO import Model as RETO
from models.RETO_plus import Model as RETOPlus
from models.RETO_plus_v2 import Model as RETOPlusV2
from models.transolver_model import Model as Transolver

BUILTIN_MODELS = {
    "Transolver", "LinearNO", "CANO", "RETO", "RETO_plus", "RETO_plus_v2", "SATO"
}

TRAINING_PROTOCOLS = (
    "adamw_cosine_warmup",
    "adam_onecycle",
    "adamw_step",
    "adamw_plateau",
    "adamw_onecycle",
)
SELECTION_METRICS = ("val_loss", "relative_error")


def _checkpoint_sort_key(path: Path):
    """Prefer the timestamp in new filenames, then filesystem mtime."""
    match = re.search(r"_(\d{8}_\d{4})(?:_(?:val_loss|relative_error))?$", path.stem)
    return (match.group(1) if match else "", path.stat().st_mtime_ns, path.name)


def resolve_checkpoint_path(
    checkpoint: Path,
    training_protocol: str | None = None,
    selection_metric: str | None = None,
) -> Path:
    """Resolve an explicit legacy file or select a new protocol checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint
    if training_protocol is not None and training_protocol not in TRAINING_PROTOCOLS:
        raise ValueError(
            f"Unknown training_protocol={training_protocol!r}; choose from {TRAINING_PROTOCOLS}"
        )
    if selection_metric is not None and selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"Unknown selection_metric={selection_metric!r}; choose from {SELECTION_METRICS}"
        )

    # A missing old base path may now have _val_loss/_relative_error appended.
    if not checkpoint.exists() and checkpoint.suffix:
        metrics = [selection_metric] if selection_metric else list(SELECTION_METRICS)
        derived = [
            checkpoint.with_name(f"{checkpoint.stem}_{metric}{checkpoint.suffix}")
            for metric in metrics
        ]
        existing = [path for path in derived if path.is_file()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise ValueError(
                f"Both validation checkpoints exist for {checkpoint}. Set "
                "selection_metric to 'val_loss' or 'relative_error'."
            )

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint file or directory not found: {checkpoint}")
    candidates = sorted(
        (path for path in checkpoint.glob("*.pth") if path.is_file()),
        key=_checkpoint_sort_key,
    )
    if training_protocol:
        candidates = [
            path for path in candidates
            if f"_{training_protocol}_" in path.name
        ]
    if selection_metric:
        candidates = [
            path for path in candidates
            if path.name.endswith(f"_{selection_metric}.pth")
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matched directory={checkpoint}, "
            f"training_protocol={training_protocol!r}, "
            f"selection_metric={selection_metric!r}."
        )
    if len(candidates) > 1 and not (training_protocol and selection_metric):
        preview = "\n  ".join(path.name for path in candidates[-10:])
        raise ValueError(
            "Checkpoint selection is ambiguous. Set both training_protocol and "
            f"selection_metric. Matching files:\n  {preview}"
        )
    return candidates[-1]


def _infer_checkpoint_metadata(path: Path) -> dict:
    name = path.name
    protocol = next(
        (value for value in TRAINING_PROTOCOLS if f"_{value}_" in name), None
    )
    metric = next(
        (value for value in SELECTION_METRICS if name.endswith(f"_{value}.pth")),
        None,
    )
    return {"training_protocol": protocol, "selection_metric": metric}


def _extract_state_dict(payload, path: Path):
    metadata = {}
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict(), "serialized_module", metadata
    if isinstance(payload, dict):
        for metadata_key in ("metadata", "meta", "training_metadata"):
            if isinstance(payload.get(metadata_key), dict):
                metadata.update(payload[metadata_key])
        for key in (
            "model_state_dict", "state_dict", "model", "model_state", "net", "network"
        ):
            nested = payload.get(key)
            if isinstance(nested, torch.nn.Module):
                return nested.state_dict(), f"wrapped:{key}:module", metadata
            if isinstance(nested, dict):
                return nested, f"wrapped:{key}", metadata
        if all(isinstance(key, str) for key in payload):
            return payload, "state_dict", metadata
    raise TypeError(f"{path} does not contain a supported model state_dict")


def _strip_state_dict_prefixes(state_dict):
    result = state_dict
    stripped = []
    for prefix in ("module.", "_orig_mod.", "model."):
        while result and all(str(key).startswith(prefix) for key in result):
            result = {str(key)[len(prefix):]: value for key, value in result.items()}
            stripped.append(prefix[:-1])
    return result, stripped


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
        try:
            from models.SATO import Model as SATO
        except ImportError as error:
            raise ImportError(
                "SATO is requested but models/SATO.py is unavailable in this dataset directory"
            ) from error
        return SATO(**common, fun_dim=int(value(cfg, "fun_dim", 0)),
                    slice_num=int(value(cfg, "slice_num", 64)),
                    patch_size=int(value(cfg, "patch_size", 20)),
                    shift=int(value(cfg, "shift", 2)), n_iter=int(value(cfg, "n_iter", 1)))
    module_name = cfg.get("model_module")
    class_name = str(value(cfg, "model_class", "Model"))
    if not module_name:
        raise ValueError(
            f"Unsupported model {model_name!r}. Built-ins are {sorted(BUILTIN_MODELS)}. "
            "For a future model, set model.model_module and optionally model.model_class "
            "in its YAML; constructor arguments are read from model.model_kwargs."
        )
    import importlib

    module = importlib.import_module(str(module_name))
    model_class = getattr(module, class_name)
    kwargs = cfg.get("model_kwargs", {}) or {}
    if not isinstance(kwargs, dict):
        raise TypeError("model.model_kwargs must be a mapping")
    return model_class(**kwargs)


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    expected_model_name: str | None = None,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        warnings.warn(
            f"Safe weights-only loading failed for legacy checkpoint {path}: "
            f"{error}. Falling back to full pickle loading; use only trusted files.",
            RuntimeWarning,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict, checkpoint_format, metadata = _extract_state_dict(payload, path)
    state_dict, stripped_prefixes = _strip_state_dict_prefixes(state_dict)
    checkpoint_hint = checkpoint_model_hint(state_dict)
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
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        hint = (
            f" The checkpoint signature looks like {checkpoint_hint}; verify "
            f"that the YAML model_name and all model hyperparameters match."
            if checkpoint_hint is not None
            else " Verify that the checkpoint and YAML use the same model and hyperparameters."
        )
        raise RuntimeError(f"Unable to load checkpoint {path}.{hint}\n{error}") from error
    inferred = _infer_checkpoint_metadata(path)
    return {
        "path": str(path.resolve()),
        "format": checkpoint_format,
        "stripped_prefixes": stripped_prefixes,
        "training_protocol": metadata.get("training_protocol", inferred["training_protocol"]),
        "selection_metric": metadata.get("selection_metric", inferred["selection_metric"]),
        "model_hint": checkpoint_hint,
    }
