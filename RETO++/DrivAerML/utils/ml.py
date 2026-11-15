"""Unified DrivAerML training API.

The project historically had separate ``ml.py`` and ``ml_SATO.py`` loops.
They now share the same implementation so RETO and RETO_plus_v2 receive the same AMP,
serialization, physical-unit validation and testing behavior as RETO_plus.
"""

from utils.ml_SATO import (  # noqa: F401
    batch_physical_fields,
    batch_relative_error,
    checkpoint_paths,
    configure_amp,
    forward_model,
    select_device,
    test,
    train,
)

__all__ = [
    "train", "test", "forward_model", "configure_amp", "select_device",
    "batch_relative_error", "batch_physical_fields", "checkpoint_paths",
]
