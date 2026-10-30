"""Unified DrivAerML training API.

The project historically had separate ``ml.py`` and ``ml_SATO.py`` loops.
They now share the same implementation so every supported model receives the
same AMP, serialization, physical-unit validation and testing behavior.
"""

from utils.ml_SATO import (  # noqa: F401
    batch_relative_error,
    configure_amp,
    forward_model,
    select_device,
    test,
    train,
)

__all__ = [
    "train", "test", "forward_model", "configure_amp", "select_device",
    "batch_relative_error",
]
