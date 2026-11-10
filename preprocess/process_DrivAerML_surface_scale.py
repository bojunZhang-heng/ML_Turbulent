#!/usr/bin/env python3
"""Use the DrivAerML surface-coordinate min-max scale for both point sets."""

from pathlib import Path

from _surface_scale_coordinates import run_cli


if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            dataset_name="DrivAerML",
            schema_version="drivaerml-normalized-surface-scale-v3",
            default_input=Path(__file__).resolve().parent / "drivaerML_N10k",
        )
    )
