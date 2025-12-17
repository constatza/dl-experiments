#!/usr/bin/env python3
"""CLI wrapper for src.workflows.training.train_model."""

from __future__ import annotations

import sys
from pathlib import Path

# Add graph-cg root to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_DATA_CONFIG,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
)
from src.workflows.training import train_model


def main(
    config: Path = typer.Argument(
        None,
        help="Path to TOML config",
    ),
    data_config: Path = typer.Option(
        None,
        help="Path to data config providing dataset metadata",
    ),
):
    """Train model using configuration files."""
    graph_cg_root = Path(__file__).resolve().parent.parent

    if config is None:
        config = graph_cg_root / DEFAULT_MODEL_CONFIG
    if data_config is None:
        data_config = graph_cg_root / DEFAULT_DATA_CONFIG

    print(f"Loading configuration from: {config}")

    try:
        checkpoint_path = train_model(
            config_path=config,
            data_config_path=data_config,
        )

        print(f"\nTraining complete. Checkpoint saved to: {checkpoint_path}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
