#!/usr/bin/env python3
"""CLI wrapper for neuralls.workflows.training.train_model."""

from __future__ import annotations

from pathlib import Path


import typer

from neuralls.constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_PROJECT_ROOT,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
)
from neuralls.workflows.training import train_model


def main(
    config: Path = typer.Argument(
        None,
        help="Path to TOML config",
    ),
    data_config: Path = typer.Option(
        None,
        help="Path to data config providing dataset metadata",
    ),
) -> None:
    """Train model using configuration files."""
    if config is None:
        config = DEFAULT_PROJECT_ROOT / DEFAULT_MODEL_CONFIG
    if data_config is None:
        data_config = DEFAULT_PROJECT_ROOT / DEFAULT_DATA_CONFIG

    print(f"Loading configuration from: {config}")

    try:
        checkpoint_path = train_model(
            config_path=config,
            data_config_path=data_config,
        )

        print(f"\nTraining complete. Checkpoint saved to: {checkpoint_path}")

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
