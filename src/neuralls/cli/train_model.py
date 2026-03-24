#!/usr/bin/env python3
"""Train one model against one dataset config."""

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
        help="Path to a model config TOML.",
    ),
    data_config: Path = typer.Option(
        None,
        help="Path to the dataset config used for training.",
    ),
    max_epochs: int | None = typer.Option(
        None,
        help=(
            "Override max training epochs. Use a multiple of "
            "every_n_epochs (default 50) if you need a checkpoint."
        ),
    ),
) -> None:
    """Run the single-experiment training path.

    Use this command after `process-data` when you want to validate one
    model/dataset pair before scaling up to registry-wide runs.
    """
    if config is None:
        config = DEFAULT_PROJECT_ROOT / DEFAULT_MODEL_CONFIG
    if data_config is None:
        data_config = DEFAULT_PROJECT_ROOT / DEFAULT_DATA_CONFIG

    print(f"Loading configuration from: {config}")

    try:
        checkpoint_path = train_model(
            config_path=config,
            data_config_path=data_config,
            max_epochs=max_epochs,
        )

        print(f"\nTraining complete. Checkpoint saved to: {checkpoint_path}")

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
