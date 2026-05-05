#!/usr/bin/env python3
"""Train one model against one dataset config."""

from __future__ import annotations

import os
from pathlib import Path


import typer

from neuralls.shared.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
)
from neuralls.composition.experiments.training import train_model
from neuralls.platform.config.settings import CASE_CONFIG_ENV_VAR, load_case_settings


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to a model config TOML.",
    ),
    data_config: Path = typer.Option(
        ...,
        help="Path to the dataset config used for training.",
    ),
    case_config: Path | None = typer.Option(
        None,
        "--case-config",
        help="Path to case config TOML.",
    ),
    max_epochs: int | None = typer.Option(
        None,
        help=(
            "Override max training epochs. Use a multiple of "
            "every_n_epochs (default 50) if you need a checkpoint."
        ),
    ),
    env_file: Path | None = typer.Option(
        None,
        help="Optional env file to load before config resolution.",
    ),
) -> None:
    """Run the single-experiment training path.

    Use this command after `process-data` when you want to validate one
    model/dataset pair before scaling up to case-wide runs.
    """
    print(f"Loading configuration from: {config}")

    try:
        resolved_case_config = case_config
        if resolved_case_config is None:
            configured = os.getenv(CASE_CONFIG_ENV_VAR)
            if configured is not None and configured.strip():
                resolved_case_config = Path(configured)
        if resolved_case_config is None:
            raise ValueError(
                "This command requires a case config. Pass --case-config or set "
                "NEURALLS_CASE_CONFIG."
            )
        settings = load_case_settings(resolved_case_config, env_file)
        checkpoint_path = train_model(
            config_path=config,
            settings=settings,
            case_config_path=resolved_case_config,
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
