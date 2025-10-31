#!/usr/bin/env python3
"""CLI wrapper for src.cli.training.train_model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from src.constants import EXIT_FAILURE, EXIT_KEYBOARD_INTERRUPT
from src.cli.training import train_model


def main(
    config: Path = typer.Argument(
        Path(__file__).parent / "configs" / "ffnn.toml",
        help="Path to TOML config",
    ),
    data_config: Path = typer.Option(
        Path(__file__).parent / "data-configs" / "collect-504.toml",
        help="Path to data config providing dataset metadata",
    ),
    features: Path | None = typer.Option(
        None, help="Override path to features (RHS) .npy"
    ),
    targets: Path | None = typer.Option(
        None, help="Override path to targets (solution) .npy"
    ),
    out_dir: Path | None = typer.Option(
        None, help="Override Trainer default_root_dir and checkpoint dir"
    ),
    accelerator: str | None = typer.Option(
        None, help="Override accelerator: cpu|gpu|auto|tpu"
    ),
):
    """Train model with optional dataset/output overrides."""
    print(f"Loading configuration from: {config}")

    try:
        checkpoint_path = train_model(
            config_path=config,
            data_config_path=data_config,
            features_path=features,
            targets_path=targets,
            output_dir=out_dir,
            accelerator=accelerator,
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
