#!/usr/bin/env python3
"""CLI wrapper for PCA preconditioner training."""

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
    SYMBOL_CHECKMARK,
)
from src.cli.training import train_pca_preconditioner


def main(
    config: Path = typer.Option(
        None, help="Path to TOML config"
    ),
    data_config: Path = typer.Option(
        None,
        help="Path to data config providing dataset metadata",
    ),
    n_components: int = typer.Option(..., help="Number of PCA components"),
    solution_samples: Path | None = typer.Option(
        None, help="Override solution samples path"
    ),
    output: Path | None = typer.Option(None, help="Override output path"),
    normalize: bool = typer.Option(True, help="Normalize solutions before PCA"),
):
    """Train PCA preconditioner from solution samples."""
    graph_cg_root = Path(__file__).resolve().parent.parent

    # Resolve defaults
    if config is None:
        config = graph_cg_root / DEFAULT_MODEL_CONFIG
    if data_config is None:
        data_config = graph_cg_root / DEFAULT_DATA_CONFIG

    try:
        output_path, plot_path = train_pca_preconditioner(
            config_path=config,
            data_config_path=data_config,
            n_components=n_components,
            solution_samples_path=solution_samples,
            output_path=output,
            normalize=normalize,
        )
        print(f"\n{SYMBOL_CHECKMARK} PCA training complete!")
        print(f"  Model saved to: {output_path}")
        print(f"  Variance plot: {plot_path}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
