#!/usr/bin/env python3
"""CLI: generate all datasets listed in [[datasets]] of an experiments config.

Usage::

    uv run generate-multiple configs/experiments.toml
"""

from __future__ import annotations

from pathlib import Path

import typer

from neuralls.configuration.experiments import ExperimentsConfig
from neuralls.constants import EXIT_FAILURE, EXIT_KEYBOARD_INTERRUPT
from neuralls.io.toml_loader import load_raw_toml
from neuralls.workflows.multi_generation import generate_batch


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to experiments TOML (e.g. configs/experiments.toml)",
    ),
) -> None:
    """Generate all datasets listed in [[datasets]] of the experiments config."""
    try:
        raw = load_raw_toml(config)
        cfg = ExperimentsConfig.model_validate(raw)
        results = generate_batch(cfg=cfg, configs_dir=config.resolve().parent)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error during batch generation: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE)

    for r in results:
        typer.echo(f"[{r.dataset_id}] → {r.output_dir}")


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
