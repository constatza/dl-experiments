#!/usr/bin/env python3
"""Generate every dataset referenced by one experiments registry."""

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
        help="Path to an experiments registry TOML.",
    ),
) -> None:
    """Materialize the dataset layer without starting training."""
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
