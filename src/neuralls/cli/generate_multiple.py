#!/usr/bin/env python3
"""Generate every dataset referenced by one case config."""

from __future__ import annotations

from pathlib import Path

import typer

from neuralls.shared.constants import EXIT_FAILURE, EXIT_KEYBOARD_INTERRUPT
from neuralls.composition.experiments.assembler import load_validated_case_config
from neuralls.composition.generation.multi_generation import generate_batch
from neuralls.platform.config.settings import load_case_settings


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to a case config TOML.",
    ),
    env_file: Path | None = typer.Option(
        None,
        help="Optional env file to load before config resolution.",
    ),
) -> None:
    """Materialize the dataset layer without starting training."""
    try:
        settings = load_case_settings(config, env_file)
        cfg, _ = load_validated_case_config(config, settings)
        results = generate_batch(cfg=cfg, configs_dir=config.resolve().parent, settings=settings)
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
