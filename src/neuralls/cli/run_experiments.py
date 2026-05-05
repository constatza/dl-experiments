#!/usr/bin/env python3
"""Run the case-defined data generation and training matrix."""

from __future__ import annotations

from pathlib import Path


import typer

from neuralls.composition.experiments.training_batch import run_experiment_matrix
from neuralls.platform.config.settings import load_case_settings
from neuralls.shared.constants import (
    DEFAULT_PROJECT_ROOT,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
)


def main(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a case config TOML.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-train all models (ignore existing checkpoints)",
    ),
    max_epochs: int | None = typer.Option(
        None,
        help="Override max training epochs (for quick tests)",
    ),
    env_file: Path | None = typer.Option(
        None,
        help="Optional env file to load before config resolution.",
    ),
) -> None:
    """Generate datasets and train all experiments in one case config.

    This is the basic batch workflow:
    1. load one case config
    2. build each referenced dataset once
    3. train each experiment unless a checkpoint already exists

    Use `compare-all` after this command when you want solver benchmarking.
    """
    if not config.exists():
        print(f"Error: Config file not found: {config}")
        raise typer.Exit(code=EXIT_FAILURE)

    try:
        settings = load_case_settings(config, env_file)
        print(f"Training experiments from case config: {config}")
        if force:
            print("Force mode enabled: will retrain even if checkpoints exist")
        print()

        results = run_experiment_matrix(
            experiments_config_path=config,
            settings=settings,
            force=force,
            project_root=DEFAULT_PROJECT_ROOT,
            max_epochs=max_epochs,
        )

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\nCompleted {len(results)} experiments:")
        for res in results:
            status = "✓" if res.is_success else "✗"
            detail = f" ({res.error})" if res.error else ""
            label = res.experiment_display_name
            if label != res.experiment_id:
                label = f"{label} [{res.experiment_id}]"
            print(f"  {status} {label}{detail}")

        failures = [r for r in results if not r.is_success]
        if failures:
            raise typer.Exit(code=EXIT_FAILURE)

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"\nError: {e}")
        raise typer.Exit(code=EXIT_FAILURE)


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
