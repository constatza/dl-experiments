#!/usr/bin/env python3
"""CLI entry point for running the graph-cg experiment matrix without Prefect."""

from __future__ import annotations

from pathlib import Path


import typer

from neuralls.workflows.runner import run_experiment_matrix
from neuralls.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    DEFAULT_PROJECT_ROOT,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
)


def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to experiments definition file (default: configs/experiments.toml)",
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
) -> None:
    """Train all models defined in experiments.toml.

    This command:
    1. Reads experiment definitions from configs/experiments.toml
    2. Generates all unique datasets (with caching)
    3. Trains all models (skips if checkpoint exists, unless --force)

    For solver comparison after training, use:
        $ uv run train-multiple configs/experiments.toml
        $ uv run compare-preconditioners configs/experiments.toml

    Example:
        # Train all experiments
        $ uv run python src/neuralls/cli/run_experiments.py

        # Force retrain even if checkpoints exist
        $ uv run python src/neuralls/cli/run_experiments.py --force

        # Use custom experiments file
        $ uv run python src/neuralls/cli/run_experiments.py --config custom.toml
    """
    # Resolve defaults
    if config is None:
        config = DEFAULT_PROJECT_ROOT / DEFAULT_EXPERIMENTS_CONFIG

    if not config.exists():
        print(f"Error: Config file not found: {config}")
        raise typer.Exit(code=EXIT_FAILURE)

    try:
        print(f"Training experiments from: {config}")
        if force:
            print("Force mode enabled: will retrain even if checkpoints exist")
        print()

        results = run_experiment_matrix(
            experiments_config_path=config,
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
