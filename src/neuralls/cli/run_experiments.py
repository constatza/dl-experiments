#!/usr/bin/env python3
"""CLI entry point for running the graph-cg experiment matrix without Prefect."""

from __future__ import annotations

import sys
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
        help="Path to experiments definition file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-run all experiments (ignore filesystem checks and cache)",
    ),
):
    """Run all experiments defined in configs/experiments.toml.

    This command orchestrates the entire experiment matrix:
    1. Reads experiment definitions from configs/experiments.toml
    2. Generates all unique datasets in parallel (with caching)
    3. Trains all models in parallel

    Prefect automatically handles:
    - Caching (data generation runs once per unique config)
    - Parallel execution of independent tasks
    - Incremental computation (only reruns changed configs)

    Example:
        $ uv run python src/neuralls/cli/run_experiments.py
        $ uv run python src/neuralls/cli/run_experiments.py --config custom.toml
    """
    # Resolve defaults
    if config is None:
        config = DEFAULT_PROJECT_ROOT / DEFAULT_EXPERIMENTS_CONFIG

    if not config.exists():
        print(f"Error: Config file not found: {config}")
        raise typer.Exit(code=EXIT_FAILURE)

    try:
        print(f"Running experiments from: {config}")
        if force:
            print("Force mode enabled: ignoring caches and filesystem checks")
        print()

        results = run_experiment_matrix(
            experiments_config_path=config,
            force=force,
            project_root=DEFAULT_PROJECT_ROOT,
        )

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\nCompleted {len(results)} experiments:")
        for res in results:
            status = "✓" if res.is_success else "✗"
            detail = f" ({res.error})" if res.error else ""
            print(f"  {status} {res.experiment_id}{detail}")

        failures = [r for r in results if not r.is_success]
        if failures:
            raise typer.Exit(code=EXIT_FAILURE)

    except Exception as e:
        print(f"\nError: {e}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
