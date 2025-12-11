#!/usr/bin/env python3
"""CLI entry point for running the graph-cg experiment matrix.

This script provides a simple command-line interface to run all experiments
defined in configs/experiments.toml. It orchestrates data generation and model
training using Prefect for caching and parallelization.

Usage:
    # Run all experiments
    uv run python graph-cg/scripts/run_experiments.py

    # Use custom experiments config
    uv run python graph-cg/scripts/run_experiments.py --config custom-experiments.toml

Benefits:
    - Config coordination: Automatic path resolution ensures each model trains
      on correct data
    - Incremental computation: Prefect caches based on input hashes (only reruns
      when configs change)
    - Parallelization: All data generation runs in parallel; training
      parallelized per data source
    - Simplicity: Model configs stay as templates, runtime merging handles
      coordination
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add graph-cg root to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.workflows.workflow_prefect import run_experiment_matrix_flow
from src.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
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
        $ uv run python graph-cg/scripts/run_experiments.py
        $ uv run python graph-cg/scripts/run_experiments.py --config custom.toml
    """
    graph_cg_root = Path(__file__).resolve().parent.parent

    # Resolve defaults
    if config is None:
        config = graph_cg_root / DEFAULT_EXPERIMENTS_CONFIG

    if not config.exists():
        print(f"Error: Config file not found: {config}")
        raise typer.Exit(code=EXIT_FAILURE)

    try:
        print(f"Running experiments from: {config}")
        if force:
            print("Force mode enabled: Ignoring caches and filesystem checks")
        print()

        # Run the experiment matrix flow
        results = run_experiment_matrix_flow(
            experiments_config_path=str(config),
            force=force,
        )

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\nCompleted {len(results)} experiments:")
        for name in results:
            print(f"  ✓ {name}")

        print("\nAll experiments completed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
