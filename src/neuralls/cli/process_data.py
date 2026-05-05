#!/usr/bin/env python3
"""Build one processed dataset from a dataset config."""

from __future__ import annotations

from pathlib import Path
import os


import typer

from neuralls.shared.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_SUCCESS,
    SYMBOL_ERROR,
)
from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.platform.config.settings import CASE_CONFIG_ENV_VAR, load_case_settings


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to a dataset config TOML.",
    ),
    case_config: Path | None = typer.Option(
        None,
        "--case-config",
        help="Path to case config TOML.",
    ),
    env_file: Path | None = typer.Option(
        None,
        help="Optional env file to load before config resolution.",
    ),
) -> None:
    """Generate or collect one dataset.

    Start here when you want to inspect a single dataset config before running
    case-wide training or comparison commands.
    """
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
        print(f"Loading data config: {config}")
        output_path = process_data_from_config(config, settings)

        print(f"\n{SYMBOL_SUCCESS} Data processing complete!")
        print(f"  Output: {output_path}")
        print("  Files:")
        for file in sorted(output_path.glob("*")):
            print(f"    - {file.name}")

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"\n{SYMBOL_ERROR} Error: {e}")
        raise typer.Exit(code=EXIT_FAILURE)


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
