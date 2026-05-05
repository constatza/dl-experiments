"""Generate one dataset from one dataset config."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from neuralls.cli.options import CaseConfigOption, EnvFileOption, ProfileOption
from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.composition.config import CASE_CONFIG_ENV_VAR, load_case_settings
from neuralls.shared.constants import EXIT_FAILURE, SYMBOL_ERROR, SYMBOL_SUCCESS


def _resolve_case_config(case_config: Path | None) -> Path | None:
    """Resolve explicit case config or NEURALLS_CASE_CONFIG override."""
    if case_config is not None:
        return case_config
    configured = os.getenv(CASE_CONFIG_ENV_VAR)
    if configured is None or not configured.strip():
        return None
    return Path(configured)


def generate_single(
    config: Path = typer.Argument(
        ...,
        help="Path to a dataset config TOML.",
    ),
    case_config: CaseConfigOption = None,
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Generate or collect one dataset from one dataset config."""
    try:
        resolved_case_config = _resolve_case_config(case_config)
        if resolved_case_config is None:
            raise ValueError(
                "This command requires a case config. Pass --case-config or set "
                "NEURALLS_CASE_CONFIG."
            )

        settings = load_case_settings(resolved_case_config, env_file, profile=profile)
        typer.echo(f"Loading data config: {config}")
        output_path = process_data_from_config(config, settings)

        typer.echo(f"\n{SYMBOL_SUCCESS} Data processing complete!")
        typer.echo(f"  Output: {output_path}")
        typer.echo("  Files:")
        for file in sorted(output_path.glob("*")):
            typer.echo(f"    - {file.name}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"\n{SYMBOL_ERROR} Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc
