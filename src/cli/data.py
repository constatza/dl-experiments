"""Data-config driven helpers for collection and generation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import tomllib
import typer

from ..constants import FILE_MODE_READ_BINARY
from ..generation import process_config


def load_data_config(config_path: Path) -> Mapping[str, Any]:
    """Load a TOML data configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(FILE_MODE_READ_BINARY) as fh:
        return tomllib.load(fh)


def _apply_generation_overrides(
    config: Mapping[str, Any],
    *,
    solve_systems: bool | None = None,
    cg_tolerance: float | None = None,
    cg_max_iters: int | None = None,
) -> dict[str, Any]:
    """Apply generation section overrides to config (immutable).

    Args:
        config: Source configuration mapping.
        solve_systems: Override for solve_systems flag.
        cg_tolerance: Override for CG tolerance.
        cg_max_iters: Override for CG max iterations.

    Returns:
        New config dict with overrides applied.
    """
    config_copy = dict(config)
    generation = dict(config_copy.get("generation", {}))

    if solve_systems is not None:
        generation["solve_systems"] = solve_systems
    if cg_tolerance is not None:
        generation["cg_tolerance"] = cg_tolerance
    if cg_max_iters is not None:
        generation["cg_max_iters"] = cg_max_iters

    config_copy["generation"] = generation
    return config_copy


def process_data_from_config(
    config_path: Path,
    *,
    solve_systems: bool | None = None,
    cg_tolerance: float | None = None,
    cg_max_iters: int | None = None,
) -> Path:
    """Unified entry point for data collection and generation.

    This function handles both data collection (from existing RHS archives)
    and synthetic data generation through the same unified pipeline.

    Args:
        config_path: Path to TOML data configuration file.
        solve_systems: Override for solve_systems flag (collection mode).
        cg_tolerance: Override for CG solver tolerance.
        cg_max_iters: Override for CG solver max iterations.

    Returns:
        Path to output dataset directory.
    """
    config = load_data_config(config_path)
    config = _apply_generation_overrides(
        config,
        solve_systems=solve_systems,
        cg_tolerance=cg_tolerance,
        cg_max_iters=cg_max_iters,
    )
    return process_config(config, config_path=config_path)


