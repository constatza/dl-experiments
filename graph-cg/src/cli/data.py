"""Data-config driven helpers for collection and generation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import tomllib

from ..constants import FILE_MODE_READ_BINARY
from ..unified_data_processing import process_config


def load_data_config(config_path: Path) -> Mapping[str, Any]:
    """Load a TOML data configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(FILE_MODE_READ_BINARY) as fh:
        return tomllib.load(fh)


def collect_data_from_config(
    config_path: Path,
    *,
    solve_systems: bool = True,
    cg_tolerance: float | None = None,
    cg_max_iters: int | None = None,
) -> Path:
    """Collect data via the unified data processing pipeline."""

    config = dict(load_data_config(config_path))
    generation = dict(config.get("generation", {}))
    generation["solve_systems"] = solve_systems
    if cg_tolerance is not None:
        generation["cg_tolerance"] = cg_tolerance
    if cg_max_iters is not None:
        generation["cg_max_iters"] = cg_max_iters
    config["generation"] = generation

    return process_config(config, config_path=config_path)


def generate_data_from_config(config_path: Path) -> Path:
    """Generate synthetic data via the unified data processing pipeline."""

    config = load_data_config(config_path)
    return process_config(config, config_path=config_path)
