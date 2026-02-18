"""Data-config driven helpers for collection and generation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping

import tomllib

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
) -> dict[str, Any]:
    """Apply generation section overrides to config (immutable).

    Args:
        config: Source configuration mapping.

    Returns:
        New config dict with overrides applied.
    """
    config_copy = dict(config)
    generation = dict(config_copy.get("generation", {}))

    config_copy["generation"] = generation
    return config_copy


def process_data_from_config(
    config_path: Path,
) -> Path:
    """Unified entry point for data collection and generation.

    This function handles both data collection (from existing RHS archives)
    and synthetic data generation through the same unified pipeline.

    Args:
        config_path: Path to TOML data configuration file.

    Returns:
        Path to output dataset directory.
    """
    config = load_data_config(config_path)
    config = _apply_generation_overrides(config)
    return process_config(config, config_path=config_path)
