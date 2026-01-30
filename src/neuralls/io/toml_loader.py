"""TOML parsing and Pydantic validation module.

This module is responsible for loading TOML files and validating them
using Pydantic models, providing clear error messages on validation failures.
"""

from __future__ import annotations
from dlkit.tools.config import TrainingWorkflowSettings
from dlkit.tools.io import load_settings


import tomllib
from pathlib import Path
from typing import Any

from neuralls.configuration.data_models import DataConfigFile
from neuralls.configuration.comparison import ComparisonConfig


def load_model_config(path: Path) -> TrainingWorkflowSettings:
    """Load and validate linear.toml file.

    Args:
        path: Path to model configuration TOML file.

    Returns:
        Validated ModelConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    return load_settings(path)


def load_data_config(path: Path) -> DataConfigFile:
    """Load and validate data config TOML file.

    Args:
        path: Path to data configuration TOML file.

    Returns:
        Validated DataConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    return DataConfigFile(**raw)


def load_solver_config(path: Path) -> ComparisonConfig:
    """Load and validate solver.toml file.

    Args:
        path: Path to solver configuration TOML file.

    Returns:
        Validated SolverConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    return ComparisonConfig(**raw)


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Load TOML file as raw dictionary without validation.

    This function is provided for legacy compatibility and special cases
    where Pydantic validation is not desired.

    Args:
        path: Path to TOML file.

    Returns:
        Raw dictionary from TOML file.

    Raises:
        ConfigLoadError: If file cannot be read or parsed.
    """
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise exc
