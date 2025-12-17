"""TOML parsing and Pydantic validation module.

This module is responsible for loading TOML files and validating them
using Pydantic models, providing clear error messages on validation failures.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..data_models import DataConfigFile
from ..models import ModelConfigFile
from ..solver_models import SolverConfigFile


class ConfigLoadError(Exception):
    """Configuration loading or validation failed."""

    pass


def load_model_config(path: Path) -> ModelConfigFile:
    """Load and validate model.toml file.

    Args:
        path: Path to model configuration TOML file.

    Returns:
        Validated ModelConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigLoadError(f"Failed to load model config {path}: {exc}") from exc

    try:
        return ModelConfigFile(**raw)
    except ValidationError as exc:
        raise ConfigLoadError(
            f"Invalid model config {path}:\n{exc}"
        ) from exc


def load_data_config(path: Path) -> DataConfigFile:
    """Load and validate data config TOML file.

    Args:
        path: Path to data configuration TOML file.

    Returns:
        Validated DataConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigLoadError(f"Failed to load data config {path}: {exc}") from exc

    try:
        return DataConfigFile(**raw)
    except ValidationError as exc:
        raise ConfigLoadError(
            f"Invalid data config {path}:\n{exc}"
        ) from exc


def load_solver_config(path: Path) -> SolverConfigFile:
    """Load and validate solver.toml file.

    Args:
        path: Path to solver configuration TOML file.

    Returns:
        Validated SolverConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigLoadError(f"Failed to load solver config {path}: {exc}") from exc

    try:
        return SolverConfigFile(**raw)
    except ValidationError as exc:
        raise ConfigLoadError(
            f"Invalid solver config {path}:\n{exc}"
        ) from exc


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
        raise ConfigLoadError(f"Failed to load TOML {path}: {exc}") from exc
