"""TOML parsing and Pydantic validation module.

This module is responsible for loading TOML files and validating them
using Pydantic models, providing clear error messages on validation failures.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from dlkit.tools.config import TrainingWorkflowSettings
from dlkit.tools.io import load_settings

from neuralls.configuration.comparison import ComparisonConfig, parse_comparison_config
from neuralls.configuration.data_models import DataConfigFile
from neuralls.configuration.mlflow_normalization import normalize_model_mlflow


def load_model_config(path: Path) -> TrainingWorkflowSettings:
    """Load and validate linear.toml file.

    Args:
        path: Path to model configuration TOML file.

    Returns:
        Validated ModelConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    normalized = normalize_model_mlflow(raw, path)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=True) as tmp:
        tomli_w.dump(normalized, tmp)
        tmp.flush()
        return load_settings(Path(tmp.name))


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


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Load and validate comparison TOML file.

    Args:
        path: Path to comparison configuration TOML file.

    Returns:
        Parsed ComparisonConfig dataclass.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    return parse_comparison_config(raw)


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
