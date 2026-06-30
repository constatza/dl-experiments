"""TOML loading and validation for neuralls-owned config types."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.comparison import (
    ComparisonConfig,
    parse_comparison_config,
)
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.settings import NeurallsSettings, load_case_settings


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Load TOML file as raw dict without validation."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise exc


def load_data_config(path: Path, settings: NeurallsSettings) -> DataConfigFile:
    """Load and validate a data config TOML using context-aware path expansion."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    return DataConfigFile.model_validate(raw, context=ctx.as_pydantic_context())


def load_comparison_config(path: Path, settings: NeurallsSettings) -> ComparisonConfig:
    """Load and validate a comparison TOML using context-aware path expansion."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    return parse_comparison_config(raw, context=ctx)


def load_case_config(path: Path, settings: NeurallsSettings) -> CaseConfig:
    """Load and validate the top-level case TOML."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    return CaseConfig.model_validate(raw, context=ctx.as_pydantic_context())


def load_case(path: Path, env_file: Path | None = None) -> tuple[CaseConfig, NeurallsSettings]:
    """Load one case config together with its resolved runtime settings."""
    settings = load_case_settings(path, env_file)
    return load_case_config(path, settings), settings
