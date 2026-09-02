"""TOML loading and validation for neuralls-owned config types."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from neuralls.platform.config.context import ConfigContext, expand_config_path
from neuralls.platform.config.models.comparison import (
    ComparisonConfig,
    parse_comparison_config,
)
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.config.models.experiments import CaseConfig, SharedTrackingSettings
from neuralls.platform.config.settings import NeurallsSettings, load_case_settings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT

_DEFAULT_TRACKING_TOML = DEFAULT_PROJECT_ROOT / "configs" / "tracking.toml"


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Load TOML file as raw dict without validation."""
    with open(path, "rb") as f:
        return tomllib.load(f)


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


def _fill_missing_dataset_ids(raw: dict[str, Any], ctx: ConfigContext) -> None:
    """Default missing [[datasets]] entry ids from their referenced dataset config's own id.

    A [[datasets]] entry's ``id`` is only a case-registry lookup key; the dataset
    config's own ``id`` is what generation actually names the processed directory
    with. Reading it here (instead of requiring it hand-typed twice) makes the
    dataset config the single source of truth for entries that don't need a
    distinct local alias. Mutates ``raw["datasets"]`` in place; entries with an
    explicit non-blank id are left untouched.
    """
    datasets = raw.get("datasets")
    if not isinstance(datasets, list):
        return
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        existing_id = entry.get("id")
        if isinstance(existing_id, str) and existing_id.strip():
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        dataset_path = Path(expand_config_path(raw_path, ctx))
        dataset_raw = load_raw_toml(dataset_path)
        dataset_id = dataset_raw.get("id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError(
                f"[[datasets]] entry with path '{raw_path}' has no 'id' and its dataset "
                f"config at '{dataset_path}' also has no 'id'. Set one explicitly."
            )
        entry["id"] = dataset_id


def load_case_config(path: Path, settings: NeurallsSettings) -> CaseConfig:
    """Load and validate the top-level case TOML."""
    raw = load_raw_toml(path)
    ctx = ConfigContext(config_path=path.resolve(), settings=settings)
    _fill_missing_dataset_ids(raw, ctx)
    return CaseConfig.model_validate(raw, context=ctx.as_pydantic_context())


def load_case(path: Path, env_file: Path | None = None) -> tuple[CaseConfig, NeurallsSettings]:
    """Load one case config together with its resolved runtime settings."""
    settings = load_case_settings(path, env_file)
    return load_case_config(path, settings), settings


def load_tracking_config(path: Path | None = None) -> SharedTrackingSettings | None:
    """Load the shared dlkit tracking config from configs/tracking.toml [tracking] section.

    Args:
        path: Optional explicit path. Defaults to ``configs/tracking.toml`` at project root.

    Returns:
        Validated ``SharedTrackingSettings`` when the file exists and has a [tracking] section,
        ``None`` if the file is missing or has no [tracking] section.
    """
    resolved = path or _DEFAULT_TRACKING_TOML
    if not resolved.exists():
        return None
    raw = load_raw_toml(resolved)
    section = raw.get("tracking")
    if not section:
        return None
    return SharedTrackingSettings.model_validate(section)
