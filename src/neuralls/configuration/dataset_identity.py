"""Dataset identity resolution helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping

from neuralls.configuration.data_models import DataConfigFile

_MLFLOW_ALIAS_PATTERN = re.compile(r"^[\w\-]+$")


@dataclass(frozen=True)
class DatasetIdentity:
    """Resolved dataset identity with source traceability."""

    name: str
    registry_alias: str
    source: Literal["id"]


def _coerce_non_empty_str(value: Any) -> str | None:
    """Return stripped string value when non-empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def normalize_registry_alias(alias: str) -> str:
    """Normalize user-facing alias syntax to MLflow-safe alias.

    Supports optional ``@`` prefix for user-facing references.
    """
    normalized = alias.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:].strip()
    if not normalized:
        raise ValueError("Registry alias cannot be empty.")
    if not _MLFLOW_ALIAS_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Registry alias must match ^[\\w\\-]+$ (letters, numbers, underscore, hyphen)."
        )
    return normalized


def _require_dataset_name(configured_dataset: str | None) -> str:
    """Require explicit dataset name from top-level id."""
    if configured_dataset is None:
        raise ValueError(
            "Missing required top-level 'id' in data config. "
            "Dataset name must be explicitly defined."
        )
    return configured_dataset


def resolve_dataset_identity(
    *,
    data_cfg: DataConfigFile,
    config_path: Path | str | None = None,
) -> DatasetIdentity:
    """Resolve dataset identity from validated data config + config path."""
    _ = config_path
    configured = _coerce_non_empty_str(data_cfg.id)
    dataset_name = _require_dataset_name(configured)
    return DatasetIdentity(
        name=dataset_name,
        registry_alias=normalize_registry_alias(dataset_name),
        source="id",
    )


def resolve_dataset_identity_from_mapping(
    *,
    config: Mapping[str, Any],
    config_path: Path | str | None = None,
) -> DatasetIdentity:
    """Resolve dataset identity from raw config mapping + config path."""
    _ = config_path
    configured = _coerce_non_empty_str(config.get("id"))
    dataset_name = _require_dataset_name(configured)
    return DatasetIdentity(
        name=dataset_name,
        registry_alias=normalize_registry_alias(dataset_name),
        source="id",
    )
