"""Model-config MLflow policy helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def normalize_model_mlflow(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Validate model-side MLflow TOML against the hard-cut flat contract."""
    del config_path
    normalized = deepcopy(raw)
    mlflow_raw = normalized.get("MLFLOW")
    if not isinstance(mlflow_raw, dict):
        return normalized

    mlflow_raw.pop("enabled", None)

    legacy_sections = [key for key in ("client", "server") if key in mlflow_raw]
    if legacy_sections:
        joined = ", ".join(legacy_sections)
        raise ValueError(
            f"Legacy MLFLOW client/server sections are not supported in model configs: {joined}."
        )

    infra_fields = [key for key in ("tracking_uri", "artifacts_destination") if key in mlflow_raw]
    if infra_fields:
        joined = ", ".join(infra_fields)
        raise ValueError(
            "Model config [MLFLOW] must not define infrastructure fields: "
            f"{joined}. Use experiments topology or runtime env instead."
        )

    return normalized
