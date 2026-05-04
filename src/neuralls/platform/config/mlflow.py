"""MLflow config validation and runtime environment helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dlkit.infrastructure.io import url_resolver

from neuralls.platform.config.path_utils import build_sqlite_uri, resolve_local_path


def is_sqlite_tracking_uri(uri: str) -> bool:
    """Return True when tracking URI uses the sqlite scheme."""
    return url_resolver.scheme(uri.strip()) == "sqlite"


def _resolve_sqlite_db_path(tracking_uri: str, config_path: Path | None = None) -> Path:
    """Resolve sqlite DB path from tracking URI to an absolute local path."""
    if not is_sqlite_tracking_uri(tracking_uri):
        raise ValueError(f"Expected sqlite tracking URI, got: {tracking_uri!r}")

    uri = tracking_uri.strip()
    if uri.endswith(":memory:"):
        raise ValueError("MLflow sqlite tracking URIs must include a filesystem database path.")
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    return url_resolver.resolve_local_uri(uri, base_dir.resolve())


def build_sqlite_tracking_uri(db_path: Path) -> str:
    """Build a normalized sqlite tracking URI from a local DB path."""
    return build_sqlite_uri(db_path)


def normalize_tracking_uri(
    tracking_uri: str,
    *,
    config_path: Path | None = None,
) -> str:
    """Validate and normalize a runtime MLflow tracking URI."""
    uri = tracking_uri.strip()
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https", "sqlite"}:
        raise ValueError("MLflow tracking_uri must use one of: http://, https://, sqlite:///.")

    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ValueError(
                f"MLflow tracking_uri must include a host for {parsed.scheme} URIs: {uri!r}"
            )
        return uri.rstrip("/")

    db_path = _resolve_sqlite_db_path(uri, config_path)
    return build_sqlite_tracking_uri(db_path)


def normalize_artifacts_destination(
    artifacts_destination: str,
    *,
    config_path: Path | None = None,
) -> str:
    """Normalize an artifacts destination to an absolute filesystem path."""
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    return str(resolve_local_path(artifacts_destination, base_dir=base_dir))


def derive_sqlite_artifacts_destination(tracking_uri: str) -> str:
    """Derive artifacts destination as a sibling `mlartifacts/` directory to sqlite DB."""
    normalized = normalize_tracking_uri(tracking_uri)
    if not is_sqlite_tracking_uri(normalized):
        raise ValueError(
            f"Cannot derive sqlite artifacts destination from non-sqlite URI: {tracking_uri!r}"
        )
    db_path = _resolve_sqlite_db_path(normalized)
    mlruns_dir = db_path.parent
    output_root = mlruns_dir.parent if mlruns_dir.name == "mlruns" else mlruns_dir
    return str((output_root / "mlartifacts").resolve())


def derive_output_root_from_tracking_uri(
    tracking_uri: str,
    *,
    config_path: Path | None = None,
) -> Path | None:
    """Derive output root from a sqlite tracking URI, if applicable."""
    normalized = normalize_tracking_uri(tracking_uri, config_path=config_path)
    if not is_sqlite_tracking_uri(normalized):
        return None
    db_path = _resolve_sqlite_db_path(normalized)
    mlruns_dir = db_path.parent.resolve()
    return mlruns_dir.parent if mlruns_dir.name == "mlruns" else mlruns_dir


def build_mlflow_environment(
    *,
    tracking_uri: str,
    artifacts_destination: str | None = None,
    config_path: Path | None = None,
) -> dict[str, str]:
    """Build validated MLflow env vars for runtime execution."""
    normalized_tracking = normalize_tracking_uri(
        tracking_uri,
        config_path=config_path,
    )
    normalized_artifacts = artifacts_destination
    if normalized_artifacts is None and is_sqlite_tracking_uri(normalized_tracking):
        normalized_artifacts = derive_sqlite_artifacts_destination(normalized_tracking)
    elif normalized_artifacts is not None:
        normalized_artifacts = normalize_artifacts_destination(
            normalized_artifacts,
            config_path=config_path,
        )

    env = {"MLFLOW_TRACKING_URI": normalized_tracking}
    if normalized_artifacts:
        env["MLFLOW_ARTIFACT_URI"] = normalized_artifacts
    return env


@contextmanager
def scoped_mlflow_environment(overrides: Mapping[str, str | None]) -> Iterator[None]:
    """Temporarily apply MLflow environment variable overrides."""
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def normalize_model_mlflow(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Validate model-side MLflow TOML against the hard-cut flat contract."""
    normalized = deepcopy(raw)
    mlflow_raw = normalized.get("MLFLOW")
    if not isinstance(mlflow_raw, dict):
        return normalized

    # dlkit now treats the presence of [MLFLOW] as enabling tracking.
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
