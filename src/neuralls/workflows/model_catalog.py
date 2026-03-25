"""Model catalog helpers for MLflow registration and naming."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from datetime import datetime, UTC
from pathlib import Path

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.configuration.dataset_identity import normalize_registry_id
from dlkit.interfaces.api.functions.model_logged import build_logged_model_uri

RESERVED_ALIASES: set[str] = {"latest"}


@dataclass(frozen=True)
class RegisteredModelRecord:
    """Immutable record for a registration event."""

    name: str
    version: int
    model_uri: str
    run_id: str


def build_registered_model_name(model_id: str) -> str:
    """Build canonical registered model name."""
    return model_id


def _normalize_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and deduplicate aliases while rejecting reserved values."""
    normalized_aliases: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = normalize_registry_id(alias)
        if normalized in RESERVED_ALIASES:
            raise ValueError(f"Alias '{normalized}' is reserved and cannot be assigned.")
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_aliases.append(normalized)
    return tuple(normalized_aliases)


def _warn_existing_registered_model_name(
    *,
    client: MlflowClient,
    registered_model_name: str,
) -> None:
    """Warn when registration will append a version to an existing model."""
    try:
        client.get_registered_model(registered_model_name)
    except Exception:
        return
    logger.warning(
        "Registered model '{}' already exists. Registering a new version.",
        registered_model_name,
    )


def register_logged_model(
    *,
    run_id: str,
    registered_model_name: str,
    tracking_uri: str,
    artifact_path: str = "model",
    aliases: tuple[str, ...] = (),
    tags: Mapping[str, str] | None = None,
) -> RegisteredModelRecord:
    """Register a logged model artifact and attach aliases."""
    model_uri = build_logged_model_uri(run_id=run_id, artifact_path=artifact_path)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    _warn_existing_registered_model_name(
        client=client,
        registered_model_name=registered_model_name,
    )
    registered = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = int(registered.version)
    for alias in _normalize_aliases(aliases):
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=str(version),
        )
        resolved_version = client.get_model_version_by_alias(
            name=registered_model_name,
            alias=alias,
        )
        if int(resolved_version.version) != version:
            raise RuntimeError(
                "Alias assignment verification failed for "
                f"{registered_model_name}@{alias}: "
                f"expected version {version}, got {resolved_version.version}."
            )
    registration_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    effective_tags: dict[str, str] = dict(tags or {})
    effective_tags["registered_at"] = registration_ts
    for key, value in effective_tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=str(version),
            key=str(key),
            value=str(value),
        )

    return RegisteredModelRecord(
        name=registered_model_name,
        version=version,
        model_uri=model_uri,
        run_id=run_id,
    )


def assign_dataset_alias_to_registered_model(
    *,
    tracking_uri: str,
    registered_model_name: str,
    run_id: str,
    dataset_alias: str,
) -> int | None:
    """Assign dataset alias to an already-registered model version for a run.

    Returns the resolved model version when found, otherwise ``None``.
    """
    alias = _normalize_aliases((dataset_alias,))[0]
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matching = [mv for mv in versions if getattr(mv, "run_id", None) == run_id]
    if not matching:
        logger.warning(
            "No registered model version found for name='{}' and run_id='{}'. "
            "Skipping alias assignment for '{}'.",
            registered_model_name,
            run_id,
            alias,
        )
        return None

    selected = max(matching, key=lambda mv: int(str(mv.version)))
    if len(matching) > 1:
        logger.warning(
            "Multiple model versions found for name='{}' and run_id='{}'. "
            "Using highest version '{}'.",
            registered_model_name,
            run_id,
            selected.version,
        )

    version_str = str(selected.version)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version_str,
    )
    resolved = client.get_model_version_by_alias(
        name=registered_model_name,
        alias=alias,
    )
    resolved_version = int(str(resolved.version))
    if resolved_version != int(version_str):
        raise RuntimeError(
            "Alias assignment verification failed for "
            f"{registered_model_name}@{alias}: "
            f"expected version {version_str}, got {resolved.version}."
        )
    return resolved_version


def read_registered_model_name(model_config_path: Path) -> str | None:
    """Read the registered model name from a model config TOML.

    Reads ``[MODEL].name`` which is used for MLflow model registry lookup.

    Args:
        model_config_path: Path to the model configuration TOML file.

    Returns:
        Model name string, or ``None`` if missing or unreadable.
    """
    import tomllib

    try:
        with open(model_config_path, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError, OSError, ValueError:
        return None
    model_section = raw.get("MODEL")
    if not isinstance(model_section, dict):
        return None
    model_name = model_section.get("name")
    if not isinstance(model_name, str):
        return None
    return model_name.strip() or None
