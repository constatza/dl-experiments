"""Utilities for storing and retrieving extra feature names from MLflow run tags."""

from __future__ import annotations

from collections.abc import Iterable

from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.model_registry import build_registered_model_name

EXTRA_FEATURE_NAMES_TAG: str = "neuralls.extra_feature_names"


def fetch_extra_feature_names(run_id: str, *, client: MlflowClient) -> tuple[str, ...]:
    """Fetch extra feature names logged during training for a given MLflow run.

    Reads the ``neuralls.extra_feature_names`` tag from the run, splits on
    comma, filters empty strings, and returns the result as a tuple.

    Args:
        run_id: MLflow run ID from the training run.
        client: Configured ``MlflowClient`` pointing at the tracking server.

    Returns:
        Tuple of extra feature names, or empty tuple if the tag is absent or empty.
    """
    run = client.get_run(run_id)
    raw = run.data.tags.get(EXTRA_FEATURE_NAMES_TAG, "")
    return tuple(name for name in raw.split(",") if name)


def log_extra_feature_names_tag(
    tracking_uri: str,
    run_id: str,
    extra_names: Iterable[str],
) -> None:
    """Log extra feature names as a run tag on an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run ID to tag.
        extra_names: Set of extra feature names declared in the model TOML.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    tag_value = ",".join(sorted(extra_names))
    client.set_tag(run_id, EXTRA_FEATURE_NAMES_TAG, tag_value)


def _lookup_run_id_for_model(entry_id: str, client: MlflowClient) -> str | None:
    """Look up the latest registered model version run ID for an experiment entry.

    Args:
        entry_id: Experiment registry ID used as the registered model name.
        client: Configured MLflow client.

    Returns:
        MLflow run ID of the latest model version, or None if unavailable.
    """
    try:
        versions = client.search_model_versions(f"name='{build_registered_model_name(entry_id)}'")
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))
        return latest.run_id or None
    except Exception:  # noqa: BLE001
        return None


def fetch_extra_input_names_for_model(entry_id: str, client: MlflowClient) -> tuple[str, ...]:
    """Fetch extra input names from the training run tag for an experiment entry.

    Looks up the latest registered model version for ``entry_id``, then reads
    the ``neuralls.extra_feature_names`` tag from the associated training run.

    Args:
        entry_id: Experiment registry ID used as the registered model name.
        client: Configured MLflow client.

    Returns:
        Tuple of extra input names, or empty tuple if unavailable.
    """
    run_id = _lookup_run_id_for_model(entry_id, client)
    if run_id is None:
        return ()
    try:
        return fetch_extra_feature_names(run_id, client=client)
    except Exception:  # noqa: BLE001
        return ()


__all__ = [
    "EXTRA_FEATURE_NAMES_TAG",
    "fetch_extra_feature_names",
    "fetch_extra_input_names_for_model",
    "log_extra_feature_names_tag",
]
