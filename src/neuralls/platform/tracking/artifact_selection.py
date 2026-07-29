"""Artifact-path selection policy for MLflow run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.artifact_access import normalize_artifact_path

CHECKPOINT_ARTIFACT_DIR = "checkpoints"
CHECKPOINT_FILE_EXTENSION = ".ckpt"
CONFIG_ARTIFACT_DIR = "config"
SPLIT_ARTIFACT_DIR = "splits"
SPLIT_FILE_EXTENSION = ".json"


def list_artifact_files(
    client: MlflowClient,
    *,
    run_id: str,
    artifact_dir: str,
) -> tuple[str, ...]:
    """Return direct file artifact paths listed under one artifact directory."""
    normalized_dir = normalize_artifact_path(artifact_dir)
    artifacts = client.list_artifacts(run_id, normalized_dir)
    return tuple(artifact.path for artifact in artifacts if not _artifact_is_dir(artifact))


def has_artifact_dir(
    client: MlflowClient,
    *,
    run_id: str,
    artifact_dir: str,
) -> bool:
    """Return whether MLflow lists anything under an artifact directory."""
    normalized_dir = normalize_artifact_path(artifact_dir)
    return bool(client.list_artifacts(run_id, normalized_dir))


def select_split_artifact_path(
    client: MlflowClient,
    *,
    run_id: str,
    artifact_dir: str = SPLIT_ARTIFACT_DIR,
) -> str:
    """Return the single split JSON artifact path for a run."""
    split_files = _files_with_suffix(
        list_artifact_files(client, run_id=run_id, artifact_dir=artifact_dir),
        suffix=SPLIT_FILE_EXTENSION,
    )
    match split_files:
        case (split_file,):
            return split_file
        case ():
            raise FileNotFoundError(
                f"Training run '{run_id}' has no split JSON artifact under '{artifact_dir}/'."
            )
        case _:
            joined = ", ".join(Path(path).name for path in split_files)
            raise ValueError(
                f"Training run '{run_id}' has multiple split JSON artifacts under "
                f"'{artifact_dir}/': {joined}."
            )


def select_config_artifact_dir(
    client: MlflowClient,
    *,
    run_id: str,
    artifact_dir: str = CONFIG_ARTIFACT_DIR,
) -> str | None:
    """Return the config artifact directory when present."""
    normalized_dir = normalize_artifact_path(artifact_dir)
    if not has_artifact_dir(client, run_id=run_id, artifact_dir=normalized_dir):
        return None
    return normalized_dir


def _files_with_suffix(paths: tuple[str, ...], *, suffix: str) -> tuple[str, ...]:
    return tuple(path for path in sorted(paths) if Path(path).suffix == suffix)


def _artifact_is_dir(artifact: Any) -> bool:
    return bool(getattr(artifact, "is_dir", False))


__all__ = [
    "CHECKPOINT_ARTIFACT_DIR",
    "CHECKPOINT_FILE_EXTENSION",
    "CONFIG_ARTIFACT_DIR",
    "SPLIT_ARTIFACT_DIR",
    "SPLIT_FILE_EXTENSION",
    "has_artifact_dir",
    "list_artifact_files",
    "select_config_artifact_dir",
    "select_split_artifact_path",
]
