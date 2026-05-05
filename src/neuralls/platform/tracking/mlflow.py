"""Typed MLflow helpers for resolving paths and lightweight logging."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuralls.platform.config.mlflow import normalize_tracking_uri
from neuralls.platform.config.path_utils import resolve_local_path
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT

if TYPE_CHECKING:
    from mlflow import ActiveRun
else:
    ActiveRun = Any


@dataclass(frozen=True)
class MlflowPaths:
    """Resolved MLflow URIs."""

    tracking_uri: str
    artifact_uri: str


@dataclass(frozen=True)
class MlflowRunConfig:
    """Declarative MLflow run settings."""

    experiment_name: str
    run_name: str
    tags: Mapping[str, str]
    paths: MlflowPaths
    workspace_root: Path


@dataclass(frozen=True)
class MlflowRunState:
    """Active MLflow run handle."""

    run: ActiveRun
    started: bool


DEFAULT_ARTIFACT_SUBDIRS: tuple[str, ...] = (
    "checkpoints",
    "figures",
    "predictions",
    "reports",
    "metrics",
)


def _import_mlflow() -> Any:
    """Load mlflow lazily to avoid hard dependency at import time."""
    return importlib.import_module("mlflow")


def _normalize_sqlite_uri(uri: str, project_root: Path) -> str:
    """Anchor sqlite URIs to project_root when path is relative."""
    if not uri.startswith("sqlite:///"):
        return uri
    return normalize_tracking_uri(uri, config_path=project_root / "config.toml")


def resolve_mlflow_paths(
    tracking_uri: str | None,
    artifact_uri: str | None,
    project_root: Path,
    workspace: Path,
    settings: NeurallsSettings | None = None,
) -> MlflowPaths:
    """Resolve tracking/artifact URIs against project and workspace roots."""
    if tracking_uri:
        tracking = _normalize_sqlite_uri(tracking_uri, project_root)
    elif settings is not None:
        tracking = settings.mlflow_tracking_uri
    else:
        raise ValueError(
            "tracking_uri is required when NeurallsSettings is not provided."
        )
    if artifact_uri:
        artifact_target = resolve_local_path(artifact_uri, base_dir=workspace)
        artifact_root = artifact_target if artifact_target.is_absolute() else workspace / artifact_target
        return MlflowPaths(tracking_uri=tracking, artifact_uri=str(artifact_root.resolve()))
    artifact_location = (
        settings.mlflow_artifact_location if settings is not None
        else str(workspace / "mlartifacts")
    )
    return MlflowPaths(tracking_uri=tracking, artifact_uri=artifact_location)


def build_run_config(
    *,
    settings: Any,
    workspace_root: Path,
    dataset_id: str,
    model_name: str,
    session_name: str | None = None,
    enabled: bool = True,
) -> MlflowRunConfig | None:
    """Create a run config when MLflow is enabled."""
    mlflow_cfg = getattr(settings, "MLFLOW", None)
    if not enabled or mlflow_cfg is None or not getattr(mlflow_cfg, "enabled", False):
        return None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")
    paths = resolve_mlflow_paths(
        tracking_uri,
        artifact_uri,
        Path(getattr(getattr(settings, "PATHS", None), "project_root", DEFAULT_PROJECT_ROOT)),
        workspace_root,
    )
    experiment_name = getattr(mlflow_cfg, "experiment_name", None) or dataset_id
    run_name = getattr(mlflow_cfg, "run_name", None) or model_name
    tags = make_run_tags(dataset_id, model_name, session_name)
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
        paths=paths,
        workspace_root=workspace_root,
    )


def make_run_tags(
    dataset_id: str,
    model_name: str,
    session_name: str | None = None,
) -> dict[str, str]:
    """Build minimal tag set for run context."""
    tags = {"dataset": dataset_id, "model": model_name}
    if session_name:
        tags["session"] = session_name
    return tags


def ensure_experiment(name: str, paths: MlflowPaths) -> str:
    """Create or reuse an experiment and return its id."""
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(paths.tracking_uri)
    existing = mlflow.get_experiment_by_name(name)
    if existing:
        return existing.experiment_id
    return mlflow.create_experiment(name=name, artifact_location=paths.artifact_uri)


def start_run_if_needed(
    exp_id: str,
    run_name: str,
    tags: Mapping[str, str] | None = None,
) -> tuple[ActiveRun, bool]:
    """Reuse active run or start a new one with tags."""
    mlflow = _import_mlflow()
    active = mlflow.active_run()
    if active:
        return active, False
    run = mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=dict(tags or {}))
    return run, True


def open_run(config: MlflowRunConfig) -> MlflowRunState:
    """Start or reuse a run and return its state."""
    exp_id = ensure_experiment(config.experiment_name, config.paths)
    run, started = start_run_if_needed(exp_id, config.run_name, config.tags)
    return MlflowRunState(run=run, started=started)


def collect_artifacts(workspace: Path, allowlist: Sequence[str]) -> list[Path]:
    """Return existing artifact paths under workspace using an allowlist."""
    root = Path(workspace)
    return [path for name in allowlist if (path := root / name).exists()]


def log_artifacts(run: ActiveRun, artifacts: Sequence[Path]) -> None:
    """Log directories/files to MLflow for the given run."""
    if not artifacts:
        return
    mlflow = _import_mlflow()
    for path in artifacts:
        mlflow.log_artifacts(str(path), run_id=run.info.run_id)


def log_metrics(run: ActiveRun, metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow for the given run."""
    if not metrics:
        return
    mlflow = _import_mlflow()
    mlflow.log_metrics(dict(metrics), step=step, run_id=run.info.run_id)


def finalize_run(
    state: MlflowRunState | None,
    *,
    metrics: Mapping[str, float] | None = None,
    workspace_root: Path,
    allowlist: Sequence[str] = DEFAULT_ARTIFACT_SUBDIRS,
    failed: bool = False,
) -> None:
    """Upload artifacts/metrics and end run if we started it."""
    if state is None:
        return
    artifacts = collect_artifacts(workspace_root, allowlist)
    log_artifacts(state.run, artifacts)
    if metrics:
        log_metrics(state.run, metrics)
    if state.started:
        _import_mlflow().end_run(status="FAILED" if failed else "FINISHED")
