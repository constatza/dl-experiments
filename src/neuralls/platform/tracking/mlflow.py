"""Typed MLflow helpers for resolving paths and lightweight logging."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import ActiveRun

from neuralls.platform.config.resolution import (
    MlflowPaths,
    build_sqlite_tracking_uri,
    resolve_mlflow_paths,
    to_mlflow_artifact_location,
)
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT


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
    paths_cfg = getattr(settings, "PATHS", None)
    output_dir = getattr(paths_cfg, "output_dir", None)
    default_tracking_uri = None
    default_artifact_location = None
    if output_dir:
        output_root = Path(output_dir).resolve()
        default_tracking_uri = build_sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
        default_artifact_location = str((output_root / "mlartifacts").resolve())
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")
    paths = resolve_mlflow_paths(
        tracking_uri=tracking_uri,
        artifact_uri=artifact_uri,
        project_root=Path(getattr(paths_cfg, "project_root", DEFAULT_PROJECT_ROOT)),
        workspace_root=workspace_root,
        default_tracking_uri=default_tracking_uri,
        default_artifact_location=default_artifact_location,
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
    mlflow.set_tracking_uri(paths.tracking_uri)
    existing = mlflow.get_experiment_by_name(name)
    if existing:
        return existing.experiment_id
    return mlflow.create_experiment(
        name=name,
        artifact_location=to_mlflow_artifact_location(paths.artifact_uri),
    )


def start_run_if_needed(
    exp_id: str,
    run_name: str,
    tags: Mapping[str, str] | None = None,
) -> tuple[ActiveRun, bool]:
    """Reuse active run or start a new one with tags."""
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
    for path in artifacts:
        mlflow.log_artifacts(str(path), run_id=run.info.run_id)


def log_metrics(run: ActiveRun, metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow for the given run."""
    if not metrics:
        return
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
        mlflow.end_run(status="FAILED" if failed else "FINISHED")
