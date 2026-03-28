from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from neuralls.platform.tracking.mlflow import (
    DEFAULT_ARTIFACT_SUBDIRS,
    MlflowPaths,
    MlflowRunConfig,
    collect_artifacts,
    ensure_experiment,
    finalize_run,
    log_artifacts,
    log_metrics,
    open_run,
    resolve_mlflow_paths,
    start_run_if_needed,
)


class DummyRun:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)


class DummyMlflow:
    def __init__(self) -> None:
        self.experiments: dict[str, SimpleNamespace] = {}
        self.logged_artifacts: list[tuple[str | None, str]] = []
        self.logged_metrics: list[tuple[str | None, dict[str, float], int | None]] = []
        self.tracking_uri: str | None = None
        self.active: DummyRun | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        return self.experiments.get(name)

    def create_experiment(self, name: str, artifact_location: str) -> str:
        exp_id = f"exp-{len(self.experiments)}"
        self.experiments[name] = SimpleNamespace(
            experiment_id=exp_id,
            artifact_location=artifact_location,
        )
        return exp_id

    def active_run(self) -> DummyRun | None:  # pragma: no cover - trivial
        return self.active

    def start_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: dict[str, str],
    ) -> DummyRun:
        self.active = DummyRun(f"{experiment_id}:{run_name}")
        self.last_tags = tags
        return self.active

    def log_artifacts(self, path: str, run_id: str | None = None) -> None:
        self.logged_artifacts.append((run_id, Path(path).name))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.logged_metrics.append((run_id, dict(metrics), step))

    def end_run(self, status: str | None = None) -> None:  # pragma: no cover - passthrough
        self.ended_status = status


@pytest.fixture
def dummy_mlflow(monkeypatch: pytest.MonkeyPatch) -> DummyMlflow:
    stub = DummyMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", stub)
    return stub


def test_resolve_mlflow_paths_normalizes_relative_uris(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    workspace = tmp_path / "workspace"
    project_root.mkdir()
    workspace.mkdir()

    paths = resolve_mlflow_paths(
        f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}",
        "mlartifacts",
        project_root,
        workspace,
    )

    assert paths.tracking_uri.endswith("/mlruns.db")
    assert paths.artifact_uri == str((workspace / "mlartifacts").resolve())


def test_collect_artifacts_filters_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "figures").mkdir()
    (workspace / "checkpoints").mkdir()

    collected = collect_artifacts(workspace, DEFAULT_ARTIFACT_SUBDIRS)

    names = {path.name for path in collected}
    assert {"figures", "checkpoints"} <= names
    assert "reports" not in names


def test_start_run_and_logging(dummy_mlflow: DummyMlflow, tmp_path: Path) -> None:
    paths = MlflowPaths(
        tracking_uri=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        artifact_uri=str(tmp_path / "mlartifacts"),
    )
    exp_id = ensure_experiment("demo", paths)
    run, started = start_run_if_needed(exp_id, "train", {"stage": "demo"})
    assert started is True
    assert run.info.run_id == f"{exp_id}:train"

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok")
    log_artifacts(run, [artifact])
    log_metrics(run, {"loss": 1.0}, step=2)

    assert dummy_mlflow.logged_artifacts == [(run.info.run_id, "artifact.txt")]
    assert dummy_mlflow.logged_metrics == [(run.info.run_id, {"loss": 1.0}, 2)]


def test_finalize_run_ends_started_run(dummy_mlflow: DummyMlflow, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "figures").mkdir()
    paths = resolve_mlflow_paths(None, None, tmp_path, workspace_root)
    config = MlflowRunConfig(
        experiment_name="exp",
        run_name="run",
        tags={},
        paths=paths,
        workspace_root=workspace_root,
    )
    state = open_run(config)
    finalize_run(
        state,
        workspace_root=workspace_root,
        metrics={"m": 1.0},
        allowlist=("figures",),
    )
    assert dummy_mlflow.logged_artifacts
    assert dummy_mlflow.logged_metrics
    assert getattr(dummy_mlflow, "ended_status", None) == "FINISHED"
