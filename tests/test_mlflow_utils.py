from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from dlkit.infrastructure.io import url_resolver

import neuralls.platform.tracking.mlflow as tracking_mlflow
from neuralls.platform.config.resolution import MlflowPaths, resolve_mlflow_paths
from neuralls.platform.tracking.mlflow import (
    DEFAULT_ARTIFACT_SUBDIRS,
    MlflowRunConfig,
    build_runtime_environment,
    build_workflow_environment,
    collect_artifacts,
    ensure_experiment,
    finalize_run,
    log_artifacts,
    log_metrics,
    open_run,
    quote_filter_value,
    resolve_runtime_tracking_config,
    runtime_paths_from_env,
    sanitize_metric_key_segment,
    start_run_if_needed,
)
from neuralls.platform.tracking.mlflow_client import log_artifacts_to_mlflow


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
    monkeypatch.setattr(tracking_mlflow, "mlflow", stub)
    return stub


def test_resolve_mlflow_paths_normalizes_relative_uris(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    workspace = tmp_path / "workspace"
    project_root.mkdir()
    workspace.mkdir()

    paths = resolve_mlflow_paths(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}",
        artifact_uri="mlartifacts",
        project_root=project_root,
        workspace_root=workspace,
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


def test_sanitize_metric_key_segment_replaces_invalid_characters() -> None:
    assert sanitize_metric_key_segment("Residual | Error / Foo") == "Residual_Error_Foo"
    assert sanitize_metric_key_segment("///", default="fallback") == "fallback"


def test_quote_filter_value_escapes_backslashes_and_quotes() -> None:
    assert quote_filter_value(r"foo\bar'baz") == r"foo\\bar\'baz"


def test_build_workflow_environment_uses_default_output_root(tmp_path: Path) -> None:
    runtime = build_workflow_environment(
        tracking_uri=None,
        artifact_location=None,
        default_output_root=tmp_path,
    )

    assert runtime.tracking_uri.endswith("/mlruns/mlflow.db")
    assert runtime.artifact_uri == str((tmp_path / "mlartifacts").resolve())


def test_runtime_paths_from_env_reads_tracking_values() -> None:
    paths = runtime_paths_from_env(
        {
            "MLFLOW_TRACKING_URI": "sqlite:////tmp/mlruns.db",
            "MLFLOW_ARTIFACT_URI": "/tmp/mlartifacts",
        }
    )

    assert paths == MlflowPaths(
        tracking_uri="sqlite:////tmp/mlruns.db",
        artifact_uri="/tmp/mlartifacts",
    )


def test_resolve_runtime_tracking_config_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/runtime.db")
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", "/tmp/runtime-artifacts")

    assert resolve_runtime_tracking_config() == (
        "sqlite:////tmp/runtime.db",
        "/tmp/runtime-artifacts",
    )


def test_build_runtime_environment_prefers_existing_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/existing.db")

    runtime = build_runtime_environment(tmp_path / "unused", default_output_root=tmp_path)

    assert runtime.tracking_uri == "sqlite:////tmp/existing.db"
    assert runtime.artifact_uri is None


def test_build_runtime_environment_preserves_existing_artifact_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/existing.db")
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", "/tmp/existing-artifacts")

    runtime = build_runtime_environment(tmp_path / "unused", default_output_root=tmp_path)

    assert runtime.env == {
        "MLFLOW_TRACKING_URI": "sqlite:////tmp/existing.db",
        "MLFLOW_ARTIFACT_URI": "/tmp/existing-artifacts",
    }
    assert runtime.tracking_uri == "sqlite:////tmp/existing.db"
    assert runtime.artifact_uri == "/tmp/existing-artifacts"


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


def test_ensure_experiment_uses_file_uri_for_local_artifact_root(
    dummy_mlflow: DummyMlflow,
    tmp_path: Path,
) -> None:
    paths = MlflowPaths(
        tracking_uri=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        artifact_uri=str(tmp_path / "mlartifacts"),
    )

    exp_id = ensure_experiment("demo-file-uri", paths)

    assert exp_id == "exp-0"
    assert dummy_mlflow.experiments["demo-file-uri"].artifact_location == (
        url_resolver.build_uri((tmp_path / "mlartifacts").resolve(), scheme="file")
    )


def test_finalize_run_ends_started_run(
    dummy_mlflow: DummyMlflow,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "figures").mkdir()
    paths = resolve_mlflow_paths(
        tracking_uri=None,
        artifact_uri=None,
        project_root=tmp_path,
        workspace_root=workspace_root,
        default_tracking_uri=neuralls_settings.mlflow_tracking_uri,
        default_artifact_location=neuralls_settings.mlflow_artifact_location,
    )
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


def test_resolve_mlflow_paths_requires_tracking_or_default(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="tracking_uri is required"):
        resolve_mlflow_paths(
            tracking_uri=None,
            artifact_uri=None,
            workspace_root=workspace,
        )


def test_resolve_mlflow_paths_requires_workspace_for_local_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="workspace_root is required"):
        resolve_mlflow_paths(
            tracking_uri=f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}",
            artifact_uri="mlartifacts",
        )


def test_log_artifacts_to_mlflow_excludes_checkpoints(tmp_path: Path) -> None:
    """Workspace upload should not re-log checkpoints already handled by DLKit."""
    workspace_root = tmp_path / "workspace"
    for name in ("checkpoints", "figures", "metrics", "predictions", "config"):
        directory = workspace_root / name
        directory.mkdir(parents=True)
        (directory / f"{name}.txt").write_text(name)

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        log_artifacts_to_mlflow(
            tracking_uri=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
            run_id="run-1",
            workspace_root=workspace_root,
        )

    logged_paths = [Path(call.args[1]).name for call in client.log_artifacts.call_args_list]
    assert "checkpoints" not in logged_paths
    assert set(logged_paths) == {"figures", "metrics", "predictions", "config"}
