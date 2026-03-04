"""Integration test for MLflow registered-alias model resolution.

Uses local SQLite tracking + local artifacts (no MLflow server process).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import mlflow
import pytest

from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
)
from neuralls.workflows.model_catalog import assign_dataset_alias_to_registered_model
from neuralls.workflows.model_resolution import resolve_model_ref


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _artifact_uri_to_path(artifact_uri: str) -> Path:
    """Convert MLflow artifact URI to local filesystem path."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "file":
        return Path(parsed.path).resolve()
    return Path(artifact_uri).resolve()


def _assert_path_within(path: Path, root: Path) -> None:
    """Assert that path is under root, with resolved absolute paths."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise AssertionError(f"Path {resolved_path} is not under {resolved_root}")


class _IdentityPyFuncModel(mlflow.pyfunc.PythonModel):
    """Minimal model used only for registry wiring in tests."""

    def predict(self, context, model_input):  # type: ignore[override]
        _ = context
        return model_input


@pytest.mark.integration
def test_registered_alias_resolution_with_local_sqlite_tracking(tmp_path: Path) -> None:
    """Assign dataset alias to a registered run and resolve it via model_ref."""
    tracking_uri = _tracking_uri(tmp_path)
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"alias-resolution-integration-{tmp_path.name}"
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=str(artifacts_dir),
        )
    mlflow.set_experiment(experiment_name)

    checkpoint_file = tmp_path / "dummy.ckpt"
    checkpoint_file.write_text("dummy-checkpoint")

    with mlflow.start_run(run_name="tiny-model-run") as run:
        run_id = run.info.run_id
        run_artifact_path = _artifact_uri_to_path(run.info.artifact_uri)
        _assert_path_within(run_artifact_path, artifacts_dir)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_IdentityPyFuncModel(),
        )
        mlflow.log_artifact(str(checkpoint_file), artifact_path="model")

    model_name = "TinyAliasIntegrationModel"
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
    )
    registered_version = int(str(registered.version))

    assigned_version = assign_dataset_alias_to_registered_model(
        tracking_uri=tracking_uri,
        registered_model_name=model_name,
        run_id=run_id,
        dataset_alias="solutions",
    )
    assert assigned_version == registered_version

    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref={
            "source": "registered",
            "name": model_name,
            "alias": "@solutions",
        },
    )

    download_root = tmp_path / "downloads"
    resolution = resolve_model_ref(
        spec=spec,
        tracking_uri=tracking_uri,
        destination=download_root,
    )
    assert resolution.run_id == run_id
    assert resolution.checkpoint_path.exists()
    assert resolution.checkpoint_path.suffix == ".ckpt"
    _assert_path_within(resolution.checkpoint_path, download_root)
