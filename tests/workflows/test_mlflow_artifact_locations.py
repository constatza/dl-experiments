"""Lightweight MLflow artifact-location tests using local SQLite tracking."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import tomli_w
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from neuralls.io.dataset_storage import save_dataset
from neuralls.workflows.comparison import run_comparison
from neuralls.workflows.diagnostics import compute_diagnostics
from neuralls.workflows.mlflow_client import log_diagnostics_to_mlflow
from neuralls.workflows.specs import ComparisonParams
from neuralls.workflows.training import _log_training_evaluation


def _sqlite_tracking_uri(db_path: Path) -> str:
    """Build sqlite tracking URI from DB path."""
    return f"sqlite:///{db_path.as_posix()}"


def _ensure_experiment(
    *,
    client: MlflowClient,
    name: str,
    artifact_root: Path,
) -> str:
    """Create experiment if needed and return id."""
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    try:
        return client.create_experiment(
            name=name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    except MlflowException:
        existing = client.get_experiment_by_name(name)
        if existing is None:
            raise
        return existing.experiment_id


def test_training_logs_diagnostics_artifact_to_mlflow_with_sqlite(tmp_path: Path) -> None:
    """Training diagnostics are logged under figures/ in the active MLflow run."""
    output_root = tmp_path / "output"
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="training-diagnostics-sqlite",
        artifact_root=artifact_root,
    )

    mlflow.set_tracking_uri(tracking_uri)
    predictions = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    targets = np.array([[1.1], [1.9], [3.2]], dtype=np.float64)
    fake_training_result = MagicMock()
    fake_training_result.to_numpy.return_value = {
        "predictions": {"output": predictions.flatten()},
        "targets": {"solutions": targets.flatten()},
    }

    with mlflow.start_run(experiment_id=experiment_id, run_name="diag-train") as run:
        run_id = run.info.run_id

    _log_training_evaluation(
        tracking_uri=tracking_uri,
        run_id=run_id,
        training_result=fake_training_result,
        figures_dir=tmp_path / "tmp-figures",
    )

    run_data = client.get_run(run_id).data
    assert "eval/rel_error" in run_data.metrics
    assert "eval/mae" in run_data.metrics
    assert "eval/mse" in run_data.metrics

    figures = client.list_artifacts(run_id, path="figures")
    assert any(item.path.endswith("diagnostics_training.png") for item in figures)


def test_log_diagnostics_to_mlflow_reopens_sqlite_run_without_active_context(
    tmp_path: Path,
) -> None:
    """Diagnostics helper logs directly to an existing SQLite-backed run."""
    output_root = tmp_path / "output"
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="training-diagnostics-direct-sqlite",
        artifact_root=artifact_root,
    )

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id, run_name="diag-direct") as run:
        run_id = run.info.run_id

    diagnostics = compute_diagnostics(
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([[1.1], [1.9]], dtype=np.float64),
    )
    figure_path = tmp_path / "diagnostics_training.png"
    figure_path.write_text("placeholder", encoding="utf-8")

    log_diagnostics_to_mlflow(tracking_uri, run_id, diagnostics, figure_path)

    run_data = client.get_run(run_id).data
    assert "eval/rel_error" in run_data.metrics
    assert "eval/mae" in run_data.metrics
    assert "eval/mse" in run_data.metrics

    figures = client.list_artifacts(run_id, path="figures")
    assert any(item.path.endswith("diagnostics_training.png") for item in figures)


def _write_experiments_config(path: Path, output_root: Path) -> str:
    """Write sqlite-only experiments topology and return tracking URI."""
    tracking_uri = _sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    payload = {
        "mlflow": {
            "tracking_uri": tracking_uri,
        },
            "names": {
                "training": "neuralls-training",
                "comparison": "Comparisons",
            },
    }
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)
    return tracking_uri


def _write_comparison_config(path: Path, dataset_dir: Path, tracking_uri: str) -> None:
    """Write minimal comparison config for run_comparison orchestration."""
    path.write_text(
        "\n".join(
            [
                "[general]",
                "",
                "[general.params]",
                "rtol = 1e-6",
                "atol = 1e-14",
                "max_iterations = 20",
                'stopping_criterion = "residual_norm"',
                "m_max = 5",
                "",
                "[general.data]",
                f'matrix_path = "{dataset_dir}"',
                f'rhs_path = "{dataset_dir}"',
                'normalize_system = "matrix"',
                "",
                "[[preconditioners]]",
                'name = "none"',
                'type = "identity"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_comparison_logs_artifacts_to_mlflow_with_sqlite(tmp_path: Path) -> None:
    """Comparison workflow logs figures and diagnostics files under run artifacts."""
    output_root = tmp_path / "output"
    experiments_config = tmp_path / "experiments.toml"
    tracking_uri = _write_experiments_config(experiments_config, output_root)
    artifact_root = output_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Ensure comparison experiment uses a deterministic local artifact root.
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(
        client=client,
        name="Comparisons",
        artifact_root=artifact_root,
    )

    dataset_dir = tmp_path / "dataset"
    save_dataset(
        dataset_dir=dataset_dir,
        rhs=np.ones((1, 2), dtype=np.float64),
        solutions=np.ones((1, 2), dtype=np.float64),
        matrix=np.eye(2, dtype=np.float64),
        normalization_type="matrix",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
        scale_metadata={},
    )

    comparison_config = tmp_path / "comparison.toml"
    _write_comparison_config(comparison_config, dataset_dir, tracking_uri)

    def _fake_compare_preconditioners(*, output_root: Path, **_: object) -> SimpleNamespace:
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        (figures_dir / "comparison_plot.png").write_text("placeholder", encoding="utf-8")
        return SimpleNamespace(
            condition_numbers={"none": 1.0},
            results={"none": SimpleNamespace(iterations=2, residual=1.0e-8)},
            recommendations={"best_overall": {"iterations": 2, "residual": 1.0e-8}},
        )

    with patch(
        "neuralls.workflows.comparison.resolve_preconditioner_models",
        side_effect=lambda **kwargs: kwargs["specs"],
    ), patch(
        "neuralls.workflows.comparison.compare_preconditioners",
        side_effect=_fake_compare_preconditioners,
    ):
        outcomes = run_comparison(
            comparison_config=comparison_config,
            params=ComparisonParams(),
            experiments_config_path=experiments_config,
        )

    assert outcomes and outcomes[0].success is True

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    assert runs
    run_id = runs[0].info.run_id

    root_artifacts = client.list_artifacts(run_id)
    root_paths = {item.path for item in root_artifacts}
    assert "comparison.toml" in root_paths
    assert "figures" in root_paths
    # Solver config is logged under config/ subdirectory, not at root
    assert "config" in root_paths
    config_artifacts = client.list_artifacts(run_id, path="config")
    config_names = {Path(item.path).name for item in config_artifacts}
    assert comparison_config.name in config_names

    figure_artifacts = client.list_artifacts(run_id, path="figures")
    assert any(item.path.endswith("comparison_plot.png") for item in figure_artifacts)

    metrics = client.get_run(run_id).data.metrics
    assert "best_iterations" in metrics
    assert "best_residual" in metrics
