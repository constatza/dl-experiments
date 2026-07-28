"""Unit tests for the multi_training batch workflow."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest
from dlkit.common import ChildSuccess

from neuralls.composition.assignments.multi_training import (
    TrainingRunResult,
    _annotate_mlflow_run,
    _finalize_batch_child,
    _make_label_map,
    _resolve_config_paths,
    train_batch,
)
from neuralls.composition.assignments.training import PreparedTraining
from neuralls.platform.config.loaders import load_case_config, load_raw_toml
from neuralls.platform.config.models.experiments import CaseConfig, ExperimentNamesConfig
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.tracking.mlflow import create_session_parent_run


def _write_model_profile(path: Path, model_name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
[model]
name = "{model_name}"

[data]
name = "FlexibleDataset"
batch_size = 2
num_workers = 0
pin_memory = false
shuffle = true

[data.module]
name = "ArrayDataModule"
"""
    )
    return path


def _write_job_config(
    path: Path, model_profile_path: Path, *, session_name: str = "dlkit-session"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_ref = Path(os.path.relpath(model_profile_path, start=path.parent)).as_posix()
    path.write_text(
        f"""
[run]
type = "train"
seed = 42
precision = "64"
model = "{model_ref}"
data = "{model_ref}"

[experiment]
name = "{session_name}"

[training.trainer]
max_epochs = 1
"""
    )
    return path


@pytest.fixture
def training_run_results(tmp_path: Path) -> list[TrainingRunResult]:
    """Three completed training run results for label-map and metric tests."""
    return [
        TrainingRunResult(
            label="1",
            assignment_id="ffnn_solutions",
            assignment_display_name="ffnn_solutions",
            mlflow_run_id="aaa111",
            metrics={"eval/mae": 0.05, "val_loss": 0.01},
        ),
        TrainingRunResult(
            label="2",
            assignment_id="ffnn_eigenvectors",
            assignment_display_name="ffnn_eigenvectors",
            mlflow_run_id="bbb222",
            metrics={"eval/mae": 0.03},
        ),
        TrainingRunResult(
            label="3",
            assignment_id="ffnn_rhs_largest",
            assignment_display_name="ffnn_rhs_largest",
            mlflow_run_id="ccc333",
            metrics={},
        ),
    ]


@pytest.fixture
def make_prepared_training(tmp_path: Path):
    """Factory for a minimal ``PreparedTraining`` stand-in, keyed by assignment id.

    ``assignment``/``run_config`` are duck-typed ``SimpleNamespace`` objects
    carrying only the attributes ``_finalize_batch_child``/``to_run_spec``
    actually read — real ``RunnableAssignment``/``MlflowRunConfig`` instances
    aren't needed for these unit tests.
    """

    def _make(assignment_id: str = "exp-1") -> PreparedTraining:
        model_profile = _write_model_profile(
            tmp_path / f"{assignment_id}-model.toml", "NormScaledLinearFFNN"
        )
        job_cfg = _write_job_config(tmp_path / f"{assignment_id}-job.toml", model_profile)
        data_cfg = tmp_path / f"{assignment_id}-dataset.toml"
        data_cfg.write_text('id = "dataset"\n[source]\nmatrix_path = "matrix.txt"\n')
        spec = SimpleNamespace(
            assignment_id=assignment_id,
            assignment_display_name=assignment_id,
            dataset_id="ds-1",
            dataset_display_name=None,
            job_id="job-1",
            job_display_name=None,
        )
        return PreparedTraining(
            workflow_settings=None,
            run_config=SimpleNamespace(
                experiment_name="Train",
                run_name=assignment_id,
                tags={"assignment_id": assignment_id},
            ),
            runtime_mlflow_env=None,
            tmp_path=tmp_path,
            workspace=None,
            assignment=SimpleNamespace(spec=spec),
            resolved_assignment_display_name=assignment_id,
            dataset_id="ds-1",
            resolved_dataset_display_name="ds-1",
            contract=None,
            config_path=job_cfg,
            resolved_data_config_path=data_cfg,
        )

    return _make


@pytest.fixture
def prepared_training(make_prepared_training) -> PreparedTraining:
    """A single ``PreparedTraining`` stand-in for finalize-time unit tests."""
    return make_prepared_training()


@pytest.fixture
def job_config_file(tmp_path: Path) -> Path:
    """Create a minimal job config TOML file."""
    model_profile = _write_model_profile(
        tmp_path / "models" / "ffnn-model.toml", "NormScaledLinearFFNN"
    )
    return _write_job_config(tmp_path / "jobs" / "ffnn.toml", model_profile, session_name="ffnn")


@pytest.fixture
def dataset_config_file(tmp_path: Path) -> Path:
    """Create a minimal dataset config TOML file."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    path = datasets_dir / "test-solutions.toml"
    path.write_text('id = "test-solutions"\n')
    return path


@pytest.fixture
def valid_experiments_toml(
    tmp_path: Path,
    job_config_file: Path,
    dataset_config_file: Path,
) -> Path:
    """Valid experiments TOML with one registry-backed entry."""
    path = tmp_path / "experiments.toml"
    path.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")}"',
                "",
                "[[jobs]]",
                'id = "ffnn"',
                'path = "jobs/ffnn.toml"',
                "",
                "[[datasets]]",
                'id = "test-solutions"',
                'path = "datasets/test-solutions.toml"',
                "",
                "[[assignments]]",
                'id = "ffnn_test"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
            ]
        )
    )
    return path


def test_make_label_map(training_run_results: list[TrainingRunResult]) -> None:
    """Short labels map back to the full training identity."""
    result = _make_label_map(training_run_results)
    assert result["1"]["assignment_id"] == "ffnn_solutions"
    assert result["2"]["mlflow_run_id"] == "bbb222"
    assert result["3"]["mlflow_run_id"] == "ccc333"


def test_resolve_config_paths(
    tmp_path: Path,
    job_config_file: Path,
    dataset_config_file: Path,
    valid_experiments_toml: Path,
    neuralls_settings,
) -> None:
    """Registry-backed entries resolve into concrete config files."""
    cfg = load_case_config(valid_experiments_toml, neuralls_settings)
    job_path, data_path = _resolve_config_paths(
        cfg.assignments[0],
        tmp_path,
        cfg,
    )
    assert job_path == job_config_file
    assert data_path == dataset_config_file


def test_finalize_batch_child_returns_run_id_and_metrics(
    prepared_training: PreparedTraining, neuralls_settings
) -> None:
    """Finalizing one successful multirun child returns its MLflow run id and metrics."""
    tracking_uri = build_sqlite_tracking_uri(prepared_training.tmp_path / "mlflow.db")

    with (
        patch(
            "neuralls.composition.assignments.multi_training.finalize_prepared_training",
            return_value=("run-123", tracking_uri),
        ),
        patch(
            "neuralls.composition.assignments.multi_training.fetch_mlflow_metrics",
            return_value={"eval/mae": 0.1},
        ),
        patch("neuralls.composition.assignments.multi_training.MlflowClient"),
    ):
        result = _finalize_batch_child(
            label="1",
            prepared=prepared_training,
            execution_result=object(),
            settings=neuralls_settings,
        )

    assert result.mlflow_run_id == "run-123"
    assert result.metrics["eval/mae"] == pytest.approx(0.1)
    assert result.assignment_id == "exp-1"


@pytest.fixture
def job_config_with_model_name(tmp_path: Path) -> Path:
    """Thin job config TOML with a referenced model profile."""
    model_profile = _write_model_profile(tmp_path / "model.toml", "NormScaledLinearFFNN")
    return _write_job_config(tmp_path / "job.toml", model_profile)


def test_annotate_mlflow_run_tags_run_with_assignment_id(
    tmp_path: Path,
    job_config_with_model_name: Path,
) -> None:
    """After training, the run is tagged with assignment_id and model_class — not registered."""
    with patch("neuralls.composition.assignments.multi_training.MlflowClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        _annotate_mlflow_run(
            label="1",
            run_id="run-abc",
            tracking_uri=build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
            job_config_path=job_config_with_model_name,
            assignment_id="spectral-energy",
            assignment_display_name="Spectral Energy",
            resolved_dataset_id="solutions",
            dataset_display_name="Solutions",
            dataset_registry_id="solutions",
            job_registry_id="ffnn",
            job_display_name="FFNN",
        )

    expected_tags = {
        "assignment_id": "spectral-energy",
        "dataset_id": "solutions",
        "job_id": "ffnn",
        "assignment_display_name": "Spectral Energy",
        "model_class": "NormScaledLinearFFNN",
    }
    mock_client.set_tag.assert_has_calls(
        [call("run-abc", key, value) for key, value in expected_tags.items()],
        any_order=True,
    )
    assert mock_client.set_tag.call_count == len(expected_tags)


def test_train_batch_raises_for_empty_config(tmp_path: Path, neuralls_settings) -> None:
    """Training batch rejects configs with no experiments."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns.db")}"',
            ]
        )
    )
    cfg = CaseConfig.model_validate(load_raw_toml(config))
    with pytest.raises(ValueError, match="No .* entries found"):
        train_batch(cfg=cfg, configs_dir=tmp_path, settings=neuralls_settings)


def test_train_batch_returns_local_output_dir(
    valid_experiments_toml: Path,
    tmp_path: Path,
    neuralls_settings,
    make_prepared_training,
) -> None:
    """Batch output is a local training directory, sourced from one dlkit multirun sweep."""
    tracking_uri = build_sqlite_tracking_uri(tmp_path / "mlflow.db")
    cfg = load_case_config(valid_experiments_toml, neuralls_settings)
    prepared = make_prepared_training("ffnn_test")
    sweep_result = SimpleNamespace(
        parent_run_id="parent-run-1",
        tracking_uri=tracking_uri,
        children=(
            ChildSuccess(
                child_id="ffnn_test", label="ffnn_test", run_id="run-123", result=object()
            ),
        ),
    )

    with (
        patch(
            "neuralls.composition.assignments.multi_training.prepare_training_settings",
            return_value=prepared,
        ) as mock_prepare,
        patch(
            "neuralls.composition.assignments.multi_training.run_multirun_spec",
            return_value=sweep_result,
        ) as mock_sweep,
        patch(
            "neuralls.composition.assignments.multi_training.finalize_prepared_training",
            return_value=("run-123", tracking_uri),
        ),
        patch(
            "neuralls.composition.assignments.multi_training.finalize_session_parent_run"
        ) as mock_finalize_parent,
        patch(
            "neuralls.composition.assignments.multi_training.fetch_mlflow_metrics", return_value={}
        ),
        patch("neuralls.composition.assignments.multi_training.MlflowClient"),
    ):
        result = train_batch(
            cfg=cfg,
            configs_dir=valid_experiments_toml.parent,
            settings=neuralls_settings,
            case_config_path=valid_experiments_toml,
        )

    assert mock_prepare.call_count == 1
    assert (
        mock_prepare.call_args.kwargs["mlflow_experiment_name"] == ExperimentNamesConfig().training
    )
    assert mock_prepare.call_args.kwargs["batched"] is True
    mock_sweep.assert_called_once()
    mock_finalize_parent.assert_called_once_with(
        tracking_uri=tracking_uri, run_id="parent-run-1", status="FINISHED"
    )
    assert result.output_dir == (tmp_path / "training")
    assert result.label_map["1"]["assignment_id"] == "ffnn_test"


def test_train_batch_forwards_custom_training_experiment_name(
    valid_experiments_toml: Path,
    tmp_path: Path,
    neuralls_settings,
    make_prepared_training,
) -> None:
    """Batch training forwards the case-configured training experiment name."""
    valid_experiments_toml.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")}"',
                "",
                "[names]",
                'training = "CustomTrain"',
                'comparison = "CustomComparison"',
                "",
                "[[jobs]]",
                'id = "ffnn"',
                'path = "jobs/ffnn.toml"',
                "",
                "[[datasets]]",
                'id = "test-solutions"',
                'path = "datasets/test-solutions.toml"',
                "",
                "[[assignments]]",
                'id = "ffnn_test"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_case_config(valid_experiments_toml, neuralls_settings)
    tracking_uri = build_sqlite_tracking_uri(tmp_path / "mlflow.db")
    prepared = make_prepared_training("ffnn_test")
    sweep_result = SimpleNamespace(
        parent_run_id="parent-run-1",
        tracking_uri=tracking_uri,
        children=(
            ChildSuccess(
                child_id="ffnn_test", label="ffnn_test", run_id="run-123", result=object()
            ),
        ),
    )

    with (
        patch(
            "neuralls.composition.assignments.multi_training.prepare_training_settings",
            return_value=prepared,
        ) as mock_prepare,
        patch(
            "neuralls.composition.assignments.multi_training.run_multirun_spec",
            return_value=sweep_result,
        ),
        patch(
            "neuralls.composition.assignments.multi_training.finalize_prepared_training",
            return_value=("run-123", tracking_uri),
        ),
        patch("neuralls.composition.assignments.multi_training.finalize_session_parent_run"),
        patch(
            "neuralls.composition.assignments.multi_training.fetch_mlflow_metrics", return_value={}
        ),
        patch("neuralls.composition.assignments.multi_training.MlflowClient"),
    ):
        train_batch(
            cfg=cfg,
            configs_dir=valid_experiments_toml.parent,
            settings=neuralls_settings,
            case_config_path=valid_experiments_toml,
        )

    assert mock_prepare.call_args.kwargs["mlflow_experiment_name"] == "CustomTrain"


def test_create_session_parent_run_uses_case_config_identity(tmp_path: Path) -> None:
    """Session parents are named from case config stem and timestamp."""
    case_config_path = tmp_path / "ffnn.toml"
    tracking_db = tmp_path / "mlflow.db"
    artifact_dir = tmp_path / "mlartifacts"
    with (
        patch(
            "neuralls.platform.tracking.mlflow.ensure_experiment",
            return_value="exp-1",
        ),
        patch("neuralls.platform.tracking.mlflow.MlflowClient") as mock_client_cls,
    ):
        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value.info.run_id = "parent-123"
        run_id = create_session_parent_run(
            tracking_uri=f"sqlite:///{tracking_db.as_posix()}",
            artifact_uri=str(artifact_dir),
            run_name="ffnn | 2026-03-12T12:00:00",
            tags={
                "phase": "session_training",
                "case_config": "ffnn",
                "case_config_path": str(case_config_path),
                "started_at": "2026-03-12T12:00:00",
                "experiment_name": "Train",
            },
            experiment_name="Train",
        )

    assert run_id == "parent-123"
    mock_client.create_run.assert_called_once_with(
        experiment_id="exp-1",
        tags={
            "mlflow.runName": "ffnn | 2026-03-12T12:00:00",
            "phase": "session_training",
            "case_config": "ffnn",
            "case_config_path": str(case_config_path),
            "started_at": "2026-03-12T12:00:00",
            "experiment_name": "Train",
        },
    )


def test_experiments_config_rejects_legacy_comparison_profiles(tmp_path: Path) -> None:
    """Legacy comparison profile registry is rejected."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")}"',
                "",
                "[[comparison_profiles]]",
                'id = "linear"',
                'path = "comparison/linear.toml"',
            ]
        )
    )

    with pytest.raises(ValueError, match="comparison_profiles"):
        CaseConfig.model_validate(load_raw_toml(config))


def test_experiments_config_rejects_legacy_singular_experiment_table(
    tmp_path: Path, neuralls_settings
) -> None:
    """Master configs must use only [[assignments]]."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        '[[jobs]]\nid = "ffnn"\npath = "jobs/ffnn.toml"\n\n[[datasets]]\nid = "test-solutions"\npath = "datasets/test-solutions.toml"\n\n[[experiment]]\nid = "ffnn_test"\njob = "ffnn"\ndataset = "test-solutions"'
    )

    with pytest.raises(ValueError, match=r"\[\[experiment\]\]"):
        load_case_config(config, neuralls_settings)


def test_experiments_config_rejects_missing_dataset_id(tmp_path: Path) -> None:
    """Experiments must reference dataset ids declared in [[datasets]]."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        '[[jobs]]\nid = "ffnn"\npath = "jobs/ffnn.toml"\n\n[[assignments]]\nid = "ffnn_test"\njob = "ffnn"\ndataset = "missing-dataset"'
    )

    with pytest.raises(
        ValueError, match="Assignment 'ffnn_test' references dataset id 'missing-dataset'"
    ):
        CaseConfig.model_validate(load_raw_toml(config))


def test_experiments_config_rejects_missing_job_id(tmp_path: Path) -> None:
    """Experiments must reference job ids declared in [[jobs]]."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        '[[datasets]]\nid = "test-solutions"\npath = "datasets/test-solutions.toml"\n\n[[assignments]]\nid = "ffnn_test"\njob = "missing-job"\ndataset = "test-solutions"'
    )

    with pytest.raises(ValueError, match="Assignment 'ffnn_test' references job id 'missing-job'"):
        CaseConfig.model_validate(load_raw_toml(config))
