"""Unit tests for the multi_training batch workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neuralls.platform.config.models.experiments import CaseConfig, ExperimentNamesConfig
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.config.loaders import load_experiments_config, load_raw_toml
from neuralls.composition.experiments.multi_training import (
    TrainingRunResult,
    _annotate_mlflow_run,
    _collect_batch_metrics,
    _create_training_session_parent_run,
    _make_label_map,
    _resolve_config_paths,
    _train_single,
    train_batch,
)


@pytest.fixture
def training_run_results(tmp_path: Path) -> list[TrainingRunResult]:
    """Three completed training run results for label-map and metric tests."""
    return [
        TrainingRunResult(
            label="1",
            experiment_id="ffnn_solutions",
            experiment_display_name="ffnn_solutions",
            checkpoint_path=tmp_path / "a" / "model.ckpt",
            mlflow_run_id="aaa111",
            metrics={"eval/rel_error": 0.05, "val_loss": 0.01},
        ),
        TrainingRunResult(
            label="2",
            experiment_id="ffnn_eigenvectors",
            experiment_display_name="ffnn_eigenvectors",
            checkpoint_path=tmp_path / "b" / "model.ckpt",
            mlflow_run_id="bbb222",
            metrics={"eval/rel_error": 0.03},
        ),
        TrainingRunResult(
            label="3",
            experiment_id="ffnn_rhs_largest",
            experiment_display_name="ffnn_rhs_largest",
            checkpoint_path=tmp_path / "c" / "model.ckpt",
            mlflow_run_id=None,
            metrics={},
        ),
    ]


@pytest.fixture
def job_config_file(tmp_path: Path) -> Path:
    """Create a minimal job config TOML file."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    path = jobs_dir / "ffnn.toml"
    path.write_text('[run]\ntype = "train"\n[model]\nclass = "NormScaledLinearFFNN"\n')
    return path


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
                "[[experiments]]",
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
    assert result["1"]["experiment_id"] == "ffnn_solutions"
    assert result["2"]["mlflow_run_id"] == "bbb222"
    assert result["3"]["mlflow_run_id"] is None


def test_collect_batch_metrics(training_run_results: list[TrainingRunResult]) -> None:
    """Batch metrics average matching eval or val keys."""
    result = _collect_batch_metrics(training_run_results)
    assert result["avg_eval/rel_error"] == pytest.approx(0.04)
    assert "avg_val_loss" not in result


def test_resolve_config_paths(
    tmp_path: Path,
    job_config_file: Path,
    dataset_config_file: Path,
    valid_experiments_toml: Path,
    neuralls_settings,
) -> None:
    """Registry-backed entries resolve into concrete config files."""
    cfg = load_experiments_config(valid_experiments_toml, neuralls_settings)
    job_path, data_path = _resolve_config_paths(
        cfg.experiments[0],
        tmp_path,
        cfg,
    )
    assert job_path == job_config_file
    assert data_path == dataset_config_file


def test_train_single_reads_sidecar_and_metrics(tmp_path: Path, neuralls_settings) -> None:
    """Single training run returns the training checkpoint and MLflow metadata."""
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text('[run]\ntype = "train"\n[model]\nclass = "NormScaledLinearFFNN"\n')
    data_cfg = tmp_path / "dataset.toml"
    data_cfg.write_text('id = "dataset"\n[source]\nmatrix_path = "matrix.txt"\n')
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "model.ckpt"
    ckpt.touch()
    (ckpt_dir / "mlflow_run.json").write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "tracking_uri": build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
            }
        )
    )

    with (
        patch("neuralls.composition.experiments.multi_training.train_model", return_value=ckpt),
        patch("neuralls.composition.experiments.multi_training.register_logged_model"),
        patch(
            "neuralls.composition.experiments.multi_training.fetch_mlflow_metrics",
            return_value={"eval/rel_error": 0.1},
        ),
        patch("neuralls.composition.experiments.multi_training.MlflowClient"),
    ):
        result = _train_single(
            settings=neuralls_settings,
            experiment_id="exp-1",
            experiment_display_name="exp-1",
            job_config_path=job_cfg,
            data_config_path=data_cfg,
            label="1",
            output_root=None,
            mlflow_experiment_name="Train",
        )

    assert result.checkpoint_path == ckpt
    assert result.mlflow_run_id == "run-123"
    assert result.metrics["eval/rel_error"] == pytest.approx(0.1)


def test_train_single_forwards_parent_run_id(tmp_path: Path, neuralls_settings) -> None:
    """Single training forwards the optional batch parent run id unchanged."""
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text('[run]\ntype = "train"\n[model]\nclass = "NormScaledLinearFFNN"\n')
    data_cfg = tmp_path / "dataset.toml"
    data_cfg.write_text('id = "dataset"\n[source]\nmatrix_path = "matrix.txt"\n')
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "model.ckpt"
    ckpt.touch()

    with (
        patch(
            "neuralls.composition.experiments.multi_training.train_model", return_value=ckpt
        ) as mock_train,
        patch("neuralls.composition.experiments.multi_training.load_data_config"),
        patch(
            "neuralls.composition.experiments.multi_training.resolve_dataset_identity"
        ) as mock_identity,
        patch("neuralls.composition.experiments.multi_training.read_registered_model_name"),
        patch(
            "neuralls.composition.experiments.multi_training.read_mlflow_sidecar", return_value=None
        ),
    ):
        mock_identity.return_value.name = "dataset-id"
        _train_single(
            settings=neuralls_settings,
            experiment_id="exp-1",
            experiment_display_name="exp-1",
            job_config_path=job_cfg,
            data_config_path=data_cfg,
            label="1",
            output_root=None,
            mlflow_experiment_name="Train",
            parent_run_id="parent-123",
        )

    assert mock_train.call_args.kwargs["parent_run_id"] == "parent-123"


@pytest.fixture
def job_config_with_model_name(tmp_path: Path) -> Path:
    """Job config TOML with [model].class set."""
    path = tmp_path / "job.toml"
    path.write_text('[run]\ntype = "train"\n[model]\nclass = "NormScaledLinearFFNN"\n')
    return path


def test_annotate_mlflow_run_registers_under_experiment_id(
    tmp_path: Path,
    job_config_with_model_name: Path,
) -> None:
    """After training, model is registered under experiment_id with model_class tag."""
    with (
        patch(
            "neuralls.composition.experiments.multi_training.register_logged_model"
        ) as mock_register,
        patch("neuralls.composition.experiments.multi_training.MlflowClient"),
    ):
        _annotate_mlflow_run(
            label="1",
            run_id="run-abc",
            tracking_uri=build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
            checkpoint_path=tmp_path / "model.ckpt",
            job_config_path=job_config_with_model_name,
            experiment_id="spectral-energy",
            experiment_display_name="Spectral Energy",
            dataset_id="solutions",
            dataset_display_name="Solutions",
            dataset_registry_id="solutions",
            job_registry_id="ffnn",
            job_display_name="FFNN",
        )

    mock_register.assert_called_once_with(
        run_id="run-abc",
        registered_model_name="spectral-energy",
        tracking_uri=build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
        tags={
            "experiment_id": "spectral-energy",
            "dataset_id": "solutions",
            "job_id": "ffnn",
            "experiment_display_name": "Spectral Energy",
            "model_class": "NormScaledLinearFFNN",
        },
    )


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
) -> None:
    """Batch output is a local training directory and no batch comparison run is opened."""
    fake_ckpt = tmp_path / "ckpt" / "model.ckpt"
    fake_ckpt.parent.mkdir()
    fake_ckpt.touch()
    cfg = load_experiments_config(valid_experiments_toml, neuralls_settings)

    with (
        patch(
            "neuralls.composition.experiments.multi_training.train_model", return_value=fake_ckpt
        ) as mock_train,
        patch(
            "neuralls.composition.experiments.multi_training._create_training_session_parent_run",
            return_value="parent-run-1",
        ) as mock_create_parent,
        patch(
            "neuralls.composition.experiments.multi_training._finalize_training_session_parent_run"
        ) as mock_finalize_parent,
    ):
        result = train_batch(
            cfg=cfg,
            configs_dir=valid_experiments_toml.parent,
            settings=neuralls_settings,
            case_config_path=valid_experiments_toml,
        )

    assert mock_train.call_count == 1
    assert mock_train.call_args.kwargs["mlflow_experiment_name"] == ExperimentNamesConfig().training
    assert mock_train.call_args.kwargs["parent_run_id"] == "parent-run-1"
    mock_create_parent.assert_called_once()
    mock_finalize_parent.assert_called_once()
    assert result.output_dir == (tmp_path / "training")
    assert result.label_map["1"]["experiment_id"] == "ffnn_test"


def test_train_batch_forwards_custom_training_experiment_name(
    valid_experiments_toml: Path,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """Batch training forwards the case-configured training experiment name."""
    fake_ckpt = tmp_path / "ckpt" / "model.ckpt"
    fake_ckpt.parent.mkdir()
    fake_ckpt.touch()
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
                "[[experiments]]",
                'id = "ffnn_test"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_experiments_config(valid_experiments_toml, neuralls_settings)

    with (
        patch(
            "neuralls.composition.experiments.multi_training.train_model", return_value=fake_ckpt
        ) as mock_train,
        patch(
            "neuralls.composition.experiments.multi_training._create_training_session_parent_run",
            return_value="parent-run-1",
        ),
        patch(
            "neuralls.composition.experiments.multi_training._finalize_training_session_parent_run"
        ),
    ):
        train_batch(
            cfg=cfg,
            configs_dir=valid_experiments_toml.parent,
            settings=neuralls_settings,
            case_config_path=valid_experiments_toml,
        )

    assert mock_train.call_args.kwargs["mlflow_experiment_name"] == "CustomTrain"


def test_create_training_session_parent_run_uses_case_config_identity(tmp_path: Path) -> None:
    """Training session parents are named from case config stem and timestamp."""
    with (
        patch(
            "neuralls.composition.experiments.multi_training._ensure_training_experiment",
            return_value="exp-1",
        ),
        patch("neuralls.composition.experiments.multi_training.MlflowClient") as mock_client_cls,
        patch(
            "neuralls.composition.experiments.multi_training.build_training_session_run_spec",
            return_value=(
                "ffnn | 2026-03-12T12:00:00",
                SimpleNamespace(
                    as_mlflow_tags=lambda: {
                        "phase": "session_training",
                        "case_config": "ffnn",
                        "case_config_path": "/tmp/ffnn.toml",
                        "started_at": "2026-03-12T12:00:00",
                        "training_experiment_name": "Train",
                    }
                ),
            ),
        ),
    ):
        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value.info.run_id = "parent-123"
        run_id = _create_training_session_parent_run(
            tracking_uri="sqlite:///tmp/mlflow.db",
            artifact_uri="/tmp/mlartifacts",
            case_config_path=tmp_path / "ffnn.toml",
            training_experiment_name="Train",
        )

    assert run_id == "parent-123"
    mock_client.create_run.assert_called_once_with(
        experiment_id="exp-1",
        tags={
            "mlflow.runName": "ffnn | 2026-03-12T12:00:00",
            "phase": "session_training",
            "case_config": "ffnn",
            "case_config_path": "/tmp/ffnn.toml",
            "started_at": "2026-03-12T12:00:00",
            "training_experiment_name": "Train",
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
    """Master configs must use only [[experiments]]."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        "\n".join(
            [
                "[[jobs]]",
                'id = "ffnn"',
                'path = "jobs/ffnn.toml"',
                "",
                "[[datasets]]",
                'id = "test-solutions"',
                'path = "datasets/test-solutions.toml"',
                "",
                "[[experiment]]",
                'id = "ffnn_test"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
            ]
        )
    )

    with pytest.raises(ValueError, match=r"\[\[experiment\]\]"):
        load_experiments_config(config, neuralls_settings)


def test_experiments_config_rejects_missing_dataset_id(tmp_path: Path) -> None:
    """Experiments must reference dataset ids declared in [[datasets]]."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        "\n".join(
            [
                "[[jobs]]",
                'id = "ffnn"',
                'path = "jobs/ffnn.toml"',
                "",
                "[[experiments]]",
                'id = "ffnn_test"',
                'job = "ffnn"',
                'dataset = "missing-dataset"',
            ]
        )
    )

    with pytest.raises(
        ValueError, match="Experiment 'ffnn_test' references dataset id 'missing-dataset'"
    ):
        CaseConfig.model_validate(load_raw_toml(config))


def test_experiments_config_rejects_missing_job_id(tmp_path: Path) -> None:
    """Experiments must reference job ids declared in [[jobs]]."""
    config = tmp_path / "experiments.toml"
    config.write_text(
        "\n".join(
            [
                "[[datasets]]",
                'id = "test-solutions"',
                'path = "datasets/test-solutions.toml"',
                "",
                "[[experiments]]",
                'id = "ffnn_test"',
                'job = "missing-job"',
                'dataset = "test-solutions"',
            ]
        )
    )

    with pytest.raises(ValueError, match="Experiment 'ffnn_test' references job id 'missing-job'"):
        CaseConfig.model_validate(load_raw_toml(config))
