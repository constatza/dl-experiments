"""Tests for the refactored training workflow architecture.

Verifies the single-responsibility helpers extracted from train_model().
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config import (
    DataModuleSettings,
    DatasetSettings,
    ModelComponentSettings as ModelSettings,
    SessionSettings,
    TrainingSettings,
)
from dlkit.infrastructure.config.data_entries import IPathBased
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.io import save_sparse_pack

from neuralls.platform.storage.training_artifacts import TrainingArrays
from neuralls.platform.tracking.mlflow_client import parent_run_context
from neuralls.composition.experiments.training import (
    _configure_dataloader_runtime,
    _configure_mlflow,
    _configure_output_paths,
    _create_feature_configs,
    _extract_evaluation_arrays,
    _resolve_mlflow_logging_config,
    _resolve_dataset,
    _validate_dataset_section,
    GRAPH_DATASET_NAME,
    FLEXIBLE_DATASET_NAME,
)


@pytest.fixture
def sample_arrays(tmp_path: Path) -> TrainingArrays:
    """Sample training artifact paths."""
    rhs = np.random.rand(10, 5).astype(np.float64)
    solutions = np.random.rand(10, 5).astype(np.float64)
    rhs_path = tmp_path / "rhs.npy"
    solutions_path = tmp_path / "solutions.npy"
    np.save(rhs_path, rhs)
    np.save(solutions_path, solutions)

    matrix = np.eye(5, dtype=np.float64)
    rows, cols = np.nonzero(matrix)
    indices_single = np.vstack((rows, cols)).astype(np.int64)
    values_single = matrix[rows, cols].astype(np.float64)
    n = rhs.shape[0]
    indices = np.tile(indices_single, (1, n))
    values = np.tile(values_single, n)
    nnz_ptr = np.arange(0, values_single.size * n + 1, values_single.size, dtype=np.int64)
    matrix_pack = tmp_path / "matrix_coo"
    save_sparse_pack(
        path=matrix_pack,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=(5, 5),
    )
    return TrainingArrays(
        rhs=rhs_path,
        solutions=solutions_path,
        matrix_pack=matrix_pack,
        sample_count=10,
    )


@pytest.fixture
def training_settings(tmp_path: Path) -> GeneralSettings:
    """Concrete settings instance for immutable patch-model assertions."""
    trainer_root = tmp_path / "original-root"
    trainer_root.mkdir()
    return GeneralSettings(
        SESSION=SessionSettings(seed=42),
        MODEL=ModelSettings(name="LinearModel"),
        DATAMODULE=DataModuleSettings(
            dataloader=DataloaderSettings(batch_size=8, num_workers=2, pin_memory=True)
        ),
        DATASET=DatasetSettings(name="FlexibleDataset", memmap_cache=True),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                max_epochs=3,
                default_root_dir=trainer_root,
            )
        ),
        MLFLOW=MLflowSettings(experiment_name="BaseExperiment"),
    )


def test_validate_dataset_section_raises_on_none() -> None:
    """_validate_dataset_section raises ValueError if DATASET is None."""
    mock_settings = MagicMock(spec=GeneralSettings)
    mock_settings.DATASET = None

    with pytest.raises(ValueError, match=r"missing \[DATASET\] section"):
        _validate_dataset_section(mock_settings)


def test_validate_dataset_section_passes_if_present() -> None:
    """_validate_dataset_section passes if DATASET is present."""
    mock_settings = MagicMock(spec=GeneralSettings)
    mock_settings.DATASET = MagicMock()

    _validate_dataset_section(mock_settings)  # Should not raise


def test_create_feature_configs_graph_dataset(sample_arrays: TrainingArrays) -> None:
    """_create_feature_configs returns x and sparse matrix entries for GraphDataset."""
    features = _create_feature_configs(sample_arrays, GRAPH_DATASET_NAME)

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, IPathBased)
    assert matrix_feature.model_input is False
    assert matrix_feature.get_path() == sample_arrays.matrix_pack


def test_create_feature_configs_flexible_dataset(sample_arrays: TrainingArrays) -> None:
    """_create_feature_configs returns x and sparse matrix entries for FlexibleDataset."""
    features = _create_feature_configs(sample_arrays, FLEXIBLE_DATASET_NAME)

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, IPathBased)
    assert matrix_feature.model_input is False
    assert matrix_feature.get_path() == sample_arrays.matrix_pack


def test_create_feature_configs_default_to_flexible(sample_arrays: TrainingArrays) -> None:
    """_create_feature_configs defaults to x + sparse matrix behavior."""
    features = _create_feature_configs(sample_arrays, None)

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, IPathBased)
    assert matrix_feature.get_path() == sample_arrays.matrix_pack


def test_parent_run_context_manager() -> None:
    """_parent_run_context correctly sets and restores MLFLOW_PARENT_RUN_ID."""
    var_name = "MLFLOW_PARENT_RUN_ID"
    original_val = os.environ.get(var_name)

    try:
        test_id = "test-parent-id"
        with parent_run_context(test_id):
            assert os.environ.get(var_name) == test_id

        assert os.environ.get(var_name) == original_val

        # Test with None (should do nothing)
        with parent_run_context(None):
            assert os.environ.get(var_name) == original_val

    finally:
        if original_val is not None:
            os.environ[var_name] = original_val
        elif var_name in os.environ:
            del os.environ[var_name]


def test_resolve_dataset_returns_new_settings(
    training_settings: GeneralSettings,
) -> None:
    """Dataset injection returns a new settings object without mutating the original."""
    features = []
    targets = []

    updated = _resolve_dataset(training_settings, features, targets)

    assert updated is not training_settings
    assert updated.DATASET is not None
    assert updated.DATASET.features == tuple(features)
    assert updated.DATASET.targets == tuple(targets)
    assert updated.DATASET.memmap_cache is False
    assert training_settings.DATASET is not None
    assert training_settings.DATASET.features == ()
    assert training_settings.DATASET.targets == ()
    assert training_settings.DATASET.memmap_cache is True


def test_configure_output_paths_returns_new_settings(
    training_settings: GeneralSettings,
    tmp_path: Path,
) -> None:
    """Output-path configuration uses immutable patching."""
    output_dir = tmp_path / "new-root"
    output_dir.mkdir()

    updated = _configure_output_paths(training_settings, output_dir)

    assert updated is not training_settings
    assert updated.TRAINING is not None
    assert updated.TRAINING.trainer is not None
    assert updated.TRAINING.trainer.default_root_dir == output_dir
    assert training_settings.TRAINING is not None
    assert training_settings.TRAINING.trainer is not None
    assert training_settings.TRAINING.trainer.default_root_dir != output_dir


def test_configure_dataloader_runtime_forces_single_process(
    training_settings: GeneralSettings,
) -> None:
    """_configure_dataloader_runtime returns a patched settings object."""
    updated = _configure_dataloader_runtime(training_settings)

    assert updated is not training_settings
    assert updated.DATAMODULE is not None
    assert updated.DATAMODULE.dataloader is not None
    assert updated.DATAMODULE.dataloader.num_workers == 0
    assert updated.DATAMODULE.dataloader.persistent_workers is False
    assert updated.DATAMODULE.dataloader.pin_memory is False
    assert training_settings.DATAMODULE is not None
    assert training_settings.DATAMODULE.dataloader is not None
    assert training_settings.DATAMODULE.dataloader.num_workers == 2
    assert training_settings.DATAMODULE.dataloader.pin_memory is True


def test_configure_mlflow_returns_new_settings(
    training_settings: GeneralSettings,
) -> None:
    """MLflow naming updates use immutable patching."""
    updated = _configure_mlflow(
        training_settings,
        mlflow_experiment_name="Train",
        mlflow_run_name="run-42",
    )

    assert updated is not training_settings
    assert updated.MLFLOW is not None
    assert updated.MLFLOW.experiment_name == "Train"
    assert updated.MLFLOW.run_name == "run-42"
    assert training_settings.MLFLOW is not None
    assert training_settings.MLFLOW.experiment_name == "BaseExperiment"
    assert training_settings.MLFLOW.run_name is None


@patch("neuralls.composition.experiments.training.compute_diagnostics")
@patch("neuralls.composition.experiments.training.write_diagnostics_figure")
@patch("neuralls.composition.experiments.training.log_diagnostics_to_mlflow")
def test_log_training_evaluation_orchestration(
    mock_mlflow_log: MagicMock,
    mock_write: MagicMock,
    mock_compute: MagicMock,
    tmp_path: Path,
) -> None:
    """_log_training_evaluation delegates to diagnostics and figure helpers."""
    from neuralls.composition.experiments.training import _log_training_evaluation

    mock_result = MagicMock()
    mock_result.to_numpy.return_value = {
        "predictions": {"output": np.zeros((10, 1))},
        "targets": {"y": np.zeros((10, 1))},
    }

    mock_compute.return_value = MagicMock()
    mock_write.return_value = Path("dummy.png")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"

    _log_training_evaluation(
        tracking_uri=tracking_uri,
        run_id="run123",
        training_result=mock_result,
        figures_dir=tmp_path / "figures",
    )

    mock_compute.assert_called_once()
    mock_write.assert_called_once()
    mock_mlflow_log.assert_called_once()


def test_extract_evaluation_arrays_with_array_predictions() -> None:
    """Extraction supports numpy prediction arrays with dict targets."""
    selected = _extract_evaluation_arrays(
        {
            "predictions": np.array([[1.0], [2.0], [3.0]]),
            "targets": {"y": np.array([[1.1], [1.9], [3.2]])},
        }
    )
    assert selected is not None
    y_pred, y_true = selected
    np.testing.assert_allclose(y_pred.ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_true.ravel(), [1.1, 1.9, 3.2])


def test_extract_evaluation_arrays_with_non_output_prediction_key() -> None:
    """Extraction falls back to target-matching prediction keys."""
    selected = _extract_evaluation_arrays(
        {
            "predictions": {"solutions": np.array([[1.0], [2.0]])},
            "targets": {"solutions": np.array([[0.8], [2.2]])},
        }
    )
    assert selected is not None
    y_pred, y_true = selected
    np.testing.assert_allclose(y_pred.ravel(), [1.0, 2.0])
    np.testing.assert_allclose(y_true.ravel(), [0.8, 2.2])


def test_extract_evaluation_arrays_returns_none_when_missing_keys() -> None:
    """Extraction returns None when predictions/targets are unavailable."""
    assert _extract_evaluation_arrays({"predictions": np.array([1.0, 2.0])}) is None


def test_resolve_mlflow_logging_config_reads_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MLflow logging now resolves infra from runtime env only."""
    expected_tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_uri = str(tmp_path / "mlartifacts")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", expected_tracking_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", artifact_uri)

    tracking_uri, artifacts_destination = _resolve_mlflow_logging_config()
    assert tracking_uri == expected_tracking_uri
    assert artifacts_destination == artifact_uri


def test_resolve_mlflow_logging_config_handles_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing runtime env returns an empty MLflow infra payload."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    tracking_uri, artifacts_destination = _resolve_mlflow_logging_config()
    assert tracking_uri is None
    assert artifacts_destination == ""
