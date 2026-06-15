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
from dlkit.infrastructure.config.data_entries import DataRole, IPathBased, ValueEntry
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings

from neuralls.platform.storage.training_artifacts import TrainingArrays
from neuralls.platform.tracking.mlflow import resolve_runtime_tracking_config
from neuralls.platform.tracking.mlflow_client import parent_run_context
from neuralls.composition.experiments.runtime_dataset_contract import (
    RuntimeDatasetContract,
    default_training_dataset_contract,
)
from neuralls.composition.experiments.training import (
    _configure_dataloader_runtime,
    _configure_mlflow,
    _configure_output_paths,
    _create_feature_configs,
    _normalize_training_numpy_payload,
    _create_target_configs,
    _extract_evaluation_arrays,
    _resolve_dataset,
    _validate_runtime_dataset_contract,
    _validate_dataset_section,
)


@pytest.fixture
def sample_rhs() -> np.ndarray:
    """Sample RHS numpy array (10 samples, 5 features)."""
    return np.random.default_rng(seed=42).random((10, 5)).astype(np.float64)


@pytest.fixture
def sample_solutions() -> np.ndarray:
    """Sample solutions numpy array (10 samples, 5 features)."""
    return np.random.default_rng(seed=43).random((10, 5)).astype(np.float64)


@pytest.fixture
def sample_arrays(tmp_path: Path) -> TrainingArrays:
    """Sample training artifact paths."""
    rng = np.random.default_rng(seed=42)
    rhs = rng.random((10, 5)).astype(np.float64)
    solutions = rng.random((10, 5)).astype(np.float64)
    rhs_path = tmp_path / "rhs.npy"
    solutions_path = tmp_path / "solutions.npy"
    np.save(rhs_path, rhs)
    np.save(solutions_path, solutions)

    matrix_zarr = tmp_path / "matrix.zarr"
    matrix_zarr.mkdir()
    (matrix_zarr / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')
    return TrainingArrays(
        rhs=rhs_path,
        solutions=solutions_path,
        matrix_zarr=matrix_zarr,
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
        DATASET=DatasetSettings(name="FlexibleDataset"),
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


def test_create_feature_configs_graph_dataset(
    sample_arrays: TrainingArrays, sample_rhs: np.ndarray
) -> None:
    """_create_feature_configs returns in-memory rhs + path-dropped matrix for GraphDataset."""
    contract = default_training_dataset_contract()
    features = _create_feature_configs(sample_arrays, sample_rhs, contract, frozenset())

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    rhs_feature = next(f for f in features if f.name == "x")
    assert isinstance(rhs_feature, ValueEntry)
    np.testing.assert_array_equal(rhs_feature.value, sample_rhs)
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, IPathBased)
    assert matrix_feature.model_input is False
    assert matrix_feature.get_path() == sample_arrays.matrix_zarr


def test_create_feature_configs_flexible_dataset(
    sample_arrays: TrainingArrays, sample_rhs: np.ndarray
) -> None:
    """_create_feature_configs returns in-memory rhs + path-dropped matrix for FlexibleDataset."""
    contract = default_training_dataset_contract()
    features = _create_feature_configs(sample_arrays, sample_rhs, contract, frozenset())

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    rhs_feature = next(f for f in features if f.name == "x")
    assert isinstance(rhs_feature, ValueEntry)
    np.testing.assert_array_equal(rhs_feature.value, sample_rhs)
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, IPathBased)
    assert matrix_feature.model_input is False
    assert matrix_feature.get_path() == sample_arrays.matrix_zarr


def test_create_feature_configs_default_to_flexible(
    sample_arrays: TrainingArrays, sample_rhs: np.ndarray
) -> None:
    """_create_feature_configs defaults to in-memory rhs + path-dropped matrix behavior."""
    contract = default_training_dataset_contract()
    features = _create_feature_configs(sample_arrays, sample_rhs, contract, frozenset())

    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    rhs_feature = next(f for f in features if f.name == "x")
    assert isinstance(rhs_feature, ValueEntry)
    np.testing.assert_array_equal(rhs_feature.value, sample_rhs)
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert isinstance(matrix_feature, IPathBased)
    assert matrix_feature.get_path() == sample_arrays.matrix_zarr


def test_create_target_configs_returns_canonical_supervised_target(
    sample_solutions: np.ndarray,
) -> None:
    """_create_target_configs exposes exactly one in-memory runtime target entry."""
    contract = default_training_dataset_contract()
    targets = _create_target_configs(sample_solutions, contract)

    assert len(targets) == 1
    assert targets[0].name == "y"
    assert isinstance(targets[0], ValueEntry)
    assert targets[0].data_role == DataRole.TARGET
    np.testing.assert_array_equal(targets[0].value, sample_solutions)


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
    contract = default_training_dataset_contract()
    updated = _resolve_dataset(training_settings, features, targets, contract)
    assert updated is not training_settings
    assert updated.DATASET is not None
    assert updated.DATASET.features == tuple(features)
    assert updated.DATASET.targets == tuple(targets)
    assert training_settings.DATASET is not None
    assert training_settings.DATASET.features == ()
    assert training_settings.DATASET.targets == ()


def test_validate_runtime_dataset_contract_rejects_duplicate_target_names(
    training_settings: GeneralSettings,
) -> None:
    """Duplicate target names must fail fast before dataset injection."""
    duplicate_targets = DatasetSettings(
        name="FlexibleDataset",
        targets=(
            ValueEntry(
                name="y", value=np.zeros((1, 1), dtype=np.float64), data_role=DataRole.TARGET
            ),
            ValueEntry(
                name="y", value=np.ones((1, 1), dtype=np.float64), data_role=DataRole.TARGET
            ),
        ),
    )
    duplicate_settings = training_settings.model_copy(update={"DATASET": duplicate_targets})

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="Duplicate DATASET target entry names"):
        _validate_runtime_dataset_contract(duplicate_settings, contract)


def test_validate_runtime_dataset_contract_rejects_duplicate_feature_names(
    training_settings: GeneralSettings,
) -> None:
    """Duplicate feature names must fail fast before dataset injection."""
    duplicate_features = DatasetSettings(
        name="FlexibleDataset",
        features=(
            ValueEntry(name="x", value=np.zeros((1, 1), dtype=np.float64)),
            ValueEntry(name="x", value=np.ones((1, 1), dtype=np.float64)),
        ),
    )
    duplicate_settings = training_settings.model_copy(update={"DATASET": duplicate_features})

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="Duplicate DATASET feature entry names"):
        _validate_runtime_dataset_contract(duplicate_settings, contract)


def test_validate_runtime_dataset_contract_rejects_unsupported_target_placeholder(
    training_settings: GeneralSettings,
) -> None:
    """Runtime target placeholders must match the resolved contract name."""
    aliased_settings = training_settings.model_copy(
        update={
            "DATASET": DatasetSettings(
                name="FlexibleDataset",
                targets=(
                    ValueEntry(
                        name="solutions",
                        value=np.zeros((1, 1), dtype=np.float64),
                        data_role=DataRole.TARGET,
                    ),
                ),
            )
        }
    )

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="resolved supervised target name 'y'"):
        _validate_runtime_dataset_contract(aliased_settings, contract)


def test_validate_runtime_dataset_contract_rejects_noncanonical_loss_target_key(
    training_settings: GeneralSettings,
) -> None:
    """Loss routing must target the canonical supervised entry."""
    assert training_settings.TRAINING is not None
    updated = training_settings.model_copy(
        update={
            "TRAINING": training_settings.TRAINING.model_copy(
                update={
                    "loss_function": MagicMock(
                        spec=["target_key"],
                        **{"target_key": "targets.solutions"},
                    )
                }
            )
        }
    )

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="TRAINING.loss_function.target_key must resolve"):
        _validate_runtime_dataset_contract(updated, contract)


def test_resolve_dataset_rejects_unmatched_target_placeholder(
    training_settings: GeneralSettings,
) -> None:
    """Placeholder target entries must match injected runtime names exactly."""
    placeholder_settings = training_settings.model_copy(
        update={
            "DATASET": DatasetSettings(
                name="FlexibleDataset",
                targets=(
                    ValueEntry(
                        name="z",
                        value=np.zeros((1, 1), dtype=np.float64),
                        data_role=DataRole.TARGET,
                    ),
                ),
            )
        }
    )

    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="resolved supervised target name 'y'"):
        _resolve_dataset(placeholder_settings, [], [], contract)


def test_resolve_dataset_rejects_unmatched_feature_placeholder(
    training_settings: GeneralSettings,
) -> None:
    """Placeholder feature entries must match injected runtime names exactly."""
    placeholder_settings = training_settings.model_copy(
        update={
            "DATASET": DatasetSettings(
                name="FlexibleDataset",
                features=(ValueEntry(name="rhs", value=np.zeros((1, 1), dtype=np.float64)),),
            )
        }
    )

    contract = default_training_dataset_contract()
    with pytest.raises(
        ValueError, match="DATASET feature placeholders must use only the resolved runtime"
    ):
        _resolve_dataset(placeholder_settings, [], [], contract)


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


def test_contract_override_drives_injection_and_validation(
    sample_arrays: TrainingArrays,
    sample_rhs: np.ndarray,
    sample_solutions: np.ndarray,
    training_settings: GeneralSettings,
) -> None:
    """Contract-driven helpers honor non-default runtime names."""
    assert training_settings.TRAINING is not None
    contract = RuntimeDatasetContract(
        primary_input_name="lhs",
        matrix_input_name="matrix",
        target_name="rhs",
        prediction_name="rhs_pred",
        loss_target_key="targets.rhs",
    )
    features = _create_feature_configs(sample_arrays, sample_rhs, contract, frozenset())
    targets = _create_target_configs(sample_solutions, contract)
    settings = training_settings.model_copy(
        update={
            "DATASET": DatasetSettings(
                name="FlexibleDataset",
                features=(ValueEntry(name="lhs", value=np.zeros((1, 1), dtype=np.float64)),),
                targets=(
                    ValueEntry(
                        name="rhs",
                        value=np.zeros((1, 1), dtype=np.float64),
                        data_role=DataRole.TARGET,
                    ),
                ),
            ),
            "TRAINING": training_settings.TRAINING.model_copy(
                update={
                    "loss_function": MagicMock(spec=["target_key"], **{"target_key": "targets.rhs"})
                }
            ),
        }
    )

    _validate_runtime_dataset_contract(settings, contract)
    updated = _resolve_dataset(settings, features, targets, contract)

    assert updated.DATASET is not None
    assert [feature.name for feature in updated.DATASET.features] == ["lhs", "matrix"]
    assert [target.name for target in updated.DATASET.targets] == ["rhs"]


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

    contract = default_training_dataset_contract()
    normalized_payload = {
        "predictions": {contract.prediction_name: np.zeros((10, 1))},
        "targets": {"y": np.zeros((10, 1))},
    }
    mock_compute.return_value = MagicMock()
    mock_write.return_value = Path("dummy.png")
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"

    _log_training_evaluation(
        tracking_uri=tracking_uri,
        run_id="run123",
        numpy_payload=normalized_payload,
        figures_dir=tmp_path / "figures",
        contract=contract,
    )

    mock_compute.assert_called_once()
    mock_write.assert_called_once()
    mock_mlflow_log.assert_called_once()


def test_extract_evaluation_arrays_with_canonical_prediction_key() -> None:
    """Extraction uses one explicit prediction and target key from the contract."""
    contract = default_training_dataset_contract()
    selected = _extract_evaluation_arrays(
        {
            "predictions": {contract.prediction_name: np.array([[1.0], [2.0], [3.0]])},
            "targets": {contract.target_name: np.array([[1.1], [1.9], [3.2]])},
        },
        contract,
    )
    assert selected is not None
    y_pred, y_true = selected
    np.testing.assert_allclose(y_pred.ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_true.ravel(), [1.1, 1.9, 3.2])


def test_extract_evaluation_arrays_rejects_noncanonical_prediction_key() -> None:
    """Extraction no longer tolerates fallback prediction aliases."""
    contract = default_training_dataset_contract()
    assert (
        _extract_evaluation_arrays(
            {
                "predictions": {"y_hat": np.array([[1.0], [2.0]])},
                "targets": {contract.target_name: np.array([[0.8], [2.2]])},
            },
            contract,
        )
        is None
    )


def test_extract_evaluation_arrays_returns_none_when_missing_keys() -> None:
    """Extraction returns None when predictions/targets are unavailable."""
    contract = default_training_dataset_contract()
    assert _extract_evaluation_arrays({"predictions": np.array([1.0, 2.0])}, contract) is None


def test_normalize_training_numpy_payload_maps_dlkit_output_once() -> None:
    """The DLKit boundary output key is normalized into the contract prediction key."""
    contract = default_training_dataset_contract()
    normalized = _normalize_training_numpy_payload(
        {
            "predictions": {"output": np.array([[1.0], [2.0]])},
            "targets": {contract.target_name: np.array([[0.8], [2.2]])},
        },
        contract,
    )

    assert normalized is not None
    assert set(normalized["predictions"]) == {contract.prediction_name}
    np.testing.assert_allclose(
        normalized["predictions"][contract.prediction_name].ravel(),
        [1.0, 2.0],
    )


def test_normalize_training_numpy_payload_rejects_unknown_prediction_shape() -> None:
    """Normalization fails fast when no accepted boundary prediction key exists."""
    contract = default_training_dataset_contract()
    with pytest.raises(ValueError, match="canonical prediction key"):
        _normalize_training_numpy_payload(
            {
                "predictions": {"y_hat": np.array([[1.0], [2.0]])},
                "targets": {contract.target_name: np.array([[0.8], [2.2]])},
            },
            contract,
        )


def test_resolve_mlflow_logging_config_reads_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MLflow logging now resolves infra from runtime env only."""
    expected_tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_uri = str(tmp_path / "mlartifacts")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", expected_tracking_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", artifact_uri)

    tracking_uri, artifacts_destination = resolve_runtime_tracking_config()
    assert tracking_uri == expected_tracking_uri
    assert artifacts_destination == artifact_uri


def test_resolve_mlflow_logging_config_handles_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing runtime env returns an empty MLflow infra payload."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    tracking_uri, artifacts_destination = resolve_runtime_tracking_config()
    assert tracking_uri is None
    assert artifacts_destination == ""
