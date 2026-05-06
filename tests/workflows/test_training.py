"""Tests for TrainingResult.stacked and TrainingResult.to_numpy() API.

Covers:
- stacked is None when no predictions captured
- to_numpy() returns None when no predictions captured
- stacked is a TensorDict when TensorDict predictions are present
- to_numpy() extracts predictions and targets correctly
- fast-dev-run integration: trainer.predict() produces stacked/to_numpy output
"""

from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from neuralls.platform.config.resolution import build_sqlite_tracking_uri
import torch
from tensordict import TensorDict

from dlkit.common.results import TrainingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def predictions_as_dicts() -> list[TensorDict]:
    """Prediction batches in TensorDict predict_step format.

    Mirrors the structure dlkit predict_step returns:
    {"predictions": TensorDict({"output": tensor}), "targets": TensorDict({})}
    """
    return [
        TensorDict(
            {
                "predictions": TensorDict(
                    {"output": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3]
                ),
                "targets": TensorDict({}, batch_size=[3]),
            },
            batch_size=[3],
        ),
        TensorDict(
            {
                "predictions": TensorDict({"output": torch.tensor([4.0, 5.0])}, batch_size=[2]),
                "targets": TensorDict({}, batch_size=[2]),
            },
            batch_size=[2],
        ),
    ]


@pytest.fixture
def predictions_as_tensors() -> list[torch.Tensor]:
    """Prediction batches as plain tensors (fallback format)."""
    return [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
    ]


@pytest.fixture
def training_result_no_predictions() -> TrainingResult:
    """TrainingResult with no captured predictions (None)."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=None,
    )


@pytest.fixture
def training_result_empty_predictions() -> TrainingResult:
    """TrainingResult with empty prediction list."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=[],
    )


@pytest.fixture
def training_result_dict_predictions(predictions_as_dicts: list[TensorDict]) -> TrainingResult:
    """TrainingResult with TensorDict-format predictions (no targets)."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_as_dicts,
    )


@pytest.fixture
def training_result_tensor_predictions(
    predictions_as_tensors: list[torch.Tensor],
) -> TrainingResult:
    """TrainingResult with plain tensor format predictions."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_as_tensors,
    )


@pytest.fixture
def predictions_with_targets() -> list[TensorDict]:
    """Prediction batches with matching targets in TensorDict format."""
    return [
        TensorDict(
            {
                "predictions": TensorDict({"output": torch.tensor([1.0, 2.0])}, batch_size=[2]),
                "targets": TensorDict({"y": torch.tensor([1.1, 2.1])}, batch_size=[2]),
            },
            batch_size=[2],
        ),
        TensorDict(
            {
                "predictions": TensorDict({"output": torch.tensor([3.0])}, batch_size=[1]),
                "targets": TensorDict({"y": torch.tensor([3.1])}, batch_size=[1]),
            },
            batch_size=[1],
        ),
    ]


@pytest.fixture
def training_result_with_targets(predictions_with_targets: list[TensorDict]) -> TrainingResult:
    """TrainingResult with both predictions and targets."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_with_targets,
    )


# ---------------------------------------------------------------------------
# stacked property tests
# ---------------------------------------------------------------------------


def test_stacked_none_when_no_predictions(
    training_result_no_predictions: TrainingResult,
) -> None:
    """result.stacked is None when predictions field is None."""
    assert training_result_no_predictions.stacked is None


def test_stacked_none_when_empty_predictions(
    training_result_empty_predictions: TrainingResult,
) -> None:
    """result.stacked is None when predictions is an empty list."""
    assert training_result_empty_predictions.stacked is None


def test_stacked_is_tensordict_for_tensordict_input(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """result.stacked is a TensorDict when TensorDict predictions are stored."""
    assert isinstance(training_result_dict_predictions.stacked, TensorDict)


# ---------------------------------------------------------------------------
# to_numpy() tests
# ---------------------------------------------------------------------------


def test_to_numpy_returns_none_when_no_predictions(
    training_result_no_predictions: TrainingResult,
) -> None:
    """to_numpy() returns None when no predictions were captured."""
    assert training_result_no_predictions.to_numpy() is None


def test_to_numpy_extracts_predictions_from_tensordict(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """to_numpy() extracts and concatenates output across TensorDict batches.

    Two batches of size 3 and 2 → flat ndarray of shape (5,).
    """
    result = training_result_dict_predictions.to_numpy()

    assert result is not None
    output = result.get("predictions", {}).get("output")
    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.shape == (5,)
    np.testing.assert_allclose(output, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_to_numpy_extracts_targets_when_present(
    training_result_with_targets: TrainingResult,
) -> None:
    """to_numpy() extracts targets from TensorDict predictions."""
    result = training_result_with_targets.to_numpy()

    assert result is not None
    targets = result.get("targets", {})
    assert "y" in targets
    y = targets["y"]
    assert isinstance(y, np.ndarray)
    assert y.shape == (3,)
    np.testing.assert_allclose(y, [1.1, 2.1, 3.1], atol=1e-6)


def test_to_numpy_targets_empty_when_no_targets(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """to_numpy()['targets'] is empty when targets dicts are empty in all batches."""
    result = training_result_dict_predictions.to_numpy()
    assert result is not None
    targets = result.get("targets", {})
    assert targets == {}


def test_to_numpy_predictions_shape_matches_targets(
    training_result_with_targets: TrainingResult,
) -> None:
    """Predictions and targets arrays have matching shapes from to_numpy()."""
    result = training_result_with_targets.to_numpy()
    assert result is not None
    output = result["predictions"]["output"]
    y = result["targets"]["y"]
    assert output.shape == y.shape


# ---------------------------------------------------------------------------
# fast_dev_run integration: verify trainer.predict() structure
# ---------------------------------------------------------------------------


def test_fast_dev_run_predict_returns_list_of_dicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trainer.predict() produces stacked TensorDict and to_numpy() output.

    This test documents the expected API so _log_training_evaluation() can rely on it.
    Runs a single fast_dev_run training step using DLKit programmatic settings.
    """
    import dlkit.engine.tracking.uri_resolver as uri_resolver
    from dlkit.interfaces.api import execute
    from dlkit.infrastructure.config import SessionSettings
    from dlkit.infrastructure.config import DataModuleSettings, DatasetSettings, TrainingSettings
    from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
    from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
    from dlkit.infrastructure.config.trainer_settings import TrainerSettings
    from dlkit.infrastructure.config.model_components import (
        ModelComponentSettings,
        MetricComponentSettings,
    )
    from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig
    from dlkit.infrastructure.config.data_entries import ValueFeature, ValueTarget

    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    n_samples, n_features, n_targets = 20, 4, 2
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_targets))

    settings = TrainingWorkflowConfig(
        SESSION=SessionSettings(name="test_predict_structure", seed=42, workflow="train"),
        MLFLOW=MLflowSettings(experiment_name="test_predict_structure"),
        OPTUNA={"enabled": False},
        DATAMODULE=DataModuleSettings(
            dataloader=DataloaderSettings(batch_size=4, num_workers=0),
        ),
        DATASET=DatasetSettings(
            name="FlexibleDataset",
            features=(ValueFeature(name="x", value=X),),
            targets=(ValueTarget(name="y", value=Y),),
        ),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                fast_dev_run=True,
                enable_checkpointing=False,
                accelerator="cpu",
                default_root_dir=tmp_path,
            ),
            metrics=(
                MetricComponentSettings(
                    name="MeanSquaredError",
                    module_path="dlkit.domain.metrics",
                ),
            ),
        ),
        MODEL=ModelComponentSettings(
            name="ConstantWidthFFNN",
            module_path="dlkit.nn",
            hidden_size=4,
            num_layers=1,
        ),
    )

    result = cast(TrainingResult, execute(settings))

    # Verify predictions were captured
    assert result.predictions is not None, "trainer.predict() should have captured predictions"
    assert isinstance(result.predictions, list)
    assert len(result.predictions) > 0

    # Verify stacked TensorDict is accessible
    stacked = result.stacked
    assert stacked is not None, "result.stacked should be a TensorDict after prediction"

    # Verify to_numpy() returns the expected dict structure
    all_numpy = result.to_numpy()
    assert all_numpy is not None, "to_numpy() should return a dict after prediction"
    assert "predictions" in all_numpy, f"Expected 'predictions' key, got {list(all_numpy.keys())}"
    preds = all_numpy["predictions"]
    # predictions may be a numpy array directly or a dict with sub-keys
    if isinstance(preds, dict):
        output = next(iter(preds.values()))
    else:
        output = preds
    assert isinstance(output, np.ndarray)
    assert output.ndim >= 1


def test_train_model_passes_explicit_mlflow_run_config_to_execute(tmp_path: Path) -> None:
    """Registry-backed training passes explicit experiment/run names and structured tags."""
    from neuralls.platform.config.models.workspace import ExperimentWorkspace
    from neuralls.composition.experiments.training import train_model

    config_path = tmp_path / "model.toml"
    data_config_path = tmp_path / "data.toml"
    config_path.write_text("")
    data_config_path.write_text("")

    workspace = ExperimentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)
    checkpoint_path = workspace.checkpoint_dir / "model.ckpt"
    checkpoint_path.write_text("checkpoint")

    experiment = SimpleNamespace(
        settings=object(),
        workspace=workspace,
        spec=SimpleNamespace(
            experiment_id="exp-1",
            experiment_display_name="Experiment One",
            dataset_registry_id="dataset-1",
            dataset_display_name="Dataset One",
            model_registry_id="model-1",
            model_display_name="Model One",
        ),
    )
    training_result = SimpleNamespace(run_id="run-123", metrics={})

    with (
        patch(
            "neuralls.composition.experiments.training.load_experiment", return_value=experiment
        ) as mock_load,
        patch(
            "neuralls.composition.experiments.training._load_and_prepare_data",
            return_value=(None, [], []),
        ),
        patch(
            "neuralls.composition.experiments.training._configure_training_pipeline",
            return_value=(experiment.settings, workspace),
        ),
        patch(
            "neuralls.composition.experiments.training.execute", return_value=training_result
        ) as mock_execute,
        patch(
            "neuralls.composition.experiments.training.get_latest_checkpoint",
            return_value=checkpoint_path,
        ),
        patch(
            "neuralls.composition.experiments.training._resolve_mlflow_logging_config",
            return_value=(
                build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
                str((tmp_path / "mlartifacts").resolve()),
            ),
        ),
        patch(
            "neuralls.composition.experiments.training._resolve_mlflow_run_ids",
            return_value=(
                build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
                "mlflow-exp-1",
                "run-123",
            ),
        ),
        patch("neuralls.composition.experiments.training._log_training_context"),
        patch("neuralls.composition.experiments.training.write_mlflow_sidecar"),
        patch("neuralls.composition.experiments.training._stage_training_artifacts"),
        patch("neuralls.composition.experiments.training._log_training_evaluation"),
        patch("neuralls.composition.experiments.training.log_artifacts_to_mlflow"),
    ):
        train_model(
            config_path=str(config_path),
            data_config_path=str(data_config_path),
            output_root=tmp_path / "output",
            experiment_id="exp-1",
            experiment_display_name="Experiment One",
            dataset_registry_id="dataset-1",
            dataset_display_name="Dataset One",
            model_registry_id="model-1",
            model_display_name="Model One",
            mlflow_experiment_name="Train",
        )

    load_args = mock_load.call_args.args
    assert load_args[0] == config_path
    assert load_args[1] == data_config_path

    execute_kwargs = mock_execute.call_args.kwargs
    overrides = execute_kwargs["overrides"]
    assert overrides.experiment_name == "Train"
    assert re.match(
        r"^Experiment One-[A-Z][a-z]{2} \d{2} [A-Z][a-z]{2} \d{4} - \d{2}:\d{2}:\d{2}$",
        overrides.run_name,
    )
    assert overrides.tags == {
        "phase": "training",
        "experiment_id": "exp-1",
        "dataset_id": "dataset-1",
        "model_id": "model-1",
        "experiment_display_name": "Experiment One",
    }


def test_train_model_falls_back_to_dataset_display_name_without_structured_tags(
    tmp_path: Path,
) -> None:
    """Legacy callers without explicit experiment names use dataset display name and no tags."""
    from neuralls.platform.config.models.workspace import ExperimentWorkspace
    from neuralls.composition.experiments.training import train_model

    config_path = tmp_path / "model.toml"
    data_config_path = tmp_path / "data.toml"
    config_path.write_text("")
    data_config_path.write_text("")

    workspace = ExperimentWorkspace(
        dataset_id="dataset-legacy",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)
    checkpoint_path = workspace.checkpoint_dir / "model.ckpt"
    checkpoint_path.write_text("checkpoint")

    experiment = SimpleNamespace(
        settings=object(),
        workspace=workspace,
        spec=SimpleNamespace(
            experiment_id="legacy-exp",
            experiment_display_name="Legacy Experiment",
            dataset_registry_id="dataset-legacy",
            dataset_display_name="Dataset Display",
            model_registry_id=None,
            model_display_name=None,
        ),
    )

    with (
        patch("neuralls.composition.experiments.training.load_experiment", return_value=experiment),
        patch(
            "neuralls.composition.experiments.training._load_and_prepare_data",
            return_value=(None, [], []),
        ),
        patch(
            "neuralls.composition.experiments.training._configure_training_pipeline",
            return_value=(experiment.settings, workspace),
        ),
        patch(
            "neuralls.composition.experiments.training.execute",
            return_value=SimpleNamespace(run_id="run-123", metrics={}),
        ) as mock_execute,
        patch(
            "neuralls.composition.experiments.training.get_latest_checkpoint",
            return_value=checkpoint_path,
        ),
        patch(
            "neuralls.composition.experiments.training._resolve_mlflow_logging_config",
            return_value=(
                build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
                str((tmp_path / "mlartifacts").resolve()),
            ),
        ),
        patch(
            "neuralls.composition.experiments.training._resolve_mlflow_run_ids",
            return_value=(
                build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
                "mlflow-exp-1",
                "run-123",
            ),
        ),
        patch("neuralls.composition.experiments.training._log_training_context"),
        patch("neuralls.composition.experiments.training.write_mlflow_sidecar"),
        patch("neuralls.composition.experiments.training._stage_training_artifacts"),
        patch("neuralls.composition.experiments.training._log_training_evaluation"),
        patch("neuralls.composition.experiments.training.log_artifacts_to_mlflow"),
    ):
        train_model(
            config_path=config_path,
            data_config_path=data_config_path,
            output_root=tmp_path / "output",
            experiment_id="legacy-exp",
            experiment_display_name="Legacy Experiment",
            dataset_registry_id="dataset-legacy",
            dataset_display_name="Dataset Display",
        )

    execute_kwargs = mock_execute.call_args.kwargs
    overrides = execute_kwargs["overrides"]
    assert overrides.experiment_name == "Train"
    assert re.match(
        r"^Legacy Experiment-[A-Z][a-z]{2} \d{2} [A-Z][a-z]{2} \d{4} - \d{2}:\d{2}:\d{2}$",
        overrides.run_name,
    )
    assert overrides.tags == {}


def test_train_model_max_epochs_override_keeps_original_settings_immutable(
    tmp_path: Path,
) -> None:
    """`max_epochs` override patches a fresh settings object before execute()."""
    from dlkit.infrastructure.config import GeneralSettings
    from dlkit.infrastructure.config import (
        DatasetSettings,
        ModelComponentSettings as ModelSettings,
        SessionSettings,
        TrainingSettings,
    )
    from dlkit.infrastructure.config.trainer_settings import TrainerSettings
    from neuralls.platform.config.models.workspace import ExperimentWorkspace
    from neuralls.composition.experiments.training import train_model

    config_path = tmp_path / "model.toml"
    data_config_path = tmp_path / "data.toml"
    config_path.write_text("")
    data_config_path.write_text("")
    trainer_root = tmp_path / "trainer-root"
    trainer_root.mkdir()

    base_settings = GeneralSettings(
        SESSION=SessionSettings(seed=42),
        MODEL=ModelSettings(name="LinearModel"),
        DATASET=DatasetSettings(name="FlexibleDataset"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(max_epochs=1, default_root_dir=trainer_root)
        ),
    )

    workspace = ExperimentWorkspace(
        dataset_id="dataset-1",
        run_id="workspace-run",
        root_dir=tmp_path / "workspace",
        data_dir=tmp_path / "workspace" / "data",
    )
    workspace.checkpoint_dir.mkdir(parents=True)
    checkpoint_path = workspace.checkpoint_dir / "model.ckpt"
    checkpoint_path.write_text("checkpoint")

    experiment = SimpleNamespace(
        settings=base_settings,
        workspace=workspace,
        spec=SimpleNamespace(
            experiment_id="exp-1",
            experiment_display_name="Experiment One",
            dataset_registry_id="dataset-1",
            dataset_display_name="Dataset One",
            model_registry_id=None,
            model_display_name=None,
        ),
    )

    with (
        patch("neuralls.composition.experiments.training.load_experiment", return_value=experiment),
        patch(
            "neuralls.composition.experiments.training._load_and_prepare_data",
            return_value=(None, [], []),
        ),
        patch(
            "neuralls.composition.experiments.training._configure_training_pipeline",
            return_value=(base_settings, workspace),
        ),
        patch(
            "neuralls.composition.experiments.training.execute",
            return_value=SimpleNamespace(run_id="run-123", metrics={}),
        ) as mock_execute,
        patch(
            "neuralls.composition.experiments.training.get_latest_checkpoint",
            return_value=checkpoint_path,
        ),
        patch(
            "neuralls.composition.experiments.training._resolve_mlflow_logging_config",
            return_value=(
                build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
                str((tmp_path / "mlartifacts").resolve()),
            ),
        ),
        patch(
            "neuralls.composition.experiments.training._resolve_mlflow_run_ids",
            return_value=(
                build_sqlite_tracking_uri(tmp_path / "mlflow.db"),
                "mlflow-exp-1",
                "run-123",
            ),
        ),
        patch("neuralls.composition.experiments.training._log_training_context"),
        patch("neuralls.composition.experiments.training.write_mlflow_sidecar"),
        patch("neuralls.composition.experiments.training._stage_training_artifacts"),
        patch("neuralls.composition.experiments.training._log_training_evaluation"),
        patch("neuralls.composition.experiments.training.log_artifacts_to_mlflow"),
    ):
        train_model(
            config_path=config_path,
            data_config_path=data_config_path,
            output_root=tmp_path / "output",
            experiment_id="exp-1",
            experiment_display_name="Experiment One",
            dataset_registry_id="dataset-1",
            dataset_display_name="Dataset One",
            max_epochs=9,
        )

    execute_settings = mock_execute.call_args.args[0]
    assert execute_settings.TRAINING is not None
    assert execute_settings.TRAINING.trainer is not None
    assert execute_settings.TRAINING.trainer.max_epochs == 9
    assert base_settings.TRAINING is not None
    assert base_settings.TRAINING.trainer is not None
    assert base_settings.TRAINING.trainer.max_epochs == 1


def test_execute_result_unwraps_optimization_result() -> None:
    """Optimization results are normalized to their nested training result."""
    from dlkit.common.results import OptimizationResult, TrialRecord
    from neuralls.composition.experiments.training import _unwrap_execution_result

    training_result = TrainingResult(
        model_state=None,
        metrics={"val_loss": 0.1},
        artifacts={},
        duration_seconds=1.0,
    )
    optimization_result = OptimizationResult(
        best_trial=TrialRecord(number=1, value=0.1, params={}, state="COMPLETE"),
        training_result=training_result,
        study_summary={},
        duration_seconds=2.0,
    )

    assert _unwrap_execution_result(training_result) is training_result
    assert _unwrap_execution_result(optimization_result) is training_result
