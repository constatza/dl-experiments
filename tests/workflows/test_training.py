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
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from dlkit.interfaces.api.domain.models import TrainingResult


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
                "predictions": TensorDict({"output": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3]),
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
def training_result_tensor_predictions(predictions_as_tensors: list[torch.Tensor]) -> TrainingResult:
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


def test_fast_dev_run_predict_returns_list_of_dicts(tmp_path: Path) -> None:
    """trainer.predict() produces stacked TensorDict and to_numpy() output.

    This test documents the expected API so _log_training_evaluation() can rely on it.
    Runs a single fast_dev_run training step using DLKit programmatic settings.
    """
    from dlkit.interfaces.api import execute
    from dlkit.tools.config import GeneralSettings, SessionSettings
    from dlkit.tools.config import DataModuleSettings, DatasetSettings, TrainingSettings
    from dlkit.tools.config.dataloader_settings import DataloaderSettings
    from dlkit.tools.config.trainer_settings import TrainerSettings
    from dlkit.tools.config.components.model_components import (
        ModelComponentSettings,
        MetricComponentSettings,
    )
    from dlkit.tools.config.data_entries import ValueFeature, ValueTarget

    n_samples, n_features, n_targets = 20, 4, 2
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))
    Y = rng.random((n_samples, n_targets))

    settings = GeneralSettings(
        SESSION=SessionSettings(name="test_predict_structure", seed=42),
        MLFLOW=None,
        OPTUNA=None,
        DATAMODULE=DataModuleSettings(
            dataloader=DataloaderSettings(batch_size=4, num_workers=0),
        ),
        DATASET=DatasetSettings(
            name="FlexibleDataset",
            features=[ValueFeature(name="x", value=X)],
            targets=[ValueTarget(name="y", value=Y)],
        ),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                fast_dev_run=True,
                enable_checkpointing=False,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=str(tmp_path),
            ),
            metrics=(
                MetricComponentSettings(
                    name="MeanSquaredError",
                    module_path="dlkit.core.training.metrics",
                ),
            ),
        ),
        MODEL=ModelComponentSettings(
            name="ConstantWidthFFNN",
            module_path="dlkit.core.models.nn.ffnn.simple",
            hidden_size=4,
            num_layers=1,
        ),
    )

    result = execute(settings)

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
