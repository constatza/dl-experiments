"""Tests for training workflow and TrainingResult.stacked_predictions().

Covers:
- stacked_predictions() with predict_step dict format
- stacked_predictions() with plain tensor batches
- stacked_predictions() returns None when no predictions captured
- Fast-dev-run integration: trainer.predict() returns expected dict structure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dlkit.interfaces.api.domain.models import TrainingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def predictions_as_dicts() -> list[dict[str, Any]]:
    """Prediction batches in predict_step dict format.

    Mirrors: {"predictions": {"output": tensor}, "targets": {...}, "latents": {}}
    """
    return [
        {"predictions": {"output": torch.tensor([1.0, 2.0, 3.0])}, "targets": {}, "latents": {}},
        {"predictions": {"output": torch.tensor([4.0, 5.0])}, "targets": {}, "latents": {}},
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
    """TrainingResult with no captured predictions."""
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
def training_result_dict_predictions(predictions_as_dicts: list[dict]) -> TrainingResult:
    """TrainingResult with predict_step dict format predictions."""
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


# ---------------------------------------------------------------------------
# stacked_predictions() unit tests
# ---------------------------------------------------------------------------


def test_stacked_predictions_none_when_no_predictions(
    training_result_no_predictions: TrainingResult,
) -> None:
    """Returns None when predictions field is None."""
    assert training_result_no_predictions.stacked_predictions() is None


def test_stacked_predictions_none_when_empty_list(
    training_result_empty_predictions: TrainingResult,
) -> None:
    """Returns None when predictions is an empty list."""
    assert training_result_empty_predictions.stacked_predictions() is None


def test_stacked_predictions_from_dict_format(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """Correctly extracts and concatenates from predict_step dict format.

    predict_step returns: {"predictions": {"output": tensor}, "targets": {}, "latents": {}}
    stacked_predictions() should yield array of shape (5,) from two batches of 3+2.
    """
    result = training_result_dict_predictions.stacked_predictions()

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_stacked_predictions_from_tensor_format(
    training_result_tensor_predictions: TrainingResult,
) -> None:
    """Correctly stacks plain tensor batches without dict wrapping.

    Batches of shape (2, 2) + (1, 2) → (3, 2).
    """
    result = training_result_tensor_predictions.stacked_predictions()

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)


def test_stacked_predictions_returns_numpy_array(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """stacked_predictions() always returns numpy, not torch.Tensor."""
    result = training_result_dict_predictions.stacked_predictions()

    assert result is not None
    assert not isinstance(result, torch.Tensor)
    assert isinstance(result, np.ndarray)


@pytest.fixture
def predictions_with_targets() -> list[dict[str, Any]]:
    """Prediction batches with matching targets in predict_step format."""
    return [
        {
            "predictions": {"output": torch.tensor([1.0, 2.0])},
            "targets": {"y": torch.tensor([1.1, 2.1])},
            "latents": {},
        },
        {
            "predictions": {"output": torch.tensor([3.0])},
            "targets": {"y": torch.tensor([3.1])},
            "latents": {},
        },
    ]


@pytest.fixture
def training_result_with_targets(predictions_with_targets: list[dict]) -> TrainingResult:
    """TrainingResult with both predictions and targets."""
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.0,
        predictions=predictions_with_targets,
    )


def test_stacked_targets_matches_predict_step_targets(
    training_result_with_targets: TrainingResult,
) -> None:
    """stacked_targets() extracts targets from predict_step dict format."""
    targets = training_result_with_targets.stacked_targets()

    assert targets is not None
    assert isinstance(targets, np.ndarray)
    assert targets.shape == (3,)
    np.testing.assert_allclose(targets, [1.1, 2.1, 3.1], atol=1e-6)


def test_stacked_targets_aligns_with_predictions(
    training_result_with_targets: TrainingResult,
) -> None:
    """Predictions and targets have matching shapes (same data split)."""
    preds = training_result_with_targets.stacked_predictions()
    targets = training_result_with_targets.stacked_targets()

    assert preds is not None
    assert targets is not None
    assert preds.shape == targets.shape


def test_stacked_targets_none_when_no_targets(
    training_result_dict_predictions: TrainingResult,
) -> None:
    """stacked_targets() returns None when targets dict is empty in all batches."""
    targets = training_result_dict_predictions.stacked_targets()
    # predictions_as_dicts has empty targets dicts — no values to extract
    assert targets is None


# ---------------------------------------------------------------------------
# fast_dev_run integration: verify trainer.predict() dict structure
# ---------------------------------------------------------------------------


def test_fast_dev_run_predict_returns_list_of_dicts(tmp_path: Path) -> None:
    """trainer.predict() returns list[dict] matching predict_step output format.

    This test documents the expected structure so stacked_predictions() can rely on it.
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
    assert result.predictions is not None, "trainer.predict() should have been called and captured"
    assert isinstance(result.predictions, list)
    assert len(result.predictions) > 0

    # Verify each batch has the expected predict_step dict structure
    first_batch = result.predictions[0]
    assert isinstance(first_batch, dict), f"Expected dict, got {type(first_batch)}"
    assert "predictions" in first_batch, f"Expected 'predictions' key, got {list(first_batch.keys())}"

    # Verify stacked_predictions() produces a valid array
    stacked = result.stacked_predictions()
    assert stacked is not None
    assert isinstance(stacked, np.ndarray)
    assert stacked.ndim >= 1

    # Verify stacked_targets() aligns with predictions (same split, same count)
    targets = result.stacked_targets()
    assert targets is not None, "Targets should be captured alongside predictions"
    assert targets.shape == stacked.shape, (
        f"Predictions shape {stacked.shape} must match targets shape {targets.shape}"
    )