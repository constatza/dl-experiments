"""Shared pytest fixtures for workflow tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.composition.experiments.multi_training import TrainingRunResult

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

EXP_ID_ALPHA: str = "alpha-experiment"
EXP_ID_BETA: str = "beta-experiment"


@pytest.fixture
def checkpoint_alpha(tmp_path: Path) -> Path:
    """Fake checkpoint file for the alpha experiment."""
    ckpt = tmp_path / "alpha" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    return ckpt


@pytest.fixture
def checkpoint_beta(tmp_path: Path) -> Path:
    """Fake checkpoint file for the beta experiment."""
    ckpt = tmp_path / "beta" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    return ckpt


@pytest.fixture
def result_alpha(checkpoint_alpha: Path) -> TrainingRunResult:
    """TrainingRunResult for the alpha experiment.

    Args:
        checkpoint_alpha: Path to the alpha checkpoint.

    Returns:
        Immutable TrainingRunResult.
    """
    return TrainingRunResult(
        label="1",
        experiment_id=EXP_ID_ALPHA,
        experiment_display_name=EXP_ID_ALPHA,
        checkpoint_path=checkpoint_alpha,
        mlflow_run_id="run-alpha-0001",
        metrics={"eval/loss": 0.12},
    )


@pytest.fixture
def result_beta(checkpoint_beta: Path) -> TrainingRunResult:
    """TrainingRunResult for the beta experiment.

    Args:
        checkpoint_beta: Path to the beta checkpoint.

    Returns:
        Immutable TrainingRunResult.
    """
    return TrainingRunResult(
        label="2",
        experiment_id=EXP_ID_BETA,
        experiment_display_name=EXP_ID_BETA,
        checkpoint_path=checkpoint_beta,
        mlflow_run_id="run-beta-0002",
        metrics={"eval/loss": 0.08},
    )


@pytest.fixture
def training_results(
    result_alpha: TrainingRunResult,
    result_beta: TrainingRunResult,
) -> list[TrainingRunResult]:
    """Two-element ordered list of training results.

    Args:
        result_alpha: Alpha experiment result.
        result_beta: Beta experiment result.

    Returns:
        List of TrainingRunResult instances.
    """
    return [result_alpha, result_beta]


@pytest.fixture
def jacobi_spec() -> StandardPreconditionerConfig:
    """Non-neural Jacobi preconditioner spec.

    Returns:
        StandardPreconditionerConfig for jacobi.
    """
    return StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)


@pytest.fixture
def neural_spec_with_checkpoint(checkpoint_alpha: Path) -> NeuralPreconditionerConfig:
    """Neural preconditioner spec with an explicit checkpoint_path.

    Args:
        checkpoint_alpha: Concrete checkpoint path.

    Returns:
        NeuralPreconditionerConfig with checkpoint_path set.
    """
    return NeuralPreconditionerConfig(
        name="neural-explicit",
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint_alpha,
    )


@pytest.fixture
def neural_spec_with_experiment() -> NeuralPreconditionerConfig:
    """Neural preconditioner spec with an experiment reference (no checkpoint).

    Returns:
        NeuralPreconditionerConfig with experiment set but checkpoint_path=None.
    """
    return NeuralPreconditionerConfig(
        name="neural-by-experiment",
        type=PreconditionerType.NEURAL,
        experiment=EXP_ID_ALPHA,
    )


@pytest.fixture
def neural_spec_invalid() -> NeuralPreconditionerConfig:
    """Neural preconditioner spec missing both checkpoint_path and experiment.

    Returns:
        NeuralPreconditionerConfig that should fail validation.
    """
    return NeuralPreconditionerConfig(
        name="neural-invalid",
        type=PreconditionerType.NEURAL,
    )
