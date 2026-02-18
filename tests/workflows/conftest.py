"""Shared pytest fixtures for workflow tests.

Provides composable, pure-data fixtures for ComparisonRun, TrainingRunResult,
and preconditioner spec objects used across test_comparison_run.py and
test_comparison_workflow.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.configuration.preconditioner import NeuralPreconditionerConfig, PreconditionerType, StandardPreconditionerConfig
from neuralls.workflows.comparison_run import ComparisonRun
from neuralls.workflows.multi_training import TrainingRunResult

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

PARENT_RUN_ID: str = "aaaabbbb-0000-1111-2222-ccccddddeeee"
TRACKING_URI: str = "sqlite:///mlruns.db"
EXP_ID_ALPHA: str = "alpha-experiment"
EXP_ID_BETA: str = "beta-experiment"
SOLVER_STEM: str = "default"
MLFLOW_EXPERIMENT_ID: str = "42"


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpoint_alpha(tmp_path: Path) -> Path:
    """Fake checkpoint file for the alpha experiment.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to a placeholder checkpoint file.
    """
    ckpt = tmp_path / "alpha" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    return ckpt


@pytest.fixture
def checkpoint_beta(tmp_path: Path) -> Path:
    """Fake checkpoint file for the beta experiment.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to a placeholder checkpoint file.
    """
    ckpt = tmp_path / "beta" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    return ckpt


@pytest.fixture
def artifact_uri() -> str:
    """Fake MLflow artifact URI for the batch run.

    Returns:
        URI string as returned by ``mlflow.get_artifact_uri()``.
    """
    return "mlartifacts/42/aaaabbbb-0000-1111-2222-ccccddddeeee/artifacts"


@pytest.fixture
def artifact_location(tmp_path: Path) -> str:
    """Artifact directory for the comparisons DB.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        String path for the comparisons artifact directory.
    """
    return str(tmp_path / "comparisons" / "artifacts")


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Local directory used as a generic output path in tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        An existing directory path.
    """
    out = tmp_path / "output"
    out.mkdir(parents=True)
    return out


# ---------------------------------------------------------------------------
# TrainingRunResult fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ComparisonRun fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def comparison_run(
    checkpoint_alpha: Path,
    checkpoint_beta: Path,
    artifact_uri: str,
    artifact_location: str,
) -> ComparisonRun:
    """Pre-built ComparisonRun with two experiments in checkpoint_map.

    Args:
        checkpoint_alpha: Alpha checkpoint path.
        checkpoint_beta: Beta checkpoint path.
        artifact_uri: Fake MLflow artifact URI for the batch run.
        artifact_location: Comparisons artifact directory path.

    Returns:
        Frozen ComparisonRun ready for lookup tests.
    """
    return ComparisonRun(
        mlflow_run_id=PARENT_RUN_ID,
        mlflow_experiment_id=MLFLOW_EXPERIMENT_ID,
        tracking_uri=TRACKING_URI,
        artifact_location=artifact_location,
        checkpoint_map={
            EXP_ID_ALPHA: checkpoint_alpha,
            EXP_ID_BETA: checkpoint_beta,
        },
        artifact_uri=artifact_uri,
    )


# ---------------------------------------------------------------------------
# Preconditioner spec fixtures (real Pydantic models, no Mocks)
# ---------------------------------------------------------------------------


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
