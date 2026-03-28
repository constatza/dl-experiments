"""Direct unit tests for the neural preconditioner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.solver.preconditioners.implementations.neural import NeuralPreconditioner
from neuralls.domain.solver.preconditioners.ports import PredictorAdapter, PredictorPort


class DummyPredictor(PredictorPort):
    """Minimal predictor stub for direct neural preconditioner tests."""

    def __init__(self) -> None:
        self.cleaned_up = False

    def apply(self, residual: np.ndarray) -> np.ndarray:
        return residual * 0.5

    def cleanup(self) -> None:
        self.cleaned_up = True


class DummyAdapter(PredictorAdapter):
    """Adapter stub that returns a stable predictor instance."""

    def __init__(self, predictor: DummyPredictor) -> None:
        self.predictor = predictor

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> PredictorPort:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        return self.predictor


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    """Create a throwaway checkpoint path for adapter wiring."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("checkpoint")
    return checkpoint


def test_neural_preconditioner_applies_predictor_output(checkpoint_path: Path) -> None:
    """NeuralPreconditioner delegates `apply` to the predictor port."""
    predictor = DummyPredictor()
    preconditioner = NeuralPreconditioner(
        checkpoint_path=checkpoint_path,
        adapter=DummyAdapter(predictor),
    )

    residual = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    result = preconditioner.apply(residual)

    np.testing.assert_allclose(result, residual * 0.5)


def test_neural_preconditioner_cleanup_is_idempotent(checkpoint_path: Path) -> None:
    """Repeated cleanup calls are safe and reach the predictor once loaded."""
    predictor = DummyPredictor()
    preconditioner = NeuralPreconditioner(
        checkpoint_path=checkpoint_path,
        adapter=DummyAdapter(predictor),
    )

    preconditioner.cleanup()
    preconditioner.cleanup()

    assert predictor.cleaned_up is True
