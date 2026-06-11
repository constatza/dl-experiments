"""Test extra named inputs support in NeuralPreconditioner."""

import numpy as np
import pytest

from neuralls.domain.solver.preconditioners.implementations.neural import NeuralPreconditioner
from neuralls.domain.solver.preconditioners.ports import PredictorAdapter, PredictorPort


class _RecordingPredictor(PredictorPort):
    """Mock predictor that records calls for testing."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply(self, residual, **extra_inputs):
        self.calls.append({"residual": residual.copy(), "extra": dict(extra_inputs)})
        return residual.copy()

    def cleanup(self):
        pass


class _MockAdapter(PredictorAdapter):
    """Mock adapter that provides a recording predictor."""

    def __init__(self):
        self.predictor = _RecordingPredictor()

    def create_predictor(self, checkpoint_path, config_path=None, data_config_path=None):
        return self.predictor


@pytest.fixture
def adapter():
    """Fixture: mock adapter with recording predictor."""
    return _MockAdapter()


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Fixture: dummy checkpoint file path."""
    p = tmp_path / "model.ckpt"
    p.touch()
    return p


@pytest.fixture
def neural_with_matrix(adapter, dummy_checkpoint):
    """Fixture: NeuralPreconditioner expecting a matrix extra input."""
    return NeuralPreconditioner(
        checkpoint_path=dummy_checkpoint,
        adapter=adapter,
        extra_input_names=("matrix",),
    )


@pytest.fixture
def residual():
    """Fixture: sample residual vector."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def small_matrix():
    """Fixture: small identity-like matrix."""
    return np.eye(3, dtype=np.float64) * 2.0


def test_extra_input_names_stored(neural_with_matrix):
    """Test that extra_input_names property returns declared inputs."""
    assert neural_with_matrix.extra_input_names == ("matrix",)


def test_bind_inputs_filters_to_declared_names(neural_with_matrix, small_matrix):
    """Test that bind_inputs only keeps keys declared in extra_input_names."""
    neural_with_matrix.bind_inputs(matrix=small_matrix, ignored_key=np.ones(3))
    neural_with_matrix.apply(np.ones(3))
    call = neural_with_matrix._predictor.calls[-1]  # noqa: SLF001
    assert "matrix" in call["extra"]
    assert "ignored_key" not in call["extra"]


def test_apply_without_bind_sends_no_extras(adapter, dummy_checkpoint, residual):
    """Test that apply sends empty extras if bind_inputs was not called."""
    precond = NeuralPreconditioner(
        checkpoint_path=dummy_checkpoint,
        adapter=adapter,
        extra_input_names=("matrix",),
    )
    precond.apply(residual)
    assert adapter.predictor.calls[-1]["extra"] == {}


def test_apply_after_bind_forwards_extra(neural_with_matrix, residual, small_matrix):
    """Test that apply forwards bound extra inputs to the predictor."""
    neural_with_matrix.bind_inputs(matrix=small_matrix)
    neural_with_matrix.apply(residual)
    call = neural_with_matrix._predictor.calls[-1]  # noqa: SLF001
    assert np.array_equal(call["extra"]["matrix"], small_matrix)


def test_no_extra_input_names_by_default(adapter, dummy_checkpoint):
    """Test that NeuralPreconditioner has empty extra_input_names by default."""
    precond = NeuralPreconditioner(checkpoint_path=dummy_checkpoint, adapter=adapter)
    assert precond.extra_input_names == ()
