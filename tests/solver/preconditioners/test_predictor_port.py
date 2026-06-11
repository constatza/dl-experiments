import numpy as np
import pytest

from neuralls.domain.solver.preconditioners.ports import PredictorPort


class _CapturingPredictor(PredictorPort):
    def __init__(self):
        self.last_extra: dict = {}

    def apply(self, residual, **extra_inputs):
        self.last_extra = extra_inputs
        return residual.copy()

    def cleanup(self):
        pass


@pytest.fixture
def residual():
    return np.ones(4, dtype=np.float64)


@pytest.fixture
def capturing_predictor():
    return _CapturingPredictor()


def test_apply_extra_inputs_forwarded(capturing_predictor, residual):
    extra = np.eye(4, dtype=np.float64)
    capturing_predictor.apply(residual, matrix=extra)
    assert "matrix" in capturing_predictor.last_extra
    assert np.array_equal(capturing_predictor.last_extra["matrix"], extra)


def test_apply_no_extra_inputs_still_works(capturing_predictor, residual):
    result = capturing_predictor.apply(residual)
    assert np.array_equal(result, residual)
    assert capturing_predictor.last_extra == {}
