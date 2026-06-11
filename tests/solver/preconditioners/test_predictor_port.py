"""Tests for PredictorPort and ExtraInputPredictorPort contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuralls.domain.solver.preconditioners.ports import ExtraInputPredictorPort, PredictorPort

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _MinimalPredictor(PredictorPort):
    """Minimal implementation of base PredictorPort — no extra inputs."""

    def apply(self, residual: NDArray) -> NDArray:
        """Return a copy of the residual unchanged.

        Args:
            residual: Input residual vector.

        Returns:
            Copy of the residual.
        """
        return residual.copy()

    def cleanup(self) -> None:
        """No-op cleanup."""
        pass


class _CapturingPredictor(ExtraInputPredictorPort):
    """Records extra inputs for assertion."""

    def __init__(self) -> None:
        """Initialize with empty extra inputs record."""
        self.last_extra: dict[str, NDArray] = {}

    def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        """Record extra inputs and return a copy of the residual.

        Args:
            residual: Input residual vector.
            **extra_inputs: Named extra arrays to capture.

        Returns:
            Copy of the residual.
        """
        self.last_extra = extra_inputs
        return residual.copy()

    def cleanup(self) -> None:
        """No-op cleanup."""
        pass


def test_base_port_apply_no_extras(residual: NDArray) -> None:
    """Base PredictorPort.apply accepts only residual."""
    pred = _MinimalPredictor()
    result = pred.apply(residual)
    assert np.array_equal(result, residual)


def test_base_port_is_context_manager(residual: NDArray) -> None:
    """Base PredictorPort supports context manager protocol."""
    with _MinimalPredictor() as pred:
        result = pred.apply(residual)
    assert np.array_equal(result, residual)


def test_apply_extra_inputs_forwarded(
    capturing_predictor: _CapturingPredictor, residual: NDArray
) -> None:
    """ExtraInputPredictorPort.apply forwards named extra arrays."""
    extra = np.eye(4, dtype=np.float64)
    capturing_predictor.apply(residual, matrix=extra)
    assert "matrix" in capturing_predictor.last_extra
    assert np.array_equal(capturing_predictor.last_extra["matrix"], extra)


def test_apply_no_extra_inputs_still_works(
    capturing_predictor: _CapturingPredictor, residual: NDArray
) -> None:
    """ExtraInputPredictorPort.apply works with no extra inputs."""
    result = capturing_predictor.apply(residual)
    assert np.array_equal(result, residual)
    assert capturing_predictor.last_extra == {}


def test_extra_input_port_is_subtype_of_base_port(capturing_predictor: _CapturingPredictor) -> None:
    """ExtraInputPredictorPort is a proper subtype of PredictorPort."""
    assert isinstance(capturing_predictor, PredictorPort)
