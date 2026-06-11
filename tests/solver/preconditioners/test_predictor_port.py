"""Tests for PredictorPort and ExtraInputPredictorPort contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuralls.domain.solver.preconditioners.ports import PredictorPort

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_base_port_apply_no_extras(minimal_predictor: PredictorPort, residual: NDArray) -> None:
    """Base PredictorPort.apply accepts only residual."""
    result = minimal_predictor.apply(residual)
    assert np.array_equal(result, residual)


def test_base_port_is_context_manager(minimal_predictor: PredictorPort, residual: NDArray) -> None:
    """Base PredictorPort supports context manager protocol."""
    with minimal_predictor as pred:
        result = pred.apply(residual)
    assert np.array_equal(result, residual)


def test_apply_extra_inputs_forwarded(
    capturing_predictor: PredictorPort, residual: NDArray
) -> None:
    """ExtraInputPredictorPort.apply forwards named extra arrays."""
    extra = np.eye(4, dtype=np.float64)
    capturing_predictor.apply(residual, matrix=extra)  # type: ignore[call-arg]
    assert "matrix" in capturing_predictor.last_extra  # ty: ignore[unresolved-attribute]
    assert np.array_equal(capturing_predictor.last_extra["matrix"], extra)  # ty: ignore[unresolved-attribute]


def test_apply_no_extra_inputs_still_works(
    capturing_predictor: PredictorPort, residual: NDArray
) -> None:
    """ExtraInputPredictorPort.apply works with no extra inputs."""
    result = capturing_predictor.apply(residual)
    assert np.array_equal(result, residual)
    assert capturing_predictor.last_extra == {}  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


def test_extra_input_port_is_subtype_of_base_port(capturing_predictor: PredictorPort) -> None:
    """ExtraInputPredictorPort is a proper subtype of PredictorPort."""
    assert isinstance(capturing_predictor, PredictorPort)
