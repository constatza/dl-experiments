"""Test bind_inputs propagation in ScheduledPreconditioner."""

import numpy as np
import pytest

from neuralls.domain.solver.preconditioners.base import Preconditioner, PreconditionerContext
from neuralls.domain.solver.preconditioners.implementations.scheduled import ScheduledPreconditioner
from numpy.typing import NDArray


class _TrackingPreconditioner(Preconditioner):
    """Test preconditioner that tracks bind_inputs calls."""

    def __init__(self) -> None:
        """Initialize tracking storage."""
        self.bound: dict[str, NDArray] = {}

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Declare that this preconditioner needs a matrix input."""
        return ("matrix",)

    def bind_inputs(self, **inputs: NDArray) -> None:
        """Store the bound inputs."""
        self.bound = dict(inputs)

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Return a copy of the residual."""
        return residual.copy()


@pytest.fixture
def primary() -> _TrackingPreconditioner:
    """Provide a tracking preconditioner for primary role."""
    return _TrackingPreconditioner()


@pytest.fixture
def fallback() -> _TrackingPreconditioner:
    """Provide a tracking preconditioner for fallback role."""
    return _TrackingPreconditioner()


@pytest.fixture
def matrix() -> NDArray:
    """Provide a test matrix."""
    return np.eye(3) * 5.0


def test_bind_propagates_to_primary(primary: _TrackingPreconditioner, matrix: NDArray) -> None:
    """Test that bind_inputs propagates to primary preconditioner."""
    scheduled = ScheduledPreconditioner(primary=primary, limit_iters=5)
    scheduled.bind_inputs(matrix=matrix)
    assert np.array_equal(primary.bound["matrix"], matrix)


def test_bind_propagates_to_fallback(
    primary: _TrackingPreconditioner, fallback: _TrackingPreconditioner, matrix: NDArray
) -> None:
    """Test that bind_inputs propagates to fallback preconditioner."""
    scheduled = ScheduledPreconditioner(primary=primary, fallback=fallback, limit_iters=5)
    scheduled.bind_inputs(matrix=matrix)
    assert np.array_equal(fallback.bound["matrix"], matrix)


def test_bind_with_none_fallback_no_error(
    primary: _TrackingPreconditioner, matrix: NDArray
) -> None:
    """Test that bind_inputs with None fallback does not raise."""
    scheduled = ScheduledPreconditioner(primary=primary, limit_iters=5)
    scheduled.bind_inputs(matrix=matrix)  # fallback is None — should not raise


def test_bind_with_none_fallback_leaves_fallback_none(
    primary: _TrackingPreconditioner, matrix: NDArray
) -> None:
    """bind_inputs should not mutate _fallback when no fallback was provided."""
    scheduled = ScheduledPreconditioner(primary=primary, limit_iters=5)
    assert scheduled._fallback is None  # noqa: SLF001
    scheduled.bind_inputs(matrix=matrix)
    assert scheduled._fallback is None  # noqa: SLF001
