"""Tests for _bind_system_inputs helper in comparison single_run workflow."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.composition.comparison._preconditioner_setup import _bind_system_inputs
from neuralls.domain.solver.preconditioners.base import Preconditioner, PreconditionerContext
from numpy.typing import NDArray


class _TrackingPreconditioner(Preconditioner):
    """Preconditioner that records which extra inputs were bound to it."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self.bound: dict[str, NDArray] = {}

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        return self._names

    def bind_inputs(self, **inputs: NDArray) -> None:
        self.bound = dict(inputs)

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        return residual.copy()


@pytest.fixture
def matrix() -> NDArray:
    """5x5 identity matrix."""
    return np.eye(5, dtype=np.float64)


@pytest.fixture
def coords() -> NDArray:
    """5x2 coordinate array."""
    return np.arange(10.0).reshape(5, 2)


def test_bind_matrix_to_neural_only(matrix: NDArray) -> None:
    """Only the preconditioner that declares 'matrix' receives the binding."""
    neural = _TrackingPreconditioner(("matrix",))
    jacobi = _TrackingPreconditioner(())
    _bind_system_inputs({"neural": neural, "jacobi": jacobi}, {"matrix": matrix})
    assert np.array_equal(neural.bound["matrix"], matrix)
    assert jacobi.bound == {}


def test_bind_multiple_extras(matrix: NDArray, coords: NDArray) -> None:
    """A preconditioner declaring multiple names gets all available extras bound."""
    precond = _TrackingPreconditioner(("matrix", "coordinates"))
    _bind_system_inputs({"p": precond}, {"matrix": matrix, "coordinates": coords})
    assert np.array_equal(precond.bound["matrix"], matrix)
    assert np.array_equal(precond.bound["coordinates"], coords)


def test_missing_extra_not_bound(matrix: NDArray) -> None:
    """Names declared but absent from system_data are silently omitted."""
    precond = _TrackingPreconditioner(("matrix", "coordinates"))
    _bind_system_inputs({"p": precond}, {"matrix": matrix})
    assert "matrix" in precond.bound
    assert "coordinates" not in precond.bound


def test_no_extras_declared_nothing_bound(matrix: NDArray) -> None:
    """Preconditioners that declare no extra names are untouched."""
    precond = _TrackingPreconditioner(())
    _bind_system_inputs({"p": precond}, {"matrix": matrix})
    assert precond.bound == {}


def test_empty_preconditioners_dict_no_error() -> None:
    """An empty preconditioner map is handled without error."""
    _bind_system_inputs({}, {"matrix": np.eye(3)})
