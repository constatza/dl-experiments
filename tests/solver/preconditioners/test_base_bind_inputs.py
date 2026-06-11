"""Tests that Preconditioner base has no extra-input interface, and BindableInputs Protocol works."""

from __future__ import annotations

import numpy as np

from neuralls.domain.solver.preconditioners.base import BindableInputs
from neuralls.domain.solver.preconditioners.implementations.jacobi import JacobiPreconditioner


def test_standard_preconditioner_is_not_bindable(jacobi: JacobiPreconditioner) -> None:
    assert not isinstance(jacobi, BindableInputs)


def test_standard_preconditioner_has_no_extra_input_names(jacobi: JacobiPreconditioner) -> None:
    assert not hasattr(jacobi, "extra_input_names")


def test_standard_preconditioner_has_no_bind_inputs(jacobi: JacobiPreconditioner) -> None:
    assert not hasattr(jacobi, "bind_inputs")


def test_standard_preconditioner_apply_unchanged(jacobi: JacobiPreconditioner) -> None:
    r = np.array([2.0, 4.0, 8.0])
    result = jacobi.apply(r)
    np.testing.assert_allclose(result, np.array([1.0, 1.0, 1.0]))
