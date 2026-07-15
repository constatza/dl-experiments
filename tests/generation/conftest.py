"""Shared fixtures for generation tests."""

from __future__ import annotations

from typing import Any

import pytest

from neuralls.domain.generation.interfaces import TracingSolverCallable


@pytest.fixture
def residual_solver() -> TracingSolverCallable:
    """Default tracing solver for residuals / gaussian_residuals strategies."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@pytest.fixture
def direction_solver() -> TracingSolverCallable:
    """Default tracing solver for the search_directions strategy."""
    from neuralls.composition.generation.default_services import make_solver

    return make_solver()


@pytest.fixture
def solver_overrides(
    residual_solver: TracingSolverCallable, direction_solver: TracingSolverCallable
) -> dict[str, Any]:
    """Solver overrides covering every single-RHS (trace) strategy."""
    return {
        "residuals": residual_solver,
        "gaussian_residuals": residual_solver,
        "search_directions": direction_solver,
    }
