"""Tests for IterationState dataclass.

This module tests the immutable state tracking dataclass that maintains
diagnostic information throughout flexible PCG iterations.

Test Coverage:
- State initialization with default values
- Immutable updates via dataclasses.replace()
- Field validation in __post_init__
- Residual history management
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.solver.state import IterationState


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def initial_state() -> IterationState:
    """Create initial state with default values.

    Returns:
        IterationState: Fresh state at iteration 0.

    Theory:
        Initial state represents algorithm before first iteration:
        - Not converged
        - No residual history (will be populated by initialize_state)
        - No restarts or breakdowns
        - Zero counters
    """
    return IterationState(
        converged=False,
        residual_history=[],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=0,
    )


@pytest.fixture
def state_with_history() -> IterationState:
    """Create state with populated residual history.

    Returns:
        IterationState: State after 3 iterations with decreasing residuals.

    Theory:
        Residual history tracks ||r_k||_2 at each iteration.
        In well-behaved CG, residuals decrease monotonically.
    """
    return IterationState(
        converged=False,
        residual_history=[1.0, 0.5, 0.25, 0.1],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=3,
    )


@pytest.fixture
def converged_state() -> IterationState:
    """Create converged state.

    Returns:
        IterationState: State after convergence.

    Theory:
        Converged state must have:
        - converged=True
        - Non-empty residual history
        - Final residual below tolerance
    """
    return IterationState(
        converged=True,
        residual_history=[1.0, 0.1, 1e-7],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=2,
    )


@pytest.fixture
def restart_state() -> IterationState:
    """Create state with restart triggered.

    Returns:
        IterationState: State after restart event.

    Theory:
        Restart indicates:
        - Negative/small curvature detected
        - Beta out of bounds
        - Recovery via steepest descent
    """
    return IterationState(
        converged=False,
        residual_history=[1.0, 0.8, 0.9],
        restart=True,
        breakdown=False,
        divergence=False,
        num_restarts=1,
        num_residual_replacements=0,
        iterations=2,
    )


@pytest.fixture
def breakdown_state() -> IterationState:
    """Create state with breakdown detected.

    Returns:
        IterationState: State after numerical breakdown.

    Theory:
        Breakdown is terminal:
        - NaN or Inf detected
        - Algorithm must terminate
        - Cannot recover
    """
    return IterationState(
        converged=False,
        residual_history=[1.0, 0.5],
        restart=False,
        breakdown=True,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=1,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


def test_state_default_initialization(initial_state: IterationState) -> None:
    """Test state initializes with correct default values.

    Args:
        initial_state: Fresh initial state fixture.

    Theory:
        Default state represents algorithm at iteration 0:
        - All flags False (not converged, no errors)
        - Empty residual history (populated during iteration)
        - Zero counters
    """
    assert not initial_state.converged
    assert initial_state.residual_history == []
    assert not initial_state.restart
    assert not initial_state.breakdown
    assert not initial_state.divergence
    assert initial_state.num_restarts == 0
    assert initial_state.num_residual_replacements == 0
    assert initial_state.iterations == 0


def test_state_with_residual_history(state_with_history: IterationState) -> None:
    """Test state correctly stores residual history.

    Args:
        state_with_history: State with 4 residual values.

    Theory:
        Residual history enables:
        - Convergence rate analysis
        - Divergence detection
        - Post-hoc diagnostics
    """
    assert len(state_with_history.residual_history) == 4
    assert state_with_history.residual_history == [1.0, 0.5, 0.25, 0.1]
    assert state_with_history.iterations == 3


def test_converged_state_properties(converged_state: IterationState) -> None:
    """Test converged state has required properties.

    Args:
        converged_state: State after successful convergence.

    Theory:
        Converged state must have:
        - converged=True
        - Non-empty residual history
        - Residual below tolerance (checked elsewhere)
    """
    assert converged_state.converged
    assert len(converged_state.residual_history) > 0
    assert converged_state.iterations == 2


def test_restart_state_properties(restart_state: IterationState) -> None:
    """Test restart state has correct flags and counters.

    Args:
        restart_state: State with restart triggered.

    Theory:
        Restart state indicates:
        - restart flag set (for current iteration)
        - num_restarts counter incremented
        - Algorithm continues (not terminal)
    """
    assert restart_state.restart
    assert restart_state.num_restarts == 1
    assert not restart_state.breakdown
    assert not restart_state.converged


def test_breakdown_state_properties(breakdown_state: IterationState) -> None:
    """Test breakdown state has correct flags.

    Args:
        breakdown_state: State with breakdown detected.

    Theory:
        Breakdown is terminal:
        - breakdown=True
        - Algorithm should terminate immediately
        - Cannot recover from NaN/Inf
    """
    assert breakdown_state.breakdown
    assert not breakdown_state.converged
    assert not breakdown_state.restart


# =============================================================================
# Immutable Update Tests
# =============================================================================


def test_state_immutable_update_converged(initial_state: IterationState) -> None:
    """Test immutable update via dataclasses.replace() for convergence.

    Args:
        initial_state: Fresh initial state.

    Theory:
        Immutable updates preserve original state:
        - Original state unchanged
        - New state has updated fields
        - Functional programming pattern
    """
    # Need non-empty residual history to set converged=True (validation requirement)
    state_with_residual = replace(initial_state, residual_history=[1.0])
    updated = replace(state_with_residual, converged=True)

    # Original unchanged
    assert not initial_state.converged

    # Updated has new value
    assert updated.converged

    # Other fields preserved
    assert len(updated.residual_history) == 1
    assert updated.iterations == initial_state.iterations


def test_state_immutable_update_residual_history(state_with_history: IterationState) -> None:
    """Test immutable update with new residual added to history.

    Args:
        state_with_history: State with existing residual history.

    Theory:
        Residual history grows by appending new value:
        - Original list unchanged (immutability)
        - New state has extended list
        - List concatenation creates new list
    """
    new_residual = 0.05
    updated = replace(
        state_with_history,
        residual_history=state_with_history.residual_history + [new_residual],
        iterations=state_with_history.iterations + 1,
    )

    # Original unchanged
    assert len(state_with_history.residual_history) == 4
    assert state_with_history.iterations == 3

    # Updated has new values
    assert len(updated.residual_history) == 5
    assert updated.residual_history[-1] == new_residual
    assert updated.iterations == 4


def test_state_immutable_update_restart(initial_state: IterationState) -> None:
    """Test immutable update for restart event.

    Args:
        initial_state: Fresh initial state.

    Theory:
        Restart update changes:
        - restart flag set
        - num_restarts incremented
        - Other fields unchanged
    """
    updated = replace(
        initial_state,
        restart=True,
        num_restarts=1,
    )

    # Original unchanged
    assert not initial_state.restart
    assert initial_state.num_restarts == 0

    # Updated has restart
    assert updated.restart
    assert updated.num_restarts == 1


def test_state_immutable_update_breakdown(state_with_history: IterationState) -> None:
    """Test immutable update for breakdown event.

    Args:
        state_with_history: State with some history.

    Theory:
        Breakdown is terminal:
        - breakdown flag set
        - History preserved
        - No further iterations expected
    """
    updated = replace(state_with_history, breakdown=True)

    # Original unchanged
    assert not state_with_history.breakdown

    # Updated has breakdown
    assert updated.breakdown

    # History preserved
    assert updated.residual_history == state_with_history.residual_history


def test_state_immutable_update_divergence(state_with_history: IterationState) -> None:
    """Test immutable update for divergence detection.

    Args:
        state_with_history: State with some history.

    Theory:
        Divergence triggers:
        - divergence flag set
        - num_residual_replacements incremented (recovery)
        - Algorithm attempts recovery
    """
    updated = replace(
        state_with_history,
        divergence=True,
        num_residual_replacements=1,
    )

    # Original unchanged
    assert not state_with_history.divergence
    assert state_with_history.num_residual_replacements == 0

    # Updated has divergence
    assert updated.divergence
    assert updated.num_residual_replacements == 1


# =============================================================================
# Field Validation Tests
# =============================================================================


def test_state_validation_converged_without_history() -> None:
    """Test validation fails for converged state without residual history.

    Theory:
        Converged state must have non-empty residual history:
        - Need at least one residual value
        - Residual value must be below tolerance
        - Empty history indicates invalid state
    """
    with pytest.raises(ValueError, match="Cannot be converged with empty residual history"):
        IterationState(
            converged=True,
            residual_history=[],  # Invalid: converged but no history
            restart=False,
            breakdown=False,
            divergence=False,
            num_restarts=0,
            num_residual_replacements=0,
            iterations=0,
        )


def test_state_validation_allows_breakdown_at_any_iteration() -> None:
    """Test validation allows breakdown at any iteration.

    Theory:
        Breakdown can occur at any iteration:
        - Initial guess issues (iteration 0)
        - Preconditioner failure (any iteration)
        - Numerical overflow (any iteration)
    """
    # Breakdown at iteration 0 (allowed)
    state_iter0 = IterationState(
        converged=False,
        residual_history=[1.0],
        restart=False,
        breakdown=True,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=0,
    )
    assert state_iter0.breakdown

    # Breakdown at iteration 5 (allowed)
    state_iter5 = IterationState(
        converged=False,
        residual_history=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        restart=False,
        breakdown=True,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=5,
    )
    assert state_iter5.breakdown


# =============================================================================
# Multiple Update Tests
# =============================================================================


def test_state_multiple_immutable_updates(initial_state: IterationState) -> None:
    """Test chaining multiple immutable updates.

    Args:
        initial_state: Fresh initial state.

    Theory:
        Functional programming pattern:
        - Each update creates new state
        - Original state never modified
        - Chain of transformations
    """
    # Simulate 3 iterations
    state1 = replace(
        initial_state,
        residual_history=[1.0],
        iterations=0,
    )
    state2 = replace(
        state1,
        residual_history=state1.residual_history + [0.5],
        iterations=1,
    )
    state3 = replace(
        state2,
        residual_history=state2.residual_history + [0.1],
        iterations=2,
        converged=True,
    )

    # Original unchanged
    assert initial_state.residual_history == []
    assert initial_state.iterations == 0

    # Each state independent
    assert len(state1.residual_history) == 1
    assert state1.iterations == 0
    assert not state1.converged

    assert len(state2.residual_history) == 2
    assert state2.iterations == 1
    assert not state2.converged

    assert len(state3.residual_history) == 3
    assert state3.iterations == 2
    assert state3.converged


def test_state_restart_then_converge(initial_state: IterationState) -> None:
    """Test sequence: restart followed by convergence.

    Args:
        initial_state: Fresh initial state.

    Theory:
        Realistic scenario:
        1. Algorithm encounters negative curvature (restart)
        2. Steepest descent recovers
        3. Algorithm converges
    """
    # Restart at iteration 1
    state1 = replace(
        initial_state,
        residual_history=[1.0, 0.9],
        restart=True,
        num_restarts=1,
        iterations=1,
    )

    # Clear restart flag, continue
    state2 = replace(
        state1,
        residual_history=state1.residual_history + [0.5],
        restart=False,
        iterations=2,
    )

    # Converge
    state3 = replace(
        state2,
        residual_history=state2.residual_history + [1e-7],
        converged=True,
        iterations=3,
    )

    # Check progression
    assert state1.restart
    assert state1.num_restarts == 1

    assert not state2.restart
    assert state2.num_restarts == 1  # Counter preserved

    assert state3.converged
    assert state3.num_restarts == 1  # Counter preserved
    assert len(state3.residual_history) == 4
