"""Tests for descriptive preconditioner plot labels.

Follows project conventions: fixtures over inline literals, type hints
throughout, no ``tempfile`` (no file I/O needed here — pure object
introspection).
"""

from __future__ import annotations

import pytest
import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import (
    Identity,
    JacobiPreconditioner,
    ScheduledPreconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    AMGPreconditioner,
    JacobiSmoother,
    VCycle,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.platform.reporting.preconditioner_labels import (
    build_preconditioner_labels,
    describe_preconditioner,
    preconditioner_label,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tridiag_spd_matrix() -> torch.Tensor:
    """Dense 5x5 SPD tridiagonal matrix, suitable for AMG aggregation coarsening."""
    n = 5
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def rank2_snapshot_ensemble() -> torch.Tensor:
    """Snapshot ensemble whose exact numerical rank is 2 (6 samples, 5 dofs).

    Built as ``coeffs @ base`` from two independent random directions, so a
    POD fit with a high energy-capture threshold (e.g. 0.999) resolves to
    exactly 2 retained modes — deterministic given the fixed seed, and
    distinct from the configured threshold value itself, so tests can assert
    the description reports the *actual* fitted width rather than echoing
    the configured float.
    """
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    coeffs = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    return coeffs @ base


@pytest.fixture
def aggregation_amg_preconditioner(tridiag_spd_matrix: torch.Tensor) -> AMGPreconditioner:
    """Constructed AMG preconditioner using smoothed-aggregation coarsening."""
    coarsening = AggregationCoarsening(omega=0.67)
    smoother = JacobiSmoother(omega=0.67)
    cycle = VCycle(smoother=smoother, n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


@pytest.fixture
def pod_amg_preconditioner(
    tridiag_spd_matrix: torch.Tensor, rank2_snapshot_ensemble: torch.Tensor
) -> AMGPreconditioner:
    """Constructed AMG preconditioner using POD-2G coarsening with an energy-threshold rank."""
    coarsening = PODCoarseningStrategy(snapshots=rank2_snapshot_ensemble, rank=0.999)
    smoother = JacobiSmoother(omega=0.67)
    cycle = VCycle(smoother=smoother, n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


# ==============================================================================
# describe_preconditioner — AMG with aggregation coarsening
# ==============================================================================


def test_describe_preconditioner_amg_aggregation_coarsening(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """AMG with smoothed-aggregation coarsening reports levels, cycle, and both omegas."""
    detail = describe_preconditioner(aggregation_amg_preconditioner)

    assert detail == "2-lvl V-cycle, Jacobi ω=0.67, aggregation ω=0.67"


# ==============================================================================
# describe_preconditioner — AMG with POD-2G coarsening
# ==============================================================================


def test_describe_preconditioner_amg_pod_coarsening_uses_fitted_rank_not_configured_threshold(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """POD-2G reports the actual fitted basis width, not the configured energy threshold.

    ``pod_amg_preconditioner`` was configured with ``rank=0.999`` (an energy
    threshold, not a mode count). The snapshot ensemble has exact numerical
    rank 2, so the fitted basis retains exactly 2 modes — the description
    must surface that real width (matched against the live object's own
    ``_basis`` buffer), never the configured ``0.999`` float.
    """
    coarsening = pod_amg_preconditioner._coarsening
    assert isinstance(coarsening, PODCoarseningStrategy)
    fitted_rank = coarsening._basis.shape[1]

    detail = describe_preconditioner(pod_amg_preconditioner)

    assert detail == f"2-lvl V-cycle, Jacobi ω=0.67, POD rank={fitted_rank}"
    assert fitted_rank == 2
    assert "0.999" not in detail


# ==============================================================================
# describe_preconditioner — plain preconditioners (no structural variants)
# ==============================================================================


@pytest.mark.parametrize(
    "precond",
    [
        Identity(),
    ],
)
def test_describe_preconditioner_identity_returns_empty_string(precond: Preconditioner) -> None:
    """Identity has no structural variants, so no detail is fabricated."""
    assert describe_preconditioner(precond) == ""


def test_describe_preconditioner_jacobi_returns_empty_string(
    tridiag_spd_matrix: torch.Tensor,
) -> None:
    """JacobiPreconditioner has no structural variants, so no detail is fabricated."""
    precond = JacobiPreconditioner(tridiag_spd_matrix)

    assert describe_preconditioner(precond) == ""


# ==============================================================================
# describe_preconditioner — ScheduledPreconditioner unwraps to its primary
# ==============================================================================


def test_describe_preconditioner_unwraps_scheduled_preconditioner(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """A scheduled AMG preconditioner still reports the wrapped AMG's structural detail."""
    scheduled = ScheduledPreconditioner(
        primary=aggregation_amg_preconditioner,
        fallback=Identity(),
        limit_iters=10,
        start_iter=0,
    )

    assert describe_preconditioner(scheduled) == describe_preconditioner(
        aggregation_amg_preconditioner
    )


# ==============================================================================
# preconditioner_label / build_preconditioner_labels
# ==============================================================================


def test_preconditioner_label_includes_amg_detail(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Labels combine the config name with the live structural detail."""
    label = preconditioner_label("amg", aggregation_amg_preconditioner)

    assert label == "amg (2-lvl V-cycle, Jacobi ω=0.67, aggregation ω=0.67)"


def test_preconditioner_label_falls_back_to_bare_name_without_detail() -> None:
    """Labels fall back to the bare name when there is no structural detail."""
    label = preconditioner_label("identity", Identity())

    assert label == "identity"


def test_build_preconditioner_labels_maps_each_name(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Label mapping covers every input name, mixing detailed and bare labels."""
    preconditioners: dict[str, Preconditioner] = {
        "amg": aggregation_amg_preconditioner,
        "identity": Identity(),
    }

    labels = build_preconditioner_labels(preconditioners)

    assert labels == {
        "amg": "amg (2-lvl V-cycle, Jacobi ω=0.67, aggregation ω=0.67)",
        "identity": "identity",
    }
