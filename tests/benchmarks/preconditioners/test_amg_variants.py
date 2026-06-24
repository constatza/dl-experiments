"""Benchmark: VCycleAMG vs WCycleAMG convergence on larger Poisson problems.

Theoretical Basis (Briggs, Henson & McCormick 2000, §3.3):
    W-cycle applies γ = 2 coarse-grid corrections per level vs. V-cycle's γ = 1.
    For SPD problems, W-cycle iteration counts are ≤ V-cycle counts at higher
    cost per cycle. Both should converge in O(1) iterations independent of n.

Expected Outcome:
    - Both variants converge on all problem sizes.
    - WCycleAMG iterations ≤ VCycleAMG iterations for the same problem.
    - Iteration counts remain bounded as n grows (mesh-independent convergence).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import diags as sparse_diags

from neuralls.domain.solver.factories import flexible_cg
from neuralls.domain.solver.preconditioners.implementations.amg import VCycleAMG, WCycleAMG

# (n, n_levels) — larger n needs more levels to stay mesh-independent
TEST_CASES = [
    (50, 3),
    (100, 3),
    (200, 4),
    (500, 4),
]

AMG_CLASSES = [
    pytest.param(VCycleAMG, id="VCycleAMG"),
    pytest.param(WCycleAMG, id="WCycleAMG"),
]


def _poisson_1d(n: int) -> np.ndarray:
    """1D Poisson matrix: tridiagonal [-1, 2, -1], size n.

    Args:
        n: System size.

    Returns:
        Dense (n × n) SPD tridiagonal matrix.
    """
    return sparse_diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="csr").toarray()


def _is_slow(n: int) -> bool:
    return n >= 500


@pytest.mark.benchmark
@pytest.mark.parametrize("n,n_levels", TEST_CASES)
@pytest.mark.parametrize("amg_class", AMG_CLASSES)
def test_amg_variant_converges(
    n: int,
    n_levels: int,
    amg_class: type,
    request: pytest.FixtureRequest,
) -> None:
    """Both AMG variants must converge on 1D Poisson of size n.

    Args:
        n: Problem size.
        n_levels: Hierarchy depth.
        amg_class: VCycleAMG or WCycleAMG.
        request: Pytest fixture request.
    """
    if _is_slow(n):
        request.applymarker(pytest.mark.slow)

    A = _poisson_1d(n)
    b = np.ones(n)
    precond = amg_class(A, n_levels=n_levels)

    x, info = flexible_cg(A, b, preconditioner=precond, rtol=1e-8, maxiter=200)

    assert info.converged, (
        f"{amg_class.__name__}(n={n}, levels={n_levels}) did not converge "
        f"in {info.iterations} iterations"
    )
    np.testing.assert_allclose(A @ x, b, rtol=1e-6)


@pytest.mark.benchmark
@pytest.mark.parametrize("n,n_levels", TEST_CASES)
def test_wcycle_iters_not_worse_than_vcycle(
    n: int,
    n_levels: int,
    request: pytest.FixtureRequest,
) -> None:
    """WCycleAMG must need ≤ iterations than VCycleAMG for every problem size.

    Args:
        n: Problem size.
        n_levels: Hierarchy depth.
        request: Pytest fixture request.
    """
    if _is_slow(n):
        request.applymarker(pytest.mark.slow)

    A = _poisson_1d(n)
    b = np.ones(n)

    _, info_v = flexible_cg(
        A, b, preconditioner=VCycleAMG(A, n_levels=n_levels), rtol=1e-8, maxiter=200
    )
    _, info_w = flexible_cg(
        A, b, preconditioner=WCycleAMG(A, n_levels=n_levels), rtol=1e-8, maxiter=200
    )

    assert info_v.converged and info_w.converged, "At least one variant did not converge"
    assert info_w.iterations <= info_v.iterations, (
        f"WCycle ({info_w.iterations}) used more iterations than VCycle ({info_v.iterations}) "
        f"for n={n}"
    )
