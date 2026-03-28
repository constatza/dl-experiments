"""Tests for preconditioner performance ordering.

Tests that compare multiple preconditioners to ensure they follow
expected performance ordering: ILU ≥ Jacobi ≥ Identity
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def test_preconditioner_ordering(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    identity_preconditioner: Callable[[NDArray, object], NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    ilu_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify preconditioner ordering: ILU ≤ Jacobi ≤ Identity by iteration count.

    Theory:
        Better approximations to A^{-1} lead to better conditioning and
        fewer iterations:
            ILU approximates A^{-1} better than Jacobi
            Jacobi approximates A^{-1} better than Identity

        Expected: iterations_ilu ≤ iterations_jacobi ≤ iterations_identity
    """
    from neuralls.domain.solver import flexible_cg

    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Solve with identity (no preconditioning)
    _, result_identity = flexible_cg(
        A,
        b,
        preconditioner=lambda r: identity_preconditioner(r, None),
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Solve with Jacobi
    _, result_jacobi = flexible_cg(
        A,
        b,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Solve with ILU
    _, result_ilu = flexible_cg(
        A,
        b,
        preconditioner=ilu_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # All should converge
    assert result_identity.converged
    assert result_jacobi.converged
    assert result_ilu.converged

    # Verify ordering: ILU ≤ Jacobi ≤ Identity
    assert result_ilu.iterations <= result_jacobi.iterations
    assert result_jacobi.iterations <= result_identity.iterations
