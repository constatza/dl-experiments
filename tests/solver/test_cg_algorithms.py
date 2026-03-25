from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from neuralls.solver import flexible_cg, pcg, scipy_cg
from neuralls.solver.preconditioners import Identity
from neuralls.workflows import run_cg_comparison
from neuralls.solver.monitoring.trace_mode import TraceMode

# Functional/Integration Test Tolerances
FUNCTIONAL_ATOL = 1e-6
FUNCTIONAL_RTOL = 1e-8

# Precision Test Tolerances (for scenarios where higher accuracy is expected)
# Note: Achieving true "double precision accuracy" (e.g., 1e-15/1e-16 for RTOL)
# for FCG can be very challenging and might require specific system tuning,
# higher max_iter, or indicate limitations of the current implementation.
PRECISION_ATOL = 1e-8
PRECISION_RTOL = 1e-10


def _tol_bound(b: NDArray, atol: float, rtol: float) -> float:
    return max(atol, rtol * float(np.linalg.norm(b)))


@pytest.fixture(scope="module")
def spd_system() -> tuple[NDArray, NDArray, NDArray]:
    """Deterministic 20x20 SPD system with known solution."""
    n = 20
    diag = np.full(n, 4.0, dtype=np.float64)
    off1 = np.full(n - 1, -0.5, dtype=np.float64)
    off2 = np.full(n - 2, -0.1, dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], off1)
    np.fill_diagonal(A[:, 1:], off1)
    np.fill_diagonal(A[2:], off2)
    np.fill_diagonal(A[:, 2:], off2)
    solution = np.linspace(1.0, 2.0, n, dtype=np.float64)
    b = A @ solution
    return A, b, solution


def test_pcg_captures_residual_history(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    def jacobi(residual: np.ndarray) -> np.ndarray:
        diag = np.diag(A)
        return residual / diag

    _, info = pcg(
        A,
        b,
        x0,
        preconditioner=jacobi,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        maxiter=10,
    )

    assert info.residual_history is not None
    assert len(info.residual_history) > 0
    assert len(info.residual_history) >= info.iterations
    assert not np.isnan(info.residual_history).any()
    assert info.iteration_history is not None
    # Check scalar residual norms (logged in MINIMAL mode)
    residual_norms = np.asarray(info.iteration_history.residual_norms.to_list(), dtype=np.float64)
    assert residual_norms.size > 0
    assert residual_norms.size >= info.iterations
    assert not np.isnan(residual_norms).any()


def test_pcg_captures_vector_traces(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    def jacobi(residual: np.ndarray) -> np.ndarray:
        return residual / np.diag(A)

    _, info = pcg(
        A,
        b,
        x0,
        preconditioner=jacobi,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        maxiter=15,
        trace_mode=TraceMode.FULL,
    )

    assert info.residual_vectors is not None
    assert info.solution_vectors is not None
    # Vectors include iteration 0, so shape[0] = iterations + 1
    assert info.residual_vectors.shape[0] == info.iterations + 1
    assert info.solution_vectors.shape[0] == info.iterations + 1
    assert info.residual_vectors.shape[1] == b.shape[0]
    assert info.solution_vectors.shape[1] == b.shape[0]
    final_solution = info.solution_vectors[-1]
    assert np.allclose(
        info.residual_vectors[-1],
        b - A @ final_solution,
        atol=FUNCTIONAL_ATOL,
        rtol=0.0,
    )


def test_flexible_cg_accepts_linear_operator_preconditioner(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    """Ensure FCG can use a SciPy LinearOperator preconditioner."""
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    # Jacobi as LinearOperator
    diag_inv = 1.0 / np.diag(A)

    def jacobi_mv(vec: np.ndarray) -> np.ndarray:
        return diag_inv * vec

    M = LinearOperator(matvec=jacobi_mv, shape=A.shape, dtype=np.float64)

    _, info = flexible_cg(
        A,
        b,
        x0,
        preconditioner=M,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        maxiter=200,
    )

    assert info.converged
    assert info.residual_abs <= _tol_bound(b, FUNCTIONAL_ATOL, FUNCTIONAL_RTOL)


def test_flexible_cg_with_torch_linear_preconditioner(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    """Integration: FCG using a torch.nn.Linear as neural-like preconditioner."""
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip("torch not available in environment")

    A, b, x_true = spd_system

    # Baseline without preconditioning
    _, base_info = flexible_cg(
        A,
        b,
        maxiter=20,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
    )

    layer = torch.nn.Linear(A.shape[0], A.shape[0], bias=False)
    with torch.no_grad():
        inv_diag = torch.tensor(np.diag(A), dtype=layer.weight.dtype)
        layer.weight.copy_(torch.diag(1.0 / inv_diag))
    layer.eval()

    def torch_precond(residual: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_in = torch.from_numpy(residual.astype(np.float32, copy=False))
            out = layer(tensor_in)
            return out.detach().cpu().numpy().astype(np.float64, copy=False)

    _, neural_info = flexible_cg(
        A,
        b,
        maxiter=20,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        preconditioner=torch_precond,
    )
    assert neural_info.residual_abs <= _tol_bound(b, FUNCTIONAL_ATOL, FUNCTIONAL_RTOL)


def test_run_cg_comparison_preconditioners(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    """Basic sanity check: static preconditioners run via SciPy CG."""
    A, b, _ = spd_system

    results = run_cg_comparison(
        A,
        b,
        preconditioners={"identity": Identity()},
        rtol=FUNCTIONAL_RTOL,
        atol=FUNCTIONAL_ATOL,
        maxiter=200,
    )

    # Note: "none" baseline is automatically added by run_cg_comparison
    assert set(results.keys()) == {"none", "identity"}
    for result in results.values():
        assert result.iterations > 0
        assert not result.breakdown


def test_flexible_pcg_no_traces_by_default(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    """Test that traces are not captured when trace_mode='disabled'."""
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(A, b, x0, maxiter=50, trace_mode=TraceMode.DISABLED, atol=FUNCTIONAL_ATOL)

    assert info.residual_vectors is None
    assert info.solution_vectors is None


def test_flexible_pcg_traces_satisfy_residual_equation(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    """Test that captured traces satisfy: residual = b - A @ solution."""
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(
        A,
        b,
        x0,
        maxiter=200,
        trace_mode=TraceMode.FULL,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
    )

    residual_vecs = info.residual_vectors
    solution_vecs = info.solution_vectors

    # Verify vectors were captured
    assert residual_vecs is not None, "residual_vectors should be captured in FULL mode"
    assert solution_vecs is not None, "solution_vectors should be captured in FULL mode"

    # Verify each captured pair satisfies: r_k = b - A @ x_k
    for k in range(len(residual_vecs)):
        r_k: np.ndarray = residual_vecs[k]
        x_k: np.ndarray = solution_vecs[k]

        # Compute residual from solution
        computed_residual = b - A @ x_k

        # Verify they match
        assert np.all(np.isclose(r_k, computed_residual, atol=FUNCTIONAL_ATOL, rtol=0.0)), (
            f"Pair {k}: residual does not satisfy r_k = b - A @ x_k\n"
            f"  Captured r_k = {r_k}\n"
            f"  b - A @ x_k  = {computed_residual}\n"
            f"  Difference   = {r_k - computed_residual}"
        )


def test_scipy_cg_exposes_vector_traces(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    """SciPy CG should expose traced residual and solution vectors in SolverResult."""
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = scipy_cg(
        A,
        b,
        x0,
        maxiter=200,
        trace_mode=TraceMode.FULL,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
    )

    residual_vecs = info.residual_vectors
    solution_vecs = info.solution_vectors

    assert residual_vecs is not None
    assert solution_vecs is not None
    assert residual_vecs.shape[0] == info.iterations + 1
    assert solution_vecs.shape[0] == info.iterations + 1

    for k in range(len(residual_vecs)):
        np.testing.assert_allclose(
            residual_vecs[k],
            b - A @ solution_vecs[k],
            atol=FUNCTIONAL_ATOL,
            rtol=0.0,
        )


def test_jacobi_factory_preserves_signs() -> None:
    """Test that Jacobi preconditioner preserves diagonal signs (including negatives)."""
    # Create matrix with negative diagonal element
    A = np.array([[4.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 2.0]], dtype=np.float64)
    diag = np.diag(A)

    # Jacobi preconditioner using factory function
    from neuralls.solver.preconditioners.builders import create_preconditioner
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")  # type: ignore[arg-type]
    jacobi_precond = create_preconditioner(A, config)

    # Inline version (correct reference)
    def jacobi_inline(r: np.ndarray) -> np.ndarray:
        return r / diag

    # Test with various residual vectors
    test_residuals = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -2.0, 3.0]),
        np.array([-1.0, -1.0, -1.0]),
    ]

    for r in test_residuals:
        z_factory = jacobi_precond.apply(r)
        z_inline = jacobi_inline(r)

        # Should match exactly (no tolerance needed for this operation)
        np.testing.assert_allclose(
            z_factory,
            z_inline,
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"Factory Jacobi doesn't match inline for residual {r}",
        )

    # Verify sign preservation explicitly for negative diagonal
    r_test = np.array([0.0, 1.0, 0.0])  # Only second component non-zero
    z = jacobi_precond.apply(r_test)
    # diag[1] = -3.0, so z[1] should be negative: 1.0 / -3.0 = -0.333...
    assert z[1] < 0, "Jacobi should preserve negative sign for negative diagonal element"
    assert np.isclose(z[1], 1.0 / -3.0), "Jacobi should compute correct reciprocal"


def test_jacobi_factory_convergence_with_fcg() -> None:
    """Test that Jacobi preconditioner improves convergence vs no preconditioning."""
    # Create a well-conditioned SPD system where Jacobi should help
    n = 20
    A = np.diag(np.arange(1, n + 1, dtype=np.float64))  # Diagonal matrix
    x_true = np.ones(n, dtype=np.float64)
    b = A @ x_true
    x0 = np.zeros_like(b)

    # Baseline: no preconditioning
    _, info_baseline = flexible_cg(
        A,
        b,
        x0,
        atol=1e-4,  # Relaxed tolerance since residual management is disabled
        rtol=1e-6,
        maxiter=100,
    )

    # With Jacobi preconditioning using factory function
    from neuralls.solver.preconditioners.builders import create_preconditioner
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")  # type: ignore[arg-type]
    jacobi_precond = create_preconditioner(A, config)
    _, info_jacobi = flexible_cg(
        A,
        b,
        x0,
        preconditioner=jacobi_precond,
        atol=1e-4,  # Relaxed tolerance
        rtol=1e-6,
        maxiter=100,
    )

    # With Jacobi, should converge (diagonal preconditioning for diagonal matrix is perfect)
    assert info_jacobi.converged, "Jacobi should converge for diagonal matrix"

    # Jacobi should achieve better final residual than baseline
    assert info_jacobi.residual_abs <= info_baseline.residual_abs, (
        f"Jacobi should not make convergence worse! "
        f"Baseline residual: {info_baseline.residual_abs:.2e}, "
        f"Jacobi residual: {info_jacobi.residual_abs:.2e}"
    )
