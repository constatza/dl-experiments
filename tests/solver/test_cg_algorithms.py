from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from neuralls.constants import REORTHOG_STRICT_THRESHOLD
from neuralls.solver import flexible_cg, preconditioned_cg, run_cg_comparison

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


def test_flexible_matches_classical(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    x_classic, info_classic = preconditioned_cg(
        A, b, x0, atol=FUNCTIONAL_ATOL, rtol=FUNCTIONAL_RTOL, max_iter=200
    )
    x_flex, info_flex = flexible_cg(
        A, b, x0, atol=FUNCTIONAL_ATOL, rtol=FUNCTIONAL_RTOL, max_iter=200
    )

    assert info_classic.converged == info_flex.converged
    bound = _tol_bound(b, FUNCTIONAL_ATOL, FUNCTIONAL_RTOL)
    assert info_classic.residual_abs <= bound
    assert info_flex.residual_abs <= bound


def test_pcg_captures_residual_history(
    spd_system: tuple[NDArray, NDArray, NDArray],
) -> None:
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    def jacobi(residual: np.ndarray) -> np.ndarray:
        diag = np.diag(A)
        return residual / diag

    _, info = preconditioned_cg(
        A,
        b,
        x0,
        preconditioner=jacobi,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        max_iter=10,
    )

    assert info.residual_history is not None
    assert len(info.residual_history) > 0
    assert len(info.residual_history) >= info.iterations
    assert not np.isnan(info.residual_history).any()
    assert info.event_log is not None
    # Check scalar residual norms (logged in MINIMAL mode)
    residual_norms = np.asarray(
        info.event_log.get_history("residual_norm"), dtype=np.float64
    )
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

    _, info = preconditioned_cg(
        A,
        b,
        x0,
        preconditioner=jacobi,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        max_iter=15,
        trace_mode="full",
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
        max_iter=200,
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
        max_iterations=20,
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
        max_iterations=20,
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
        preconditioners={"none": lambda r: r, "identity": lambda r: r},
        rtol=FUNCTIONAL_RTOL,
        atol=FUNCTIONAL_ATOL,
        max_iter=200,
    )

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

    _, info = flexible_cg(
        A, b, x0, max_iterations=50, trace_mode="disabled", atol=FUNCTIONAL_ATOL
    )

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
        max_iterations=200,
        trace_mode="full",
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
    )

    residual_vecs = info.residual_vectors
    solution_vecs = info.solution_vectors

    # Verify each captured pair satisfies: r_k = b - A @ x_k
    for k in range(len(residual_vecs)):
        r_k: np.ndarray = residual_vecs[k]
        x_k: np.ndarray = solution_vecs[k]

        # Compute residual from solution
        computed_residual = b - A @ x_k

        # Verify they match
        assert np.all(
            np.isclose(r_k, computed_residual, atol=FUNCTIONAL_ATOL, rtol=0.0)
        ), (
            f"Pair {k}: residual does not satisfy r_k = b - A @ x_k\n"
            f"  Captured r_k = {r_k}\n"
            f"  b - A @ x_k  = {computed_residual}\n"
            f"  Difference   = {r_k - computed_residual}"
        )


def test_partial_reorthog_unlimited_window() -> None:
    """Test that PartialReorthogonalization with window_size=None uses all vectors."""
    from neuralls.solver import PartialReorthogonalization

    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    reorthog = PartialReorthogonalization(window_size=None)

    # Create 5 basis vectors
    basis = [np.random.randn(len(b)) for _ in range(5)]
    v = np.random.randn(len(b))

    # Reorthogonalize
    v_reorthog, info = reorthog.reorthogonalize(v, basis, iteration=5)

    # Should have reorthogonalized against all 5 vectors
    assert info.reorthog_count == 5
    assert info.window_size == 5


def test_partial_reorthog_fail_fast() -> None:
    """Test that PartialReorthogonalization reports breakdown instead of raising."""
    from neuralls.solver import PartialReorthogonalization

    reorthog = PartialReorthogonalization(window_size=10)

    # Test 1: Near-zero norm basis vector
    basis = [np.array([1e-20, 1e-20])]
    v = np.array([1.0, 1.0])

    v_out, info = reorthog.reorthogonalize(v, basis, iteration=1)
    assert np.allclose(v_out, v)
    assert info.breakdown is True
    assert info.reason is not None and "near-zero" in info.reason
    assert info.basis_index == 0


def test_full_reorthog_fail_fast_zero_input() -> None:
    """Test that FullReorthogonalization handles zero input vector gracefully."""
    from neuralls.solver import FullReorthogonalization

    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    reorthog = FullReorthogonalization(A=A)

    basis = [np.array([1.0, 0.0])]
    v = np.array([1e-20, 1e-20])  # Near-zero vector

    # Should return gracefully with breakdown flag instead of raising
    v_out, info = reorthog.reorthogonalize(v, basis, iteration=1)

    assert info.breakdown is True
    assert info.reason is not None
    assert "near-zero" in info.reason.lower()
    assert "norm" in info.reason.lower()


def test_full_reorthog_fail_fast_zero_basis() -> None:
    """Test that FullReorthogonalization handles zero basis vector gracefully."""
    from neuralls.solver import FullReorthogonalization

    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    reorthog = FullReorthogonalization(A=A)

    basis = [np.array([1e-20, 1e-20])]  # Near-zero basis vector
    v = np.array([1.0, 1.0])

    # Should return gracefully with breakdown flag instead of raising
    v_out, info = reorthog.reorthogonalize(v, basis, iteration=1)

    assert info.breakdown is True
    assert info.reason is not None
    reason_lower = info.reason.lower()
    assert "near-zero" in reason_lower
    # Reason should mention either "A-norm" or "norm" or "basis"
    assert any(word in reason_lower for word in ["a-norm", "norm", "basis"])


def test_selective_reorthog_strict_threshold() -> None:
    """Test that SelectiveReorthogonalization with strict threshold is strict."""
    from neuralls.solver import SelectiveReorthogonalization

    # Create slightly non-orthogonal vectors (2% off orthogonality)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.02, 1.0])  # 2% off from orthogonal
    v2 = v2 / np.linalg.norm(v2)  # Normalize

    # With strict threshold, should trigger reorthogonalization
    reorthog_strict = SelectiveReorthogonalization(threshold=REORTHOG_STRICT_THRESHOLD)
    v_reorthog, info = reorthog_strict.reorthogonalize(v2, [v1], iteration=1)

    # Should have triggered reorthogonalization
    assert info.triggered
    assert info.reorthog_count > 0


def test_jacobi_factory_preserves_signs() -> None:
    """Test that Jacobi preconditioner preserves diagonal signs (including negatives)."""
    # Create matrix with negative diagonal element
    A = np.array([[4.0, 1.0, 0.0], [1.0, -3.0, 0.5], [0.0, 0.5, 2.0]], dtype=np.float64)
    diag = np.diag(A)

    # Jacobi preconditioner using builder pattern
    from neuralls.preconditioner.builders import JacobiBuilder
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
    jacobi_factory = builder.build(A, config)

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
        z_factory = jacobi_factory(r)
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
    z = jacobi_factory(r_test)
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
        max_iter=100,
    )

    # With Jacobi preconditioning using builder pattern
    from neuralls.preconditioner.builders import JacobiBuilder
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
    jacobi_precond = builder.build(A, config)
    _, info_jacobi = flexible_cg(
        A,
        b,
        x0,
        preconditioner=jacobi_precond,
        atol=1e-4,  # Relaxed tolerance
        rtol=1e-6,
        max_iter=100,
    )

    # With Jacobi, should converge (diagonal preconditioning for diagonal matrix is perfect)
    assert info_jacobi.converged, "Jacobi should converge for diagonal matrix"

    # Jacobi should achieve better final residual than baseline
    assert info_jacobi.residual_abs <= info_baseline.residual_abs, (
        f"Jacobi should not make convergence worse! "
        f"Baseline residual: {info_baseline.residual_abs:.2e}, "
        f"Jacobi residual: {info_jacobi.residual_abs:.2e}"
    )
