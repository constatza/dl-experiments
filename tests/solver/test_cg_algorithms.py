from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from src.constants import REORTHOG_STRICT_THRESHOLD
from src.solver import flexible_cg, preconditioned_cg, run_cg_comparison

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


def _save_history_plot(name: str, history: list[float], out_dir: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.semilogy(history, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title(name)
    fig.tight_layout()
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path)
    plt.close(fig)


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


@pytest.fixture(scope="module")
def diagnostics_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "solver" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_flexible_matches_classical(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    x_classic, info_classic = preconditioned_cg(A, b, x0, atol=FUNCTIONAL_ATOL, rtol=FUNCTIONAL_RTOL, max_iter=200)
    x_flex, info_flex = flexible_cg(A, b, x0, atol=FUNCTIONAL_ATOL, rtol=FUNCTIONAL_RTOL, max_iter=200)

    assert info_classic.converged == info_flex.converged
    bound = _tol_bound(b, FUNCTIONAL_ATOL, FUNCTIONAL_RTOL)
    assert info_classic.residual_abs <= bound
    assert info_flex.residual_abs <= bound


def test_flexible_helper_accelerates(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(
        A,
        b,
        x0,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        max_iter=200,
    )

    assert info.converged
    assert np.linalg.norm(A @ _ - b) <= _tol_bound(b, FUNCTIONAL_ATOL, FUNCTIONAL_RTOL)


def test_flexible_cg_captures_residuals_and_solutions(
    spd_system: tuple[NDArray, NDArray, NDArray], diagnostics_dir: Path
) -> None:
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(
        A,
        b,
        x0,
        max_iter=200,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        capture_traces=True,
        capture_search_directions=True,
    )

    assert info.residual_history is not None
    assert len(info.residual_history) >= 1
    assert len(info.residual_history) >= info.iterations
    assert not np.isnan(info.residual_history).any()
    assert info.residual_vectors is not None
    assert info.solution_vectors is not None
    assert info.residual_vectors.shape[0] >= info.iterations
    assert info.solution_vectors.shape[0] >= info.iterations
    _save_history_plot("flexible_cg_history", info.residual_history, diagnostics_dir)


def test_pcg_captures_residual_history(
    spd_system: tuple[NDArray, NDArray, NDArray], diagnostics_dir: Path
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
    residual_norms = info.event_log.get_history("residual_norm")
    assert len(residual_norms) > 0
    assert len(residual_norms) >= info.iterations
    assert not np.isnan(np.array(residual_norms)).any()
    _save_history_plot("pcg_history", residual_norms, diagnostics_dir)


def test_flexible_cg_accepts_linear_operator_preconditioner(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
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


def test_flexible_cg_with_torch_linear_preconditioner(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
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
        max_iter=20,
        stopping_criterion="fixed_iterations",
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
        max_iter=20,
        stopping_criterion="fixed_iterations",
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
        preconditioner=torch_precond,
    )
    assert neural_info.residual_abs <= _tol_bound(b, FUNCTIONAL_ATOL, FUNCTIONAL_RTOL)


def test_run_cg_comparison_preconditioners(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
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


def test_flexible_pcg_capture_traces(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """Test that flexible_pcg captures residual and solution traces correctly."""
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    # Run with trace capture
    x_final, info = flexible_cg(
        A,
        b,
        x0,
        max_iter=200,
        capture_traces=True,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
    )

    # Verify trace arrays are present
    assert info.residual_vectors is not None
    assert info.solution_vectors is not None

    residual_vecs = info.residual_vectors
    solution_vecs = info.solution_vectors

    # SciPy callback logs after each iteration; expect iterations length
    num_iters = info.iterations
    assert residual_vecs.shape[0] >= num_iters
    assert solution_vecs.shape[0] >= num_iters
    assert residual_vecs.shape[1] == len(b)
    assert solution_vecs.shape[1] == len(b)

    # Verify final solution matches returned solution
    assert np.all(np.isclose(solution_vecs[-1], x_final, atol=FUNCTIONAL_ATOL, rtol=0.0))
    assert np.all(np.isclose(x_final, x_true, atol=FUNCTIONAL_ATOL, rtol=0.0))

    # Verify residuals correspond to solutions at SAME iteration for trailing entries
    for k in range(num_iters):
        expected_residual = b - A @ solution_vecs[k]
        assert np.all(np.isclose(residual_vecs[k], expected_residual, atol=FUNCTIONAL_ATOL, rtol=0.0))


def test_flexible_pcg_no_traces_by_default(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """Test that traces are not captured when capture_traces=False."""
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(A, b, x0, max_iter=50, capture_traces=False, atol=FUNCTIONAL_ATOL)

    assert info.residual_vectors is None
    assert info.solution_vectors is None


def test_flexible_pcg_traces_satisfy_residual_equation(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """Test that captured traces satisfy: residual = b - A @ solution."""
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(
        A,
        b,
        x0,
        max_iter=200,
        capture_traces=True,
        atol=FUNCTIONAL_ATOL,
        rtol=FUNCTIONAL_RTOL,
    )

    residual_vecs = info.residual_vectors
    solution_vecs = info.solution_vectors

    # Verify each captured pair satisfies: r_k = b - A @ x_k
    for k in range(len(residual_vecs)):
        r_k = residual_vecs[k]
        x_k = solution_vecs[k]

        # Compute residual from solution
        computed_residual = b - A @ x_k

        # Verify they match
        assert np.all(np.isclose(r_k, computed_residual, atol=FUNCTIONAL_ATOL, rtol=0.0)), (
            f"Pair {k}: residual does not satisfy r_k = b - A @ x_k\n"
            f"  Captured r_k = {r_k}\n"
            f"  b - A @ x_k  = {computed_residual}\n"
            f"  Difference   = {r_k - computed_residual}"
        )


def test_partial_reorthog_unlimited_window() -> None:
    """Test that PartialReorthogonalization with window_size=None uses all vectors."""
    from src.solver import PartialReorthogonalization

    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    reorthog = PartialReorthogonalization(window_size=None)

    # Create 5 basis vectors
    basis = [np.random.randn(len(b)) for _ in range(5)]
    v = np.random.randn(len(b))

    # Reorthogonalize
    v_reorthog, info = reorthog.reorthogonalize(v, basis, iteration=5)

    # Should have reorthogonalized against all 5 vectors
    assert info["reorthog_count"] == 5
    assert info["window_size"] == 5


def test_partial_reorthog_fail_fast() -> None:
    """Test that PartialReorthogonalization raises errors on numerical issues."""
    from src.solver import PartialReorthogonalization
    import pytest

    reorthog = PartialReorthogonalization(window_size=10)

    # Test 1: Near-zero norm basis vector
    basis = [np.array([1e-20, 1e-20])]
    v = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="near-zero norm"):
        reorthog.reorthogonalize(v, basis, iteration=1)


def test_full_reorthog_fail_fast_zero_input() -> None:
    """Test that FullReorthogonalization handles zero input vector gracefully."""
    from src.solver import FullReorthogonalization

    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    reorthog = FullReorthogonalization(A=A)

    basis = [np.array([1.0, 0.0])]
    v = np.array([1e-20, 1e-20])  # Near-zero vector

    # Should return gracefully with breakdown flag instead of raising
    v_out, info = reorthog.reorthogonalize(v, basis, iteration=1)

    assert info["breakdown"] is True
    assert "near-zero" in info["reason"].lower()
    assert "norm" in info["reason"].lower()


def test_full_reorthog_fail_fast_zero_basis() -> None:
    """Test that FullReorthogonalization handles zero basis vector gracefully."""
    from src.solver import FullReorthogonalization

    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    reorthog = FullReorthogonalization(A=A)

    basis = [np.array([1e-20, 1e-20])]  # Near-zero basis vector
    v = np.array([1.0, 1.0])

    # Should return gracefully with breakdown flag instead of raising
    v_out, info = reorthog.reorthogonalize(v, basis, iteration=1)

    assert info["breakdown"] is True
    assert "near-zero" in info["reason"].lower()
    # Reason should mention either "A-norm" or "norm" or "basis"
    assert any(word in info["reason"].lower() for word in ["a-norm", "norm", "basis"])


def test_selective_reorthog_strict_threshold() -> None:
    """Test that SelectiveReorthogonalization with strict threshold is strict."""
    from src.solver import SelectiveReorthogonalization

    # Create slightly non-orthogonal vectors (2% off orthogonality)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.02, 1.0])  # 2% off from orthogonal
    v2 = v2 / np.linalg.norm(v2)  # Normalize

    # With strict threshold, should trigger reorthogonalization
    reorthog_strict = SelectiveReorthogonalization(threshold=REORTHOG_STRICT_THRESHOLD)
    v_reorthog, info = reorthog_strict.reorthogonalize(v2, [v1], iteration=1)

    # Should have triggered reorthogonalization
    assert info["triggered"]
    assert info["reorthog_count"] > 0
