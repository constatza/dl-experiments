from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from neuralls.solver import flexible_cg, pcg
from neuralls.solver.monitoring.trace_mode import TraceMode

# Define very strict tolerances for double precision accuracy goals
DOUBLE_PRECISION_ATOL = 1e-14
DOUBLE_PRECISION_RTOL = 1e-15  # Closer to machine epsilon for float64
MAX_SAMPLES = 30


def _tol_bound(b: NDArray, atol: float, rtol: float) -> float:
    return max(atol, rtol * float(np.linalg.norm(b)))


def _plot_history(
    name: str, history: Sequence[float] | np.ndarray, out_dir: Path
) -> None:
    if len(history) == 0:
        return
    values = np.asarray(history, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.semilogy(values, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title(name)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
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


def jacobi_preconditioner_callable(A_matrix: NDArray) -> LinearOperator:
    """Creates a Jacobi preconditioner as a SciPy LinearOperator."""
    diag_inv = 1.0 / np.diag(A_matrix)

    def matvec(vec: np.ndarray) -> np.ndarray:
        return diag_inv * vec

    return LinearOperator(matvec=matvec, shape=A_matrix.shape, dtype=np.float64)


@pytest.fixture(scope="module")
def diagnostics_dir() -> Path:
    """Directory for high-precision solver diagnostics and plots.

    Returns:
        Path to tests/artifacts/solver_plots/ directory.

    Notes:
        Plots are persistent for debugging and visual inspection.
    """
    plots_dir = Path(__file__).parent.parent / "artifacts" / "solver_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def test_pcg_jacobi_double_precision_accuracy(
    spd_system: tuple[NDArray, NDArray, NDArray], diagnostics_dir: Path
) -> None:
    """Tests that preconditioned_cg (SciPy wrapper) with Jacobi preconditioner
    can reach double precision accuracy for the spd_system.
    """
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    jacobi_M = jacobi_preconditioner_callable(A)

    x_sol, info = pcg(
        A,
        b,
        x0,
        preconditioner=jacobi_M,
        atol=DOUBLE_PRECISION_ATOL,
        rtol=DOUBLE_PRECISION_RTOL,
        maxiter=200,  # Sufficiently high maxiter
        trace_mode=TraceMode.FULL,
    )

    assert info.converged
    assert info.residual_abs <= _tol_bound(
        b, DOUBLE_PRECISION_ATOL, DOUBLE_PRECISION_RTOL
    )
    assert np.allclose(
        x_sol, x_true, atol=DOUBLE_PRECISION_ATOL, rtol=DOUBLE_PRECISION_RTOL
    )
    if info.residual_history is not None and info.residual_history_abs is not None:
        count = min(
            len(info.residual_history), len(info.residual_history_abs), MAX_SAMPLES
        )
        data = np.column_stack(
            [
                np.asarray(info.residual_history_abs[:count], dtype=np.float64),
                np.asarray(info.residual_history[:count], dtype=np.float64),
            ]
        )
        np.savetxt(
            diagnostics_dir / "pcg_double_precision_residual_norms.csv",
            data,
            delimiter=",",
            fmt=["%.6e", "%.6e"],
            header="residual_abs,residual_rel",
            comments="",
        )
    if info.residual_history is not None:
        _plot_history(
            "pcg_double_precision_history", info.residual_history, diagnostics_dir
        )
    # Uncomment to persist residual vectors for inspection (requires trace_mode="full" above).
    # if info.residual_vectors is not None:
    #     np.savetxt(
    #         diagnostics_dir / "pcg_double_precision_residual_vectors.csv",
    #         info.residual_vectors[:MAX_SAMPLES],
    #         delimiter=",",
    #         fmt="%.6e",
    #     )


def test_flexible_pcg_jacobi_double_precision_accuracy(
    spd_system: tuple[NDArray, NDArray, NDArray], diagnostics_dir: Path
) -> None:
    """Tests that flexible_cg with Jacobi preconditioner can reach double precision accuracy
    for the spd_system. This is a challenging goal for flexible PCG.
    """
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    jacobi_M = jacobi_preconditioner_callable(A)

    x_sol, info = flexible_cg(
        A,
        b,
        x0,
        preconditioner=jacobi_M,
        atol=DOUBLE_PRECISION_ATOL,
        rtol=DOUBLE_PRECISION_RTOL,
        maxiter=200,  # Sufficiently high maxiter
        m_max=20,  # Use a larger m_max for better orthogonalization
        trace_mode=TraceMode.FULL,
    )

    assert info.converged
    assert info.residual_abs <= _tol_bound(
        b, DOUBLE_PRECISION_ATOL, DOUBLE_PRECISION_RTOL
    )
    assert np.allclose(
        x_sol, x_true, atol=DOUBLE_PRECISION_ATOL, rtol=DOUBLE_PRECISION_RTOL
    )
    if info.residual_history is not None and info.residual_history_abs is not None:
        count = min(
            len(info.residual_history), len(info.residual_history_abs), MAX_SAMPLES
        )
        data = np.column_stack(
            [
                np.asarray(info.residual_history_abs[:count], dtype=np.float64),
                np.asarray(info.residual_history[:count], dtype=np.float64),
            ]
        )
        np.savetxt(
            diagnostics_dir / "flexible_pcg_jacobi_double_precision_residual_norms.csv",
            data,
            delimiter=",",
            fmt=["%.6e", "%.6e"],
            header="residual_abs,residual_rel",
            comments="",
        )
    if info.residual_history is not None:
        _plot_history(
            "flexible_pcg_jacobi_double_precision",
            info.residual_history,
            diagnostics_dir,
        )
    # Uncomment to persist residual vectors for inspection (requires trace_mode="full" above).
    # if info.residual_vectors is not None:
    #     np.savetxt(
    #         diagnostics_dir / "fcg_double_precision_residual_vectors.csv",
    #         info.residual_vectors[:MAX_SAMPLES],
    #         delimiter=",",
    #         fmt="%.6e",
    #     )
