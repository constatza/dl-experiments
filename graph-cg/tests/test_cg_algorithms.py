from __future__ import annotations

import numpy as np

from src.cg_algorithms import flexible_pcg, preconditioned_cg, run_cg_comparison


def _spd_matrix() -> tuple[np.ndarray, np.ndarray]:
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    return A, b


def test_flexible_matches_classical() -> None:
    A, b = _spd_matrix()
    x0 = np.zeros_like(b)

    x_classic, info_classic = preconditioned_cg(A, b, x0, tol=1e-10, max_iter=20)
    x_flex, info_flex = flexible_pcg(A, b, x0, tol=1e-10, max_iter=20)

    assert np.allclose(x_classic, x_flex)
    assert info_classic["converged"] == info_flex["converged"]
    assert np.isclose(info_classic["residual"], info_flex["residual"])


def test_flexible_helper_accelerates() -> None:
    A, b = _spd_matrix()
    x0 = np.zeros_like(b)

    def exact_helper(context) -> np.ndarray:
        # Map residual to exact correction A^{-1} r
        matrix = getattr(context, "matrix", A)
        residual = getattr(context, "residual", context)
        return np.linalg.solve(matrix, residual)

    _, info = flexible_pcg(
        A,
        b,
        x0,
        tol=1e-12,
        max_iter=5,
        helper=exact_helper,
    )

    assert info["converged"]
    assert info["iterations"] <= 2
    assert info["helper_iterations"], "helper should be invoked at initial iteration"
    assert info["helper_iterations"][0] == 0


def test_run_cg_comparison_includes_helpers() -> None:
    A, b = _spd_matrix()

    def helper(context) -> np.ndarray:
        return np.zeros_like(getattr(context, "residual", context))

    results = run_cg_comparison(
        A,
        b,
        preconditioners={"none": lambda r: r},
        warm_starts={"none": lambda _: None},
        step_helpers={"none": lambda _: None, "zero_helper": helper},
        tol=1e-10,
        max_iter=10,
    )

    helper_labels = {info.get("step_helper") for info in results.values()}
    assert "zero_helper" in helper_labels
