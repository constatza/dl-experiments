"""Tests for FCG based on Notay 2000 Section 5.1 artificial experiments.

Reference:
    Notay, Y. (2000). Flexible Conjugate Gradients.
    SIAM Journal on Scientific Computing, 22(4), 1444-1460.
    Section 5.1: Some artificial experiments

All tests use paper's exact parameters:
- n = 10^4
- δ = 10^-6 (relative tolerance)
- FCG(1) for Cases 1 & 2
- FCG(∞) for Case 3

Convergence Criterion:
    Despite the paper discussing "relative error in A-norm" for theoretical analysis,
    the actual experiments use the standard L2 norm for convergence checking:
        ||r_k||_2 / ||b||_2 ≤ δ

    This is confirmed by near-perfect agreement when using the default euclidean norm:
    - Case 2, ε=0: Expected 49, got 49 (exact match!)
    - Case 3, ε=0: Expected 31, got 33 (±2 difference)

    The theoretical A-norm analysis in the paper is for predicting iteration counts,
    not for the practical stopping criterion implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuralls.domain.solver import flexible_cg
from neuralls.domain.solver.monitoring.trace_mode import TraceMode

from .conftest import (
    CASE1_SPEC,
    CASE1_VARIANT_A_RESULTS,
    CASE2_SPEC,
    CASE2_VARIANT_A_RESULTS,
    CASE3_SPEC,
    CASE3_VARIANT_A_RESULTS,
    ITERATION_BASE_ATOL,
    ITERATION_EPSILON_SCALE,
    NOTAY_N,
    NOTAY_TOLERANCE,
    NotayResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def compute_tolerance(epsilon: float, expected_iterations: int) -> float:
    """Compute iteration comparison tolerance.

    Tolerance scales with epsilon since RNG effect is proportional to perturbation.

    Args:
        epsilon: Perturbation magnitude.
        expected_iterations: Expected iteration count from paper.

    Returns:
        Absolute tolerance for iteration comparison.
    """
    return ITERATION_BASE_ATOL + ITERATION_EPSILON_SCALE * epsilon * expected_iterations


@pytest.mark.benchmark
@pytest.mark.slow
class TestCase1VariantA:
    """Case 1 + Variant (a): Uniform eigenvalues (κ=6) + random perturbation.

    Paper Table 1 results (n=10^4, δ=10^-6, FCG(1)):
        ε:    0   1e-2  1e-1  1/7   1/4   1/3   1/2
        iter: 15  15    16    17    19    22    28
    """

    @pytest.mark.parametrize(
        "expected",
        CASE1_VARIANT_A_RESULTS,
        ids=lambda r: f"eps_{r.epsilon:.4f}",
    )
    def test_iterations_match_paper(
        self,
        case1_system: tuple[NDArray, NDArray, NDArray],
        variant_a_precond_factory: Callable[[float], Callable[[NDArray], NDArray]],
        expected: NotayResult,
    ) -> None:
        """Verify FCG iteration counts match paper's Table 1."""
        diag_A, b, x0 = case1_system
        A = np.diag(diag_A)
        m_max = CASE1_SPEC.m_max if CASE1_SPEC.m_max is not None else NOTAY_N

        precond = variant_a_precond_factory(expected.epsilon)

        _, result = flexible_cg(
            A,
            b,
            x0=x0,
            preconditioner=precond,
            rtol=NOTAY_TOLERANCE,
            atol=0.0,
            m_max=m_max,
            maxiter=500,
            trace_mode=TraceMode.MINIMAL,
        )

        atol = compute_tolerance(expected.epsilon, expected.iterations)
        assert result.converged, f"FCG({CASE1_SPEC.m_max}) did not converge at ε={expected.epsilon}"
        assert np.isclose(result.iterations, expected.iterations, rtol=0, atol=atol), (
            f"ε={expected.epsilon}: got {result.iterations}, "
            f"expected {expected.iterations} (±{atol:.1f})"
        )


@pytest.mark.benchmark
@pytest.mark.slow
class TestCase1VariantADynamic:
    """Case 1 + Variant (a-dynamic): Fresh random perturbation each iteration.

    This variant generates a new random perturbation at each iteration,
    unlike the paper's fixed-f approach. Results will differ from Table 1.
    """

    @pytest.mark.parametrize(
        "expected",
        CASE1_VARIANT_A_RESULTS,
        ids=lambda r: f"eps_{r.epsilon:.4f}",
    )
    def test_dynamic_perturbation(
        self,
        case1_system: tuple[NDArray, NDArray, NDArray],
        variant_a_dynamic_precond_factory: Callable[[float], Callable[[NDArray], NDArray]],
        expected: NotayResult,
    ) -> None:
        """Test FCG with dynamic random perturbations."""
        diag_A, b, x0 = case1_system
        A = np.diag(diag_A)
        m_max = CASE1_SPEC.m_max if CASE1_SPEC.m_max is not None else NOTAY_N

        precond = variant_a_dynamic_precond_factory(expected.epsilon)

        _, result = flexible_cg(
            A,
            b,
            x0=x0,
            preconditioner=precond,
            rtol=NOTAY_TOLERANCE,
            atol=0.0,
            m_max=m_max,
            maxiter=500,
            trace_mode=TraceMode.MINIMAL,
        )

        # Just verify convergence, don't compare to paper results
        assert result.converged, (
            f"FCG({CASE1_SPEC.m_max}) did not converge at ε={expected.epsilon} (dynamic variant)"
        )

        # Log results for comparison
        print(f"\nDynamic variant: ε={expected.epsilon:.4f}, iterations={result.iterations}")


@pytest.mark.benchmark
@pytest.mark.slow
class TestCase2VariantA:
    """Case 2 + Variant (a): Uniform eigenvalues (κ=51) + random perturbation.

    Paper Table 1 results (n=10^4, δ=10^-6, FCG(1)):
        ε:    0   1e-2  1e-1  1/7   1/4   1/3   1/2
        iter: 49  49    55    59    69    81    116
    """

    @pytest.mark.parametrize(
        "expected",
        CASE2_VARIANT_A_RESULTS,
        ids=lambda r: f"eps_{r.epsilon:.4f}",
    )
    def test_iterations_match_paper(
        self,
        case2_system: tuple[NDArray, NDArray, NDArray],
        variant_a_precond_factory: Callable[[float], Callable[[NDArray], NDArray]],
        expected: NotayResult,
    ) -> None:
        """Verify FCG iteration counts match paper's Table 1."""
        diag_A, b, x0 = case2_system
        A = np.diag(diag_A)
        m_max = CASE2_SPEC.m_max if CASE2_SPEC.m_max is not None else NOTAY_N

        precond = variant_a_precond_factory(expected.epsilon)

        _, result = flexible_cg(
            A,
            b,
            x0=x0,
            preconditioner=precond,
            rtol=NOTAY_TOLERANCE,
            atol=0.0,
            m_max=m_max,
            maxiter=500,
            trace_mode=TraceMode.MINIMAL,
        )

        atol = compute_tolerance(expected.epsilon, expected.iterations)
        assert result.converged, f"FCG({CASE2_SPEC.m_max}) did not converge at ε={expected.epsilon}"
        assert np.isclose(result.iterations, expected.iterations, rtol=0, atol=atol), (
            f"ε={expected.epsilon}: got {result.iterations}, "
            f"expected {expected.iterations} (±{atol:.1f})"
        )


@pytest.mark.benchmark
@pytest.mark.slow
class TestCase2VariantADynamic:
    """Case 2 + Variant (a-dynamic): Fresh random perturbation each iteration.

    This variant generates a new random perturbation at each iteration,
    unlike the paper's fixed-f approach. Results will differ from Table 1.
    """

    @pytest.mark.parametrize(
        "expected",
        CASE2_VARIANT_A_RESULTS,
        ids=lambda r: f"eps_{r.epsilon:.4f}",
    )
    def test_dynamic_perturbation(
        self,
        case2_system: tuple[NDArray, NDArray, NDArray],
        variant_a_dynamic_precond_factory: Callable[[float], Callable[[NDArray], NDArray]],
        expected: NotayResult,
    ) -> None:
        """Test FCG with dynamic random perturbations."""
        diag_A, b, x0 = case2_system
        A = np.diag(diag_A)
        m_max = CASE2_SPEC.m_max if CASE2_SPEC.m_max is not None else NOTAY_N

        precond = variant_a_dynamic_precond_factory(expected.epsilon)

        _, result = flexible_cg(
            A,
            b,
            x0=x0,
            preconditioner=precond,
            rtol=NOTAY_TOLERANCE,
            atol=0.0,
            m_max=m_max,
            maxiter=500,
            trace_mode=TraceMode.MINIMAL,
        )

        # Just verify convergence
        assert result.converged, (
            f"FCG({CASE2_SPEC.m_max}) did not converge at ε={expected.epsilon} (dynamic variant)"
        )


@pytest.mark.benchmark
@pytest.mark.slow
class TestCase3VariantA:
    """Case 3 + Variant (a): Isolated eigenvalue (κ=1100) + random perturbation.

    Paper Table 1 results (n=10^4, δ=10^-6, FCG(∞)):
        ε:    0   1e-2  1e-1  1/7   1/4   1/3   1/2
        iter: 31  31    32    33    37    40    49
    """

    @pytest.mark.parametrize(
        "expected",
        CASE3_VARIANT_A_RESULTS,
        ids=lambda r: f"eps_{r.epsilon:.4f}",
    )
    def test_iterations_match_paper(
        self,
        case3_system: tuple[NDArray, NDArray, NDArray],
        variant_a_precond_factory: Callable[[float], Callable[[NDArray], NDArray]],
        expected: NotayResult,
    ) -> None:
        """Verify FCG iteration counts match paper's Table 1."""
        diag_A, b, x0 = case3_system
        A = np.diag(diag_A)
        m_max = CASE3_SPEC.m_max if CASE3_SPEC.m_max is not None else NOTAY_N

        precond = variant_a_precond_factory(expected.epsilon)

        _, result = flexible_cg(
            A,
            b,
            x0=x0,
            preconditioner=precond,
            rtol=NOTAY_TOLERANCE,
            atol=0.0,
            m_max=m_max,
            maxiter=500,
            trace_mode=TraceMode.MINIMAL,
        )

        atol = compute_tolerance(expected.epsilon, expected.iterations)
        assert result.converged, f"FCG({CASE3_SPEC.m_max}) did not converge at ε={expected.epsilon}"
        assert np.isclose(result.iterations, expected.iterations, rtol=0, atol=atol), (
            f"ε={expected.epsilon}: got {result.iterations}, "
            f"expected {expected.iterations} (±{atol:.1f})"
        )


@pytest.mark.benchmark
@pytest.mark.slow
class TestCase3VariantADynamic:
    """Case 3 + Variant (a-dynamic): Fresh random perturbation each iteration.

    This variant generates a new random perturbation at each iteration,
    unlike the paper's fixed-f approach. Results will differ from Table 1.
    """

    @pytest.mark.parametrize(
        "expected",
        CASE3_VARIANT_A_RESULTS,
        ids=lambda r: f"eps_{r.epsilon:.4f}",
    )
    def test_dynamic_perturbation(
        self,
        case3_system: tuple[NDArray, NDArray, NDArray],
        variant_a_dynamic_precond_factory: Callable[[float], Callable[[NDArray], NDArray]],
        expected: NotayResult,
    ) -> None:
        """Test FCG with dynamic random perturbations."""
        diag_A, b, x0 = case3_system
        A = np.diag(diag_A)
        m_max = CASE3_SPEC.m_max if CASE3_SPEC.m_max is not None else NOTAY_N

        precond = variant_a_dynamic_precond_factory(expected.epsilon)

        _, result = flexible_cg(
            A,
            b,
            x0=x0,
            preconditioner=precond,
            rtol=NOTAY_TOLERANCE,
            atol=0.0,
            m_max=m_max,
            maxiter=500,
            trace_mode=TraceMode.MINIMAL,
        )

        # Just verify convergence
        assert result.converged, (
            f"FCG({CASE3_SPEC.m_max}) did not converge at ε={expected.epsilon} (dynamic variant)"
        )
