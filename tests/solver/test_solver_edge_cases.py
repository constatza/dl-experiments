"""Edge case tests for iterative solvers.

Tests unusual but valid solver inputs and boundary conditions across all solvers
(FCG, PCG, SciPy CG) to ensure robust error handling and correct behavior.

Test categories:
1. Trivial convergence (zero RHS, already converged)
2. Iteration limits and early stopping
3. Tolerance edge cases (very small RHS, atol dominance)
4. Numerical edge cases (near-singular, identity matrix)
5. Initial guess variations
"""

from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from neuralls.solver.monitoring.trace_mode import TraceMode
from neuralls.solver.preconditioners import FunctionPreconditioner


class TestTrivialConvergence:
    """Test cases where solver should converge immediately."""

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_zero_rhs_converges_immediately(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_spd_small: NDArray,
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Zero RHS should converge in 0 iterations with x=0."""
        solver = solver_factories[solver_name]
        A = tridiagonal_spd_small
        b = np.zeros(A.shape[0])
        rtol, atol = integration_tolerances

        x, result = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Solution should be zero vector
        np.testing.assert_allclose(x, np.zeros_like(b), rtol=rtol, atol=atol)

        # Should converge immediately (0 or 1 iterations depending on implementation)
        assert result.iterations <= 1, f"Expected ≤1 iterations for zero RHS, got {result.iterations}"
        assert result.converged, "Should converge for zero RHS"

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_already_converged_initial_guess(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Initial guess equal to exact solution should converge immediately."""
        solver = solver_factories[solver_name]
        A, b, x_exact = tridiagonal_system_known_solution
        rtol, atol = integration_tolerances

        # Start with exact solution
        x, result = solver(
            A, b, x0=x_exact.copy(), rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL
        )

        # Should return exact solution
        np.testing.assert_allclose(x, x_exact, rtol=rtol, atol=atol)

        # Should converge in 0 or 1 iterations
        assert result.iterations <= 1, f"Expected ≤1 iterations with exact x0, got {result.iterations}"
        assert result.converged, "Should converge when starting at exact solution"

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_identity_matrix_converges_immediately(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Identity matrix system should converge in 1 iteration."""
        solver = solver_factories[solver_name]
        n = 10
        A = np.eye(n)
        b = np.random.RandomState(42).randn(n)
        rtol, atol = integration_tolerances

        x, result = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Solution should equal RHS for identity matrix
        np.testing.assert_allclose(x, b, rtol=rtol, atol=atol)

        # Identity matrix should converge in 1 iteration (exact in Krylov theory)
        assert result.iterations <= 2, f"Expected ≤2 iterations for identity, got {result.iterations}"
        assert result.converged, "Should converge for identity matrix"


class TestIterationLimits:
    """Test max_iterations enforcement and early stopping."""

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_respects_max_iterations_limit(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_spd_small: NDArray,
        rhs_ones_small: NDArray,
    ) -> None:
        """Solver must stop at max_iterations even if not converged."""
        solver = solver_factories[solver_name]
        A = tridiagonal_spd_small
        b = rhs_ones_small
        max_iters = 3

        # Use very tight tolerance to prevent convergence
        x, result = solver(
            A, b, rtol=1e-15, atol=1e-20, max_iterations=max_iters, trace_mode=TraceMode.MINIMAL
        )

        # Should stop at max iterations
        assert result.iterations == max_iters, f"Expected {max_iters} iterations, got {result.iterations}"

        # May or may not converge (depends on problem)
        # Just verify it stopped at the limit
        assert result.iterations <= max_iters

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    @pytest.mark.parametrize("max_iters", [1, 5, 10])
    def test_various_iteration_limits(
        self,
        solver_name: str,
        max_iters: int,
        solver_factories: dict[str, Callable],
        diagonal_spd_small: NDArray,
        rhs_ones_small: NDArray,
    ) -> None:
        """Test different max_iterations values."""
        solver = solver_factories[solver_name]
        A = diagonal_spd_small
        b = rhs_ones_small

        x, result = solver(
            A, b, rtol=1e-6, atol=1e-14, max_iterations=max_iters, trace_mode=TraceMode.MINIMAL
        )

        # Iterations should not exceed limit
        assert result.iterations <= max_iters, (
            f"Exceeded max_iterations: {result.iterations} > {max_iters}"
        )


class TestToleranceEdgeCases:
    """Test absolute vs relative tolerance behavior in edge cases."""

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_very_small_rhs_uses_atol(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_spd_small: NDArray,
    ) -> None:
        """Very small RHS should trigger atol-dominated convergence."""
        solver = solver_factories[solver_name]
        A = tridiagonal_spd_small

        # RHS with very small norm (||b|| ≈ 1e-10)
        b = np.full(A.shape[0], 1e-11)
        rtol = 1e-6
        atol = 1e-14

        x, result = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Should converge (atol dominates since ||b|| is tiny)
        assert result.converged, "Should converge with atol for very small RHS"

        # Verify residual satisfies atol criterion
        residual = b - A @ x
        residual_norm = float(np.linalg.norm(residual))
        b_norm = float(np.linalg.norm(b))

        # Should satisfy: ||r|| ≤ atol (since rtol * ||b|| is even smaller)
        assert residual_norm <= atol or residual_norm <= rtol * b_norm + atol

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_loose_rtol_converges_quickly(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    ) -> None:
        """Loose rtol should require fewer iterations."""
        solver = solver_factories[solver_name]
        A, b, x_exact = tridiagonal_system_known_solution

        # Very loose tolerance
        loose_rtol = 1e-2
        atol = 1e-14

        x, result = solver(A, b, rtol=loose_rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Should converge quickly with loose tolerance
        assert result.converged, "Should converge with loose tolerance"
        assert result.iterations <= 20, f"Expected quick convergence, got {result.iterations} iterations"

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_tight_rtol_requires_more_iterations(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_spd_small: NDArray,
        rhs_ones_small: NDArray,
    ) -> None:
        """Tight rtol should require more iterations than loose rtol."""
        solver = solver_factories[solver_name]
        A = tridiagonal_spd_small
        b = rhs_ones_small
        atol = 1e-14

        # Solve with loose tolerance
        _, result_loose = solver(A, b, rtol=1e-2, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Solve with tight tolerance
        _, result_tight = solver(A, b, rtol=1e-12, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Tight tolerance should require at least as many iterations
        assert result_tight.iterations >= result_loose.iterations, (
            f"Tight tolerance ({result_tight.iterations} iters) should need "
            f"≥ loose tolerance ({result_loose.iterations} iters)"
        )


class TestInitialGuessVariations:
    """Test different initial guess strategies."""

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_zero_initial_guess(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Explicit zero initial guess should work correctly."""
        solver = solver_factories[solver_name]
        A, b, x_exact = tridiagonal_system_known_solution
        rtol, atol = integration_tolerances

        x0 = np.zeros_like(b)
        x, result = solver(A, b, x0=x0, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Should converge to correct solution
        np.testing.assert_allclose(x, x_exact, rtol=rtol, atol=atol)
        assert result.converged

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_random_initial_guess(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Random initial guess should still converge to correct solution."""
        solver = solver_factories[solver_name]
        A, b, x_exact = tridiagonal_system_known_solution
        rtol, atol = integration_tolerances

        # Random initial guess
        rng = np.random.RandomState(123)
        x0 = rng.randn(len(b)) * 10.0  # Scale up for harder initial guess

        x, result = solver(A, b, x0=x0, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Should converge to correct solution regardless of x0
        np.testing.assert_allclose(x, x_exact, rtol=rtol, atol=atol)
        assert result.converged

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_good_initial_guess_reduces_iterations(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Good initial guess should require fewer iterations than zero guess."""
        solver = solver_factories[solver_name]
        A, b, x_exact = tridiagonal_system_known_solution
        rtol, atol = integration_tolerances

        # Zero initial guess
        _, result_zero = solver(
            A, b, x0=np.zeros_like(b), rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL
        )

        # Good initial guess (80% of exact solution)
        x0_good = 0.8 * x_exact
        _, result_good = solver(A, b, x0=x0_good, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Good guess should need fewer or equal iterations
        assert result_good.iterations <= result_zero.iterations, (
            f"Good initial guess ({result_good.iterations} iters) should need "
            f"≤ zero guess ({result_zero.iterations} iters)"
        )


class TestNumericalEdgeCases:
    """Test numerical edge cases and conditioning effects on iteration count."""

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_ill_conditioned_requires_more_iterations(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Ill-conditioned matrix (κ ≈ 1e4) requires more iterations than well-conditioned (κ ≈ 2)."""
        solver = solver_factories[solver_name]
        rtol, atol = integration_tolerances
        n = 50
        rng = np.random.RandomState(456)

        # Well-conditioned: diagonal with similar eigenvalues (κ ≈ 2)
        A_well = np.diag(np.linspace(1.0, 2.0, n))
        x_well = rng.randn(n)
        b_well = A_well @ x_well  # Generate RHS from exact solution
        _, result_well = solver(A_well, b_well, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL, max_iterations=500)

        # Ill-conditioned: diagonal with widely spread eigenvalues (κ ≈ 1e4)
        A_ill = np.diag(np.logspace(-2, 2, n))
        x_ill = rng.randn(n)
        b_ill = A_ill @ x_ill  # Generate RHS from exact solution
        _, result_ill = solver(A_ill, b_ill, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL, max_iterations=500)

        # SRP: Only check iteration count comparison (not solution accuracy)
        assert result_ill.iterations >= result_well.iterations, (
            f"Ill-conditioned κ≈1e4 ({result_ill.iterations} iters) should need "
            f"≥ well-conditioned κ≈2 ({result_well.iterations} iters)"
        )
        assert result_well.converged and result_ill.converged, "Both should converge"

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_preconditioner_reduces_iterations(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        jacobi_preconditioner_medium: Callable[[NDArray], NDArray],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Jacobi preconditioner reduces iteration count on 50×50 tridiagonal system."""
        solver = solver_factories[solver_name]
        rtol, atol = integration_tolerances
        n = 50
        rng = np.random.RandomState(789)

        # Create tridiagonal SPD matrix (κ ≈ 160)
        diagonal = 4.0 * np.ones(n)
        off_diagonal = -1.0 * np.ones(n - 1)
        A = np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)
        x_exact = rng.randn(n)
        b = A @ x_exact  # Generate RHS from exact solution

        # Without preconditioner
        _, result_no_precond = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # With Jacobi preconditioner
        _, result_precond = solver(
            A, b, preconditioner=jacobi_preconditioner_medium,
            rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL
        )

        # SRP: Only check that preconditioner helps with iteration count
        assert result_precond.iterations <= result_no_precond.iterations, (
            f"Preconditioned ({result_precond.iterations}) should need "
            f"≤ unpreconditioned ({result_no_precond.iterations}) iterations"
        )
        assert result_no_precond.converged and result_precond.converged, "Both should converge"


class TestResultConsistency:
    """Test consistency of result metadata across solvers."""

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_result_has_required_fields(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Result should have all required fields populated."""
        solver = solver_factories[solver_name]
        A, b, x_exact = tridiagonal_system_known_solution
        rtol, atol = integration_tolerances

        x, result = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Check required fields exist and have reasonable values
        assert hasattr(result, 'converged'), "Result missing 'converged' field"
        assert hasattr(result, 'iterations'), "Result missing 'iterations' field"
        assert hasattr(result, 'residual_history'), "Result missing 'residual_history' field"

        assert isinstance(result.converged, bool), "converged should be bool"
        assert isinstance(result.iterations, int), "iterations should be int"
        assert result.iterations >= 0, "iterations should be non-negative"

        # Residual history should have at least initial residual
        assert len(result.residual_history) > 0, "residual_history should not be empty"

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_residual_history_length_matches_iterations(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        diagonal_spd_small: NDArray,
        rhs_ones_small: NDArray,
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Length of residual_history should match iterations + 1 (includes initial)."""
        solver = solver_factories[solver_name]
        A = diagonal_spd_small
        b = rhs_ones_small
        rtol, atol = integration_tolerances

        x, result = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Residual history includes iteration 0 (initial residual)
        expected_length = result.iterations + 1
        actual_length = len(result.residual_history)

        assert actual_length == expected_length, (
            f"Residual history length ({actual_length}) should equal "
            f"iterations + 1 ({expected_length})"
        )

    @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
    def test_residual_history_decreases_or_stable(
        self,
        solver_name: str,
        solver_factories: dict[str, Callable],
        tridiagonal_spd_small: NDArray,
        rhs_ones_small: NDArray,
        integration_tolerances: tuple[float, float],
    ) -> None:
        """Residual history should generally decrease (allowing small numerical increases)."""
        solver = solver_factories[solver_name]
        A = tridiagonal_spd_small
        b = rhs_ones_small
        rtol, atol = integration_tolerances

        x, result = solver(A, b, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

        # Check residuals are non-increasing with small tolerance for numerical noise
        residuals = result.residual_history
        for i in range(1, len(residuals)):
            # Allow 10% increase for numerical reasons (e.g., flexible CG, finite precision)
            assert residuals[i] <= residuals[i-1] * 1.1, (
                f"Residual increased significantly at iteration {i}: "
                f"{residuals[i-1]:.2e} → {residuals[i]:.2e}"
            )