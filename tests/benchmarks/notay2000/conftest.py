"""Shared fixtures for Notay 2000 Section 5.1 artificial experiments.

All tests use paper's exact parameters:
- n = 10^4
- δ = 10^-6 (tolerance on relative error in A-norm)
- Pseudorandom RHS (uniform distribution in [-1, 1])
- Zero initial guess

Variant (a) preconditioner implementations:
- variant_a_precond_factory: Fixed perturbation vector (paper version)
- variant_a_dynamic_precond_factory: Fresh random perturbation each iteration (alternative)

Reference:
    Notay, Y. (2000). Flexible Conjugate Gradients.
    SIAM Journal on Scientific Computing, 22(4), 1444-1460.
    Section 5.1: Some artificial experiments
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Paper Parameters (Section 5.1)
# =============================================================================

NOTAY_N = 10_000  # n = 10^4
NOTAY_TOLERANCE = 1e-6  # δ = 10^-6
NOTAY_SEED_RHS = 42  # Seed for right-hand side b
NOTAY_SEED_PERT = 123  # Seed for perturbation vector f

# =============================================================================
# Iteration Comparison Tolerances
# =============================================================================
# At epsilon=0: should match closely (base tolerance only).
# At epsilon>0: RNG effect scales with epsilon, so tolerance scales too.
# Formula: atol = BASE_ATOL + EPSILON_SCALE * epsilon * expected_iterations
#
# Note: Using the default euclidean norm (L2) gives excellent agreement with
# the paper's results. Case 2 at ε=0 matches exactly (49 iterations).
# Small discrepancies (±1-2) in other cases likely due to iteration counting
# conventions or numerical precision differences.

ITERATION_BASE_ATOL = 2  # Base absolute tolerance (±2 iterations)
ITERATION_EPSILON_SCALE = 1.5  # Scale factor for epsilon-dependent tolerance


# =============================================================================
# Test Case Specifications
# =============================================================================


@dataclass(frozen=True)
class NotayTestCase:
    """Specification for a Notay 2000 Section 5.1 test case.

    Attributes:
        name: Human-readable name.
        eigenvalue_formula: Description of eigenvalue distribution.
        kappa: Condition number.
        m_max: Truncation parameter (1 for FCG(1), None for FCG(∞)).
    """

    name: str
    eigenvalue_formula: str
    kappa: float
    m_max: int | None


@dataclass(frozen=True)
class NotayResult:
    """Expected result for a specific (case, epsilon) combination.

    Attributes:
        epsilon: Perturbation magnitude ε.
        iterations: Expected iteration count from Table 1.
    """

    epsilon: float
    iterations: int


# Case specifications
CASE1_SPEC = NotayTestCase(
    name="Case 1",
    eigenvalue_formula="λ_i = 1 + 5*(i-1)/(n-1)",
    kappa=6,
    m_max=1,
)

CASE2_SPEC = NotayTestCase(
    name="Case 2",
    eigenvalue_formula="λ_i = 1 + 50*(i-1)/(n-1)",
    kappa=51,
    m_max=1,
)

CASE3_SPEC = NotayTestCase(
    name="Case 3",
    eigenvalue_formula="λ_1=0.01, λ_i=1+10*(i-2)/(n-2) for i>1",
    kappa=1100,
    m_max=-1,  # FCG(∞)
)

# Epsilon values from Table 1
EPSILON_VALUES = [0.0, 1e-2, 1e-1, 1 / 7, 1 / 4, 1 / 3, 1 / 2]

# Expected iterations from Table 1 (variant a)
CASE1_ITERATIONS = [15, 15, 16, 17, 19, 22, 28]
CASE2_ITERATIONS = [49, 49, 55, 59, 69, 81, 116]
CASE3_ITERATIONS = [31, 31, 32, 33, 37, 40, 49]

# Combined results for parametrization
CASE1_VARIANT_A_RESULTS = [
    NotayResult(eps, iters) for eps, iters in zip(EPSILON_VALUES, CASE1_ITERATIONS)
]
CASE2_VARIANT_A_RESULTS = [
    NotayResult(eps, iters) for eps, iters in zip(EPSILON_VALUES, CASE2_ITERATIONS)
]
CASE3_VARIANT_A_RESULTS = [
    NotayResult(eps, iters) for eps, iters in zip(EPSILON_VALUES, CASE3_ITERATIONS)
]


# =============================================================================
# Eigenvalue Fixtures
# =============================================================================


@pytest.fixture
def case1_eigenvalues() -> NDArray:
    """Case 1: λ_i = 1 + 5*(i-1)/(n-1), κ = 6."""
    n = NOTAY_N
    return 1.0 + 5.0 * np.arange(n) / (n - 1)


@pytest.fixture
def case2_eigenvalues() -> NDArray:
    """Case 2: λ_i = 1 + 50*(i-1)/(n-1), κ = 51."""
    n = NOTAY_N
    return 1.0 + 50.0 * np.arange(n) / (n - 1)


@pytest.fixture
def case3_eigenvalues() -> NDArray:
    """Case 3: λ_1 = 0.01, λ_i = 1 + 10*(i-2)/(n-2) for i>1, κ = 1100."""
    n = NOTAY_N
    eigenvalues = np.zeros(n)
    eigenvalues[0] = 0.01
    eigenvalues[1:] = 1.0 + 10.0 * np.arange(n - 1) / (n - 2)
    return eigenvalues


# =============================================================================
# System Fixtures (Eigenvalues + RHS + Initial Guess)
# =============================================================================


def _build_system(diag_A: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Build linear system from diagonal matrix entries.

    Args:
        diag_A: Diagonal entries of matrix A.

    Returns:
        Tuple of (diag_A, b, x0).
    """
    n = len(diag_A)
    rng = np.random.default_rng(NOTAY_SEED_RHS)
    b = rng.uniform(-1.0, 1.0, size=n)
    x0 = np.zeros(n)
    return diag_A, b, x0


@pytest.fixture
def case1_system(case1_eigenvalues: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Case 1 system: (diag_A, b, x0)."""
    return _build_system(case1_eigenvalues)


@pytest.fixture
def case2_system(case2_eigenvalues: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Case 2 system: (diag_A, b, x0)."""
    return _build_system(case2_eigenvalues)


@pytest.fixture
def case3_system(case3_eigenvalues: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Case 3 system: (diag_A, b, x0)."""
    return _build_system(case3_eigenvalues)


# =============================================================================
# Perturbation Fixtures
# =============================================================================


@pytest.fixture
def perturbation_vector() -> NDArray:
    """Fixed perturbation vector f for variant (a)."""
    rng = np.random.default_rng(NOTAY_SEED_PERT)
    return rng.uniform(-1.0, 1.0, NOTAY_N)


@pytest.fixture
def variant_a_precond_factory(
    perturbation_vector: NDArray,
) -> Callable[[float], Callable[[NDArray], NDArray]]:
    """Factory for variant (a) preconditioner: w = r + ε*(||r||/||f||)*f.

    Uses a FIXED perturbation vector f (generated once).

    Returns:
        Factory: epsilon -> preconditioner callable
    """
    f = perturbation_vector
    f_norm = np.linalg.norm(f)

    def factory(epsilon: float) -> Callable[[NDArray], NDArray]:
        def precond(r: NDArray) -> NDArray:
            if epsilon == 0.0:
                return r.copy()
            r_norm = np.linalg.norm(r)
            return r + epsilon * (r_norm / f_norm) * f

        return precond

    return factory


@pytest.fixture
def variant_a_dynamic_precond_factory() -> Callable[[float], Callable[[NDArray], NDArray]]:
    """Factory for variant (a-dynamic) preconditioner: w = r + ε*(||r||/||f||)*f.

    Generates a FRESH random perturbation vector f at each iteration.

    Returns:
        Factory: epsilon -> preconditioner callable
    """

    def factory(epsilon: float) -> Callable[[NDArray], NDArray]:
        def precond(r: NDArray) -> NDArray:
            if epsilon == 0.0:
                return r.copy()
            # Generate fresh perturbation at each call
            rng = np.random.default_rng()
            f = rng.uniform(-1.0, 1.0, len(r))
            f_norm = np.linalg.norm(f)
            r_norm = np.linalg.norm(r)
            return r + epsilon * (r_norm / f_norm) * f

        return precond

    return factory


# =============================================================================
# Result Reporting
# =============================================================================
