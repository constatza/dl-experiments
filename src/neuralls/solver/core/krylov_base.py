"""Base class for Krylov subspace iterative solvers.

This module provides the abstract base class for Krylov methods (CG, PCG, FCG,
GMRES, BiCGSTAB, etc.) that share a common iteration structure based on Krylov
subspace theory.

Mathematical Background (Notay 2000):
    Krylov methods build approximate solutions in the Krylov subspace:
        K_i(A, r_0) = span{r_0, A*r_0, A^2*r_0, ..., A^(i-1)*r_0}

    All Krylov methods follow this general iteration pattern:
    1. Apply preconditioner to residual: w_i = M^{-1} r_i
    2. Construct search direction/basis vector (method-specific)
    3. Apply matrix-vector product: q_i = A * d_i
    4. Orthogonalize direction (method-specific: CG uses conjugacy, GMRES uses Arnoldi)
    5. Compute coefficients (step length, beta, etc.)
    6. Update solution: u_{i+1} = u_i + α_i * d_i
    7. Update residual: r_{i+1} = r_i - α_i * q_i

Design Principles:
    - Template Method: _iterate_step() implements common Krylov iteration
    - Open/Closed: Open for extension (CG, GMRES), closed for modification
    - Strategy Pattern: Delegates method-specific steps to abstract hooks
    - Single Responsibility: Krylov-specific logic only

References:
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
    - Greenbaum, A. (1997). Iterative Methods for Solving Linear Systems.
    - Golub & Van Loan (2013). Matrix Computations, Chapter 11.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from collections.abc import Callable


from .base import IterativeSolverBase
from ..models.state import KrylovState, SolverState

if TYPE_CHECKING:
    from numpy.typing import NDArray


class KrylovSolverBase(IterativeSolverBase):
    """Abstract base for Krylov subspace iterative solvers.

    This class implements the common iteration structure shared by all Krylov
    methods while allowing subclasses to customize method-specific operations
    like direction computation and orthogonalization.

    The class follows the Template Method pattern, where _iterate_step()
    orchestrates the Krylov iteration while delegating method-specific steps
    to abstract hooks.

    Krylov Iteration Structure (Notay 2000, implemented in _iterate_step):
        1. Apply preconditioner: w_i = M^{-1}(r_i)
        2. Compute search direction: d_i = update_direction(w_i, state)
        3. Apply matrix: q_i = A @ d_i
        4. Compute coefficients: α_i, β_i = compute_coefficients(d_i, q_i, state)
        5. Update solution: u_{i+1} = u_i + α_i * d_i
        6. Update residual: r_{i+1} = r_i - α_i * q_i
        7. Check breakdown/convergence

    Protected Abstract Hooks (Krylov-specific):
        _update_direction: Compute search direction from preconditioned residual
        _compute_coefficients: Compute step length and update coefficients
        _update_vectors: Update solution, residual, and auxiliary vectors
        _check_breakdown: Check for numerical breakdown

    Theory Note:
        The Krylov subspace K_i(A, r_0) has dimension at most i. Krylov methods
        differ in how they select u_i from this subspace:
        - CG: Minimize A-norm of error (SPD systems only)
        - GMRES: Minimize residual norm (general systems)
        - BiCGSTAB: Combination of BiCG and stabilization (non-symmetric)

    References:
        - Saad (2003), Section 6: Krylov Subspace Methods
        - Greenbaum (1997), Chapter 3: The Conjugate Gradient Method
    """

    # Krylov-specific abstract hooks

    @abstractmethod
    def _update_direction(
        self,
        w: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> NDArray:
        """Compute search direction from preconditioned residual.

        This is the key method-specific operation that differentiates Krylov methods:
        - CG: d_i = w_i + β_i * d_{i-1} (two-term recurrence)
        - FCG: d_i = w_i - Σ[(w_i, q_j)/(d_j, q_j)] * d_j (truncated orthogonalization)
        - GMRES: Arnoldi orthogonalization against all previous basis vectors

        Args:
            w: Preconditioned residual w_i = M^{-1}(r_i) (Notay 2000)
            state: Current Krylov state containing:
                - Previous directions d_{i-1}, d_{i-2}, ... (for recurrence)
                - Previous q vectors q_{i-1}, q_{i-2}, ... (for orthogonalization)
                - Other method-specific history
            **kwargs: Method-specific parameters (window size, tolerances, etc.)

        Returns:
            Search direction d_i (Notay 2000)

        Theory:
            The search direction must satisfy method-specific properties:
            - CG: A-conjugacy: (d_i, A*d_j) = 0 for i ≠ j
            - GMRES: Orthogonality: (d_i, d_j) = 0 for i ≠ j
            - FCG: Approximate A-conjugacy via truncated orthogonalization
        """
        ...

    @abstractmethod
    def _compute_coefficients(
        self,
        d: NDArray,
        q: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        """Compute step length and update coefficients.

        Computes:
        1. Step length α_i for solution/residual update
        2. Additional coefficients (β_i for CG, Givens rotations for GMRES, etc.)

        Args:
            d: Search direction d_i (Notay 2000)
            q: Matrix-vector product q_i = A @ d_i
            state: Current Krylov state
            **kwargs: Method-specific parameters

        Returns:
            Tuple (step_length, coefficients):
            - step_length: α_i for u_{i+1} = u_i + α_i * d_i
            - coefficients: Dict of additional coefficients (β, γ, etc.)

        Theory (Notay 2000):
            Step length minimizes residual or error in specific norm:
            - CG: α_i = (r_i, w_i) / (d_i, q_i) minimizes A-norm of error
            - GMRES: Least-squares minimization of residual
            - FCG: Same as CG but with flexible preconditioning
        """
        ...

    @abstractmethod
    def _update_vectors(
        self,
        state: KrylovState,
        d: NDArray,
        q: NDArray,
        w: NDArray,
        step_length: float,
        coefficients: dict[str, float],
        **kwargs,
    ) -> KrylovState:
        """Update solution, residual, and auxiliary vectors.

        Updates (Notay 2000):
        1. Solution: u_{i+1} = u_i + α_i * d_i
        2. Residual: r_{i+1} = r_i - α_i * q_i
        3. Auxiliary vectors (method-specific)

        Args:
            state: Current Krylov state
            d: Search direction used in this iteration d_i (Notay 2000)
            q: Matrix-vector product q_i = A @ d_i
            w: Preconditioned residual w_i = M^{-1}(r_i) (Notay 2000)
            step_length: Step length α_i
            coefficients: Additional update coefficients
            **kwargs: Method-specific parameters

        Returns:
            New KrylovState with updated vectors and iteration count

        Theory:
            The update preserves Krylov subspace membership:
            u_{i+1} ∈ u_0 + K_{i+1}(A, r_0)
        """
        ...

    @abstractmethod
    def _check_breakdown(
        self,
        state: KrylovState,
        **kwargs,
    ) -> bool:
        """Check for numerical breakdown.

        Breakdown conditions vary by method:
        - CG: Negative curvature (d^T A d <= 0)
        - GMRES: Linear dependence in Krylov basis
        - BiCGSTAB: Division by zero in update formulas

        Args:
            state: Current Krylov state
            **kwargs: Method-specific breakdown tolerances

        Returns:
            True if breakdown detected, False otherwise

        Theory:
            Breakdown indicates:
            1. Loss of numerical stability
            2. Preconditioner issues (for PCG/FCG)
            3. Matrix properties violated (e.g., non-SPD for CG)
        """
        ...

    # Concrete Krylov iteration template

    def _iterate_step(
        self,
        linear_op: Callable[[NDArray], NDArray],
        state: SolverState,  # TODO: why isn't this krylov state?
        **kwargs,
    ) -> SolverState:
        """Execute single Krylov iteration.

        This method implements the common Krylov iteration structure as a
        template method. Subclasses customize behavior via abstract hooks.

        Iteration Steps:
            1. Apply preconditioner to residual
            2. Compute search direction (method-specific)
            3. Apply matrix-vector product
            4. Compute coefficients (method-specific)
            5. Update solution, residual, auxiliary vectors
            6. Check for breakdown

        Args:
            linear_op: Linear operator A
            state: Current Krylov state (contains x, r, and method-specific vectors)
            **kwargs: Solver parameters (preconditioner, tolerances, etc.)

        Returns:
            Updated KrylovState for next iteration

        Theory:
            This iteration maintains the fundamental Krylov subspace property:
                u_i ∈ u_0 + K_i(A, r_0)
                r_i ∈ K_i(A, r_0)
        """
        # All Krylov solvers use KrylovState or subclasses
        # Extract current state via Protocol
        from ..models.protocols import HasVectors

        if not isinstance(state, HasVectors):
            # Should never happen - Krylov solvers always have vectors
            raise TypeError(f"Expected state with vectors, got {type(state)}")

        r_i = state.r

        # Step 1: Apply preconditioner (Notay 2000: w_i = M^{-1} r_i)
        # Prefer solver's own preconditioner, fall back to kwargs
        preconditioner = getattr(self, "preconditioner", None) or kwargs.get(
            "preconditioner"
        )
        if preconditioner is not None:
            w_i = self._apply_preconditioner(preconditioner, r_i)
        else:
            w_i = r_i.copy()

        # Step 2: Compute search direction (method-specific, Notay 2000: d_i)
        d_i = self._update_direction(w_i, state, **kwargs)

        # Step 3: Apply matrix-vector product (Notay 2000: q_i = A d_i)
        q_i = linear_op(d_i)

        # Step 4: Compute coefficients (method-specific)
        alpha_i, coeffs = self._compute_coefficients(d_i, q_i, state, **kwargs)

        # Step 5: Update vectors (method-specific)
        new_state = self._update_vectors(
            state, d_i, q_i, w_i, alpha_i, coeffs, **kwargs
        )

        # Step 6: Check for breakdown
        if self._check_breakdown(new_state, **kwargs):
            # Create new state with breakdown flag
            from dataclasses import replace

            new_state = replace(new_state, breakdown=True)

        return new_state
