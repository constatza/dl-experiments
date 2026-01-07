"""Base class for Krylov subspace iterative solvers.

This module provides the abstract base class for Krylov methods (CG, PCG, FCG,
GMRES, BiCGSTAB, etc.) that share a common iteration structure based on Krylov
subspace theory.

Mathematical Background:
    Krylov methods build approximate solutions in the Krylov subspace:
        K_k(A, r_0) = span{r_0, A*r_0, A^2*r_0, ..., A^(k-1)*r_0}

    All Krylov methods follow this general iteration pattern:
    1. Apply preconditioner to residual: z_k = M^{-1} r_k
    2. Construct search direction/basis vector (method-specific)
    3. Apply matrix-vector product: q_k = A * p_k
    4. Orthogonalize direction (method-specific: CG uses conjugacy, GMRES uses Arnoldi)
    5. Compute coefficients (step length, beta, etc.)
    6. Update solution: x_{k+1} = x_k + α_k * p_k
    7. Update residual: r_{k+1} = r_k - α_k * q_k

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

    Krylov Iteration Structure (implemented in _iterate_step):
        1. Apply preconditioner: z_k = M^{-1}(r_k)
        2. Compute search direction: p_k = update_direction(z_k, state)
        3. Apply matrix: q_k = A @ p_k
        4. Compute coefficients: α_k, β_k = compute_coefficients(p_k, q_k, state)
        5. Update solution: x_{k+1} = x_k + α_k * p_k
        6. Update residual: r_{k+1} = r_k - α_k * q_k
        7. Check breakdown/convergence

    Protected Abstract Hooks (Krylov-specific):
        _update_direction: Compute search direction from preconditioned residual
        _compute_coefficients: Compute step length and update coefficients
        _update_vectors: Update solution, residual, and auxiliary vectors
        _check_breakdown: Check for numerical breakdown

    Theory Note:
        The Krylov subspace K_k(A, r_0) has dimension at most k. Krylov methods
        differ in how they select x_k from this subspace:
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
        z: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> NDArray:
        """Compute search direction from preconditioned residual.

        This is the key method-specific operation that differentiates Krylov methods:
        - CG: p_k = z_k + β_k * p_{k-1} (two-term recurrence)
        - FCG: p_k = z_k - Σ[(z_k, q_j)/(p_j, q_j)] * p_j (truncated orthogonalization)
        - GMRES: Arnoldi orthogonalization against all previous basis vectors

        Args:
            z: Preconditioned residual z_k = M^{-1}(r_k)
            state: Current Krylov state containing:
                - Previous directions p_{k-1}, p_{k-2}, ... (for recurrence)
                - Previous q vectors q_{k-1}, q_{k-2}, ... (for orthogonalization)
                - Other method-specific history
            **kwargs: Method-specific parameters (window size, tolerances, etc.)

        Returns:
            Search direction p_k

        Theory:
            The search direction must satisfy method-specific properties:
            - CG: A-conjugacy: (p_i, A*p_j) = 0 for i ≠ j
            - GMRES: Orthogonality: (p_i, p_j) = 0 for i ≠ j
            - FCG: Approximate A-conjugacy via truncated orthogonalization
        """
        ...

    @abstractmethod
    def _compute_coefficients(
        self,
        p: NDArray,
        q: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        """Compute step length and update coefficients.

        Computes:
        1. Step length α_k for solution/residual update
        2. Additional coefficients (β_k for CG, Givens rotations for GMRES, etc.)

        Args:
            p: Search direction p_k
            q: Matrix-vector product q_k = A @ p_k
            state: Current Krylov state
            **kwargs: Method-specific parameters

        Returns:
            Tuple (step_length, coefficients):
            - step_length: α_k for x_{k+1} = x_k + α_k * p_k
            - coefficients: Dict of additional coefficients (β, γ, etc.)

        Theory:
            Step length minimizes residual or error in specific norm:
            - CG: α_k = (r_k, z_k) / (p_k, q_k) minimizes A-norm of error
            - GMRES: Least-squares minimization of residual
            - FCG: Same as CG but with flexible preconditioning
        """
        ...

    @abstractmethod
    def _update_vectors(
        self,
        state: KrylovState,
        p: NDArray,
        q: NDArray,
        z: NDArray,
        step_length: float,
        coefficients: dict[str, float],
        **kwargs,
    ) -> KrylovState:
        """Update solution, residual, and auxiliary vectors.

        Updates:
        1. Solution: x_{k+1} = x_k + α_k * p_k
        2. Residual: r_{k+1} = r_k - α_k * q_k
        3. Auxiliary vectors (method-specific)

        Args:
            state: Current Krylov state
            p: Search direction used in this iteration
            q: Matrix-vector product q = A @ p
            z: Preconditioned residual z = M^{-1}(r)
            step_length: Step length α_k
            coefficients: Additional update coefficients
            **kwargs: Method-specific parameters

        Returns:
            New KrylovState with updated vectors and iteration count

        Theory:
            The update preserves Krylov subspace membership:
            x_{k+1} ∈ x_0 + K_{k+1}(A, r_0)
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
        - CG: Negative curvature (p^T A p <= 0)
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
        state: SolverState,
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
                x_k ∈ x_0 + K_k(A, r_0)
                r_k ∈ K_k(A, r_0)
        """
        # All Krylov solvers use KrylovState or subclasses
        # Extract current state via Protocol
        from ..models.protocols import HasVectors

        if not isinstance(state, HasVectors):
            # Should never happen - Krylov solvers always have vectors
            raise TypeError(f"Expected state with vectors, got {type(state)}")

        r_k = state.r
        preconditioner = kwargs.get("preconditioner")

        # Step 1: Apply preconditioner
        if preconditioner is not None:
            z_k = self._apply_preconditioner(preconditioner, r_k)
        else:
            z_k = r_k.copy()

        # Step 2: Compute search direction (method-specific)
        p_k = self._update_direction(z_k, state, **kwargs)

        # Step 3: Apply matrix-vector product
        q_k = linear_op(p_k)

        # Step 4: Compute coefficients (method-specific)
        alpha_k, coeffs = self._compute_coefficients(p_k, q_k, state, **kwargs)

        # Step 5: Update vectors (method-specific)
        new_state = self._update_vectors(
            state, p_k, q_k, z_k, alpha_k, coeffs, **kwargs
        )

        # Step 6: Check for breakdown
        if self._check_breakdown(new_state, **kwargs):
            # Create new state with breakdown flag
            from dataclasses import replace

            new_state = replace(new_state, breakdown=True)

        return new_state
