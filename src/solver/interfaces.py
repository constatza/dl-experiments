"""Abstract base classes for solver interfaces.

This module defines the interface hierarchy for linear system solvers.
All solvers must implement ISolver, and iterative solvers should implement
IIterativeSolver.

Design Principles:
    - Interface Segregation: Minimal, focused interfaces
    - Dependency Inversion: Depend on abstractions, not implementations
    - Single Responsibility: Each interface has one clear purpose

Theory:
    ISolver defines the minimal contract for any solver (direct or iterative).
    IIterativeSolver extends this with iteration-specific methods like
    convergence checking.

Usage:
    >>> class MyIterativeSolver(IIterativeSolver):
    ...     def solve(self, A, b, x0=None, **kwargs):
    ...         # implementation
    ...         pass
    ...
    ...     def check_convergence(self, residual, rhs_norm, rtol, atol):
    ...         # implementation
    ...         pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .info import SolverResult
    from .convergence import IConvergenceCriterion


class ISolver(ABC):
    """Base interface for all linear system solvers.

    This interface defines the minimal contract that all solvers
    (direct and iterative) must satisfy. It follows the Interface
    Segregation Principle by providing only the essential solve() method.

    Mathematical Background:
        All solvers address the linear system:
            A x = b

        where:
            A is an n×n system matrix
            b is the n-dimensional right-hand side
            x is the n-dimensional solution

    Design Pattern:
        This is a Strategy interface enabling polymorphic solver selection.
        Different implementations provide different algorithms (CG, GMRES,
        direct solvers, etc.) while maintaining a consistent interface.

    Methods:
        solve: Solve the linear system Ax = b

    References:
        - SOLID Principles: Interface Segregation Principle
        - Design Patterns: Strategy Pattern
    """

    @abstractmethod
    def solve(
        self,
        A: NDArray,
        b: NDArray,
        x0: NDArray | None = None,
        **kwargs: object,
    ) -> tuple[NDArray, SolverResult]:
        """Solve linear system Ax = b.

        This is the core method that all solvers must implement. The exact
        algorithm is left to the implementation.

        Args:
            A: System matrix, shape (n, n). Must be square.
            b: Right-hand side vector, shape (n,).
            x0: Initial guess for solution, shape (n,). Optional.
                - Direct solvers typically ignore this
                - Iterative solvers use it as starting point
            **kwargs: Solver-specific parameters. Examples:
                - rtol: Relative tolerance
                - atol: Absolute tolerance
                - max_iterations: Maximum iteration count
                - preconditioner: Preconditioning strategy

        Returns:
            Tuple of (solution, solver_result):
            - solution (ndarray): Solution vector x, shape (n,)
            - solver_result (SolverResult): Detailed information about
                convergence, iterations, residuals, etc.

        Raises:
            ValueError: If A is not square or b has incompatible shape.
            NotImplementedError: If called on base class (must override).

        Theory:
            The solve() method abstracts away implementation details,
            allowing code to work with any solver algorithm. This enables:
            1. Runtime solver selection
            2. Easy testing with mock solvers
            3. Benchmarking different algorithms
            4. Swapping solvers without changing client code

        Examples:
            >>> solver = MySolver()
            >>> x, result = solver.solve(A, b, rtol=1e-6)
            >>> print(f"Converged: {result.converged}")
        """
        pass


class IIterativeSolver(ISolver):
    """Interface for iterative linear system solvers.

    Extends ISolver with iteration-specific methods including convergence
    checking. Iterative solvers build the solution incrementally, checking
    convergence at each step.

    Mathematical Background:
        Iterative solvers generate a sequence of approximations:
            x_0, x_1, x_2, ..., x_k

        The sequence converges when the residual r_k = b - A x_k satisfies:
            ||r_k|| <= max(rtol * ||b||, atol)

    Design Pattern:
        This interface adds iteration-specific responsibilities to ISolver
        while maintaining interface segregation. Only iterative solvers
        need to implement convergence checking.

    Methods:
        solve: Inherited from ISolver
        check_convergence: Check if iteration has converged

    References:
        - Saad (2003): "Iterative Methods for Sparse Linear Systems"
        - Interface Segregation Principle (SOLID)
    """

    @abstractmethod
    def check_convergence(
        self,
        residual: NDArray,
        rhs_norm: float,
        criterion: "IConvergenceCriterion",
    ) -> bool:
        """Check if convergence criterion is satisfied.

        This method implements the standard convergence test:
            ||r_k|| <= max(rtol * ||b||, atol) by default

        where:
            - criterion encapsulates relative/absolute tolerance handling

        Args:
            residual: Current residual vector r_k = b - A x_k, shape (n,).
            rhs_norm: Precomputed norm ||b|| of right-hand side.
            criterion: Convergence rule implementation (default matches SciPy).

        Returns:
            True if convergence criterion satisfied, False otherwise.

        Theory:
            Combined relative/absolute tolerance ensures:
            1. Relative tolerance: Solution accurate for typical problems
            2. Absolute tolerance: Prevents false convergence when ||b|| small
            3. max() operator: Use whichever tolerance is more permissive

            Why both tolerances?
            - If ||b|| is large: rtol * ||b|| dominates, scales with problem
            - If ||b|| is small: atol dominates, prevents over-solving
            - If ||b|| = 0: Only atol matters (special case)

        Mathematical Justification:
            The residual norm ||r_k|| measures the error in satisfying Ax = b.
            For SPD systems, the error ||x - x_k|| is bounded by:
                ||x - x_k||_A <= κ(A) * ||r_k|| / ||b||

            where κ(A) is the condition number. Thus controlling ||r_k||/||b||
            indirectly controls solution error.

        Examples:
            >>> solver = MyIterativeSolver()
            >>> r_k = b - A @ x_k
            >>> crit = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
            >>> converged = solver.check_convergence(r_k, norm(b), crit)
            >>> if converged:
            ...     print("Iteration converged!")

        References:
            - Golub & Van Loan (2013): "Matrix Computations", Section 10.2
            - Greenbaum (1997): "Iterative Methods for Solving Linear Systems"
        """
        pass
