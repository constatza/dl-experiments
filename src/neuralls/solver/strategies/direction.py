"""Direction computation strategies for Conjugate Gradient methods.

This module provides the DirectionStrategy abstraction and implementations for
computing search directions in CG variants. Follows the Strategy pattern to
enable composition over inheritance.

Design:
    - DirectionStrategy: Abstract base for all direction computation methods
    - TwoTermRecurrenceStrategy: Standard PCG two-term recurrence
    - OrthogonalizationDirectionStrategy: Adapter for OrthogonalizationStrategy
    - CompositeDirectionStrategy: Combines two-term recurrence with reorthogonalization

Theory (CG Algorithm Core):
    All CG variants follow the same structure:
        1. w_k = M^{-1}(r_k)           # Preconditioner
        2. d_k = compute_direction()   # METHOD-SPECIFIC (this module)
        3. q_k = A @ d_k               # Matrix-vector product
        4. α_k = (d_k, r_k) / (d_k, q_k)  # Step length
        5. u_{k+1} = u_k + α_k * d_k   # Update solution
        6. r_{k+1} = r_k - α_k * q_k   # Update residual

    Step 2 is the only behavioral difference between CG variants.

References:
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
    - Notay, Y. (2000). Flexible Conjugate Gradients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from ..utils.numerics import stable_dot_product
from .orthogonalization import OrthogonalizationStrategy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models.state import CGState


class DirectionStrategy(ABC):
    """Strategy for computing CG search direction.

    This abstraction unifies direction computation across CG variants,
    enabling composition over inheritance. Each variant differs only in
    how it computes the search direction from the preconditioned residual.

    Implementations:
        - TwoTermRecurrenceStrategy: d_k = w_k + β_k * d_{k-1} (PCG)
        - OrthogonalizationDirectionStrategy: d_k = orthogonalize(w_k, history) (FCG)
        - CompositeDirectionStrategy: Combines both (PCG + reorthogonalization)
    """

    @property
    @abstractmethod
    def required_history_size(self) -> int | None:
        """Return required history size for direction history.

        Returns:
            int: Maximum number of directions to retain.
            None: Unlimited history.
        """
        ...

    @abstractmethod
    def compute_direction(
        self,
        w: NDArray,
        state: CGState,
    ) -> NDArray:
        """Compute search direction from preconditioned residual.

        Args:
            w: Preconditioned residual w_k = M^{-1}(r_k)
            state: Current iteration state containing:
                - iteration: Iteration number (0-indexed)
                - r: Current residual r_k
                - d: Previous direction d_{k-1}
                - direction_history: Optional history for orthogonalization

        Returns:
            Search direction d_k for iteration k

        Theory:
            The direction computation is the only algorithmic difference
            between CG variants. All other steps (preconditioning, step
            length, updates) are identical.
        """
        ...


class TwoTermRecurrenceStrategy(DirectionStrategy):
    """Two-term recurrence for standard PCG.

    Implements the classic CG direction update using two-term recurrence:
        d_k = w_k + β_k * d_{k-1}

    where β_k is computed using the Fletcher-Reeves formula:
        β_k = (r_k, w_k) / (r_{k-1}, w_{k-1})

    This is memory-efficient (O(1) storage) and optimal for SPD systems
    with fixed SPD preconditioners.

    Design:
        This strategy is STATELESS - all required data is stored in CGState.
        Previous values (w_prev, r_prev, rw_prev) are extracted from state,
        making the strategy reusable across multiple solve() calls.

    Attributes:
        beta_formula: Formula for beta coefficient (default: fletcher_reeves)

    Theory:
        The two-term recurrence automatically maintains A-conjugacy in exact
        arithmetic. Only two directions are needed because of the special
        properties of the Krylov subspace and the conjugacy relation.

    References:
        - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
        - Saad (2003), Chapter 9: Krylov Subspace Methods.
    """

    def __init__(self, beta_formula: str = "fletcher_reeves"):
        """Initialize two-term recurrence strategy.

        Args:
            beta_formula: Beta formula to use. Currently only "fletcher_reeves"
                is implemented, which is the standard PCG formula.
        """
        self.beta_formula = beta_formula

    @property
    def required_history_size(self) -> int | None:
        """Two-term recurrence does not require direction history."""
        return 0

    def compute_direction(
        self,
        w: NDArray,
        state: CGState,
    ) -> NDArray:
        """Compute PCG direction using two-term recurrence.

        Formula:
            First iteration: d_0 = w_0 (steepest descent)
            Subsequent iterations: d_k = w_k + β_k * d_{k-1}

        where β_k = (r_k, w_k) / (r_{k-1}, w_{k-1}) (Fletcher-Reeves)

        Args:
            w: Preconditioned residual w_k
            state: Current CGState with iteration, r, d, w_prev, r_prev, rw_prev

        Returns:
            Search direction d_k

        Theory:
            Fletcher-Reeves formula ensures A-conjugacy when the preconditioner
            is fixed and SPD. The beta coefficient adjusts the direction to
            maintain orthogonality in the Krylov subspace.
        """
        # First iteration: steepest descent
        if state.iteration == 0:
            return w.copy()

        # Extract previous values from state
        rw_prev = state.rw_prev

        # Compute beta using Fletcher-Reeves formula (matches scipy.cg line 408)
        rw_curr = stable_dot_product(state.r, w)
        beta = rw_curr / rw_prev

        # Two-term recurrence
        return w + beta * state.d


class OrthogonalizationDirectionStrategy(DirectionStrategy):
    """Direction strategy using orthogonalization (FCG).

    Adapter that wraps an OrthogonalizationStrategy to compute directions
    via explicit orthogonalization against previous directions. This is
    the core of Flexible CG (Notay 2000).

    Formula:
        d_k = orthogonalize(w_k, {d_j}, {A d_j})

    This maintains approximate A-conjugacy through explicit Gram-Schmidt
    orthogonalization, enabling convergence with variable or non-SPD
    preconditioners.

    Attributes:
        orthogonalization: Orthogonalization strategy to use

    Theory:
        Unlike two-term recurrence which relies on implicit A-conjugacy,
        this explicitly enforces d_k^T A d_j ≈ 0 for all j < k (or a
        truncated window). This is necessary when the preconditioner
        changes between iterations.

    References:
        - Notay, Y. (2000). Flexible Conjugate Gradients.
    """

    def __init__(self, orthogonalization: OrthogonalizationStrategy):
        """Initialize orthogonalization direction strategy.

        Args:
            orthogonalization: OrthogonalizationStrategy instance
                (e.g., PeriodicRestartOrthogonalization, TruncatedGramSchmidt)
        """
        self.orthogonalization = orthogonalization

    @property
    def required_history_size(self) -> int | None:
        """Return window size from wrapped orthogonalization strategy."""
        return self.orthogonalization.window_size

    def compute_direction(
        self,
        w: NDArray,
        state: CGState,
    ) -> NDArray:
        """Compute FCG direction using orthogonalization.

        Formula (Notay 2000):
            d_k = w_k - Σ_j [(w_k, A d_j) / (d_j, A d_j)] d_j

        Args:
            w: Preconditioned residual w_k
            state: Current CGState with direction_history

        Returns:
            Orthogonalized search direction d_k

        Theory:
            Orthogonalization maintains approximate A-conjugacy by explicitly
            removing components of w_k that are not A-conjugate to previous
            directions. This ensures the search directions remain linearly
            independent and span the Krylov subspace.
        """
        history = state.direction_history

        # Orthogonalize using the configured strategy
        d_orthog, report = self.orthogonalization.orthogonalize(
            vector=w,
            d_vectors=history.d_vectors,
            q_vectors=history.q_vectors,
        )

        # Log breakdown warning if detected
        if report.breakdown:
            logger.warning(
                f"Orthogonalization breakdown detected at iteration {state.iteration}. "
                f"Orthogonalized vector norm is near zero."
            )

        return d_orthog


class CompositeDirectionStrategy(DirectionStrategy):
    """Composite strategy combining two-term recurrence with reorthogonalization.

    This implements PCG with optional reorthogonalization, which is the
    combination of:
        1. Phase 1: Standard two-term recurrence (PCG)
        2. Phase 2: Optional reorthogonalization (corrects rounding errors)

    Formula:
        d_k = w_k + β_k * d_{k-1}                                    # Phase 1
        d_k := d_k - Σ_j [(d_k, A d_j) / (d_j, A d_j)] d_j          # Phase 2 (optional)

    This is additive: reorthogonalization happens AFTER the standard PCG
    two-term recurrence, preserving PCG's structure while correcting errors.

    Design:
        This strategy is STATELESS when wrapping a stateless base_strategy.
        All required data is stored in CGState and passed through the call chain.
        Strategies are reusable across multiple solve() calls.

    Attributes:
        base_strategy: Base direction strategy (usually TwoTermRecurrenceStrategy)
        reorthog_strategy: Optional reorthogonalization strategy

    Use Cases:
        - Ill-conditioned problems where rounding errors degrade conjugacy
        - Long iteration sequences
        - When numerical accuracy is critical

    Theory:
        The two-term recurrence maintains A-conjugacy in exact arithmetic,
        but finite precision can degrade conjugacy over many iterations.
        Reorthogonalization explicitly corrects these accumulated errors.

    Example (Reusable):
        >>> base = TwoTermRecurrenceStrategy()
        >>> reorthog = TruncatedGramSchmidt(window_size=20)
        >>> strategy = CompositeDirectionStrategy(base, reorthog)
        >>>
        >>> # Strategy is reusable across multiple solves
        >>> x1, r1 = solver1.solve(A1, b1, direction_strategy=strategy)
        >>> x2, r2 = solver2.solve(A2, b2, direction_strategy=strategy)  # OK!

    References:
        - Saad (2003), Section 9.5: Reorthogonalization in CG.
    """

    def __init__(
        self,
        base_strategy: DirectionStrategy,
        reorthog_strategy: OrthogonalizationStrategy | None = None,
    ):
        """Initialize composite direction strategy.

        Args:
            base_strategy: Base direction computation strategy
                (typically TwoTermRecurrenceStrategy)
            reorthog_strategy: Optional orthogonalization strategy for
                correcting rounding errors (None = disabled)
        """
        self.base_strategy = base_strategy
        self.reorthog_strategy = reorthog_strategy

    @property
    def required_history_size(self) -> int | None:
        """Return history size required by reorthogonalization strategy."""
        if self.reorthog_strategy is None:
            return 0
        return self.reorthog_strategy.window_size

    def compute_direction(
        self,
        w: NDArray,
        state: CGState,
    ) -> NDArray:
        """Compute direction using base strategy + optional reorthogonalization.

        Args:
            w: Preconditioned residual w_k
            state: Current CGState

        Returns:
            Search direction d_k (with optional reorthogonalization)

        Theory:
            Phase 1 uses the base strategy (e.g., two-term recurrence) to
            compute the direction. Phase 2 optionally reorthogonalizes to
            correct accumulated rounding errors that degrade A-conjugacy.
        """
        # Phase 1: Base direction computation
        d = self.base_strategy.compute_direction(w, state)

        # Phase 2: Optional reorthogonalization
        if self.reorthog_strategy is None:
            return d

        # Reorthogonalize against previous directions
        d_reorthog, report = self.reorthog_strategy.orthogonalize(
            vector=d,
            d_vectors=state.direction_history.d_vectors,
            q_vectors=state.direction_history.q_vectors,
        )

        # Log breakdown warning if detected
        if report.breakdown:
            logger.warning(
                f"Reorthogonalization breakdown detected at iteration {state.iteration}. "
                f"Reorthogonalized vector norm is near zero."
            )

        return d_reorthog
