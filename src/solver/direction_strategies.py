"""Direction update strategies for Flexible PCG algorithm.

This module defines the DirectionStrategy abstraction and its implementations
for computing conjugate search directions in Flexible PCG. Different strategies
balance memory, computational cost, and convergence properties.

Mathematical Background:
    In Conjugate Gradient methods, the search direction p_k is computed to
    accelerate convergence beyond steepest descent. The key property is
    A-conjugacy (A-orthogonality):
        p_i^T A p_j = 0 for i ≠ j

    For SPD systems with SPD preconditioners, this is maintained exactly by
    the classical formula:
        p_{k+1} = z_{k+1} + β_k p_k
        β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)

    For non-SPD preconditioners M_k, A-conjugacy is not guaranteed. We use
    a residual-only beta formula instead:
        β_k = ||r_{k+1}||^2 / ||r_k||^2

    This formula:
    1. Does NOT require M_k to be symmetric
    2. Does NOT guarantee A-conjugacy
    3. DOES maintain descent direction properties
    4. DOES enable convergence with restarts

Theory:
    FlexibleCGDirection implements the classical two-term recurrence:
        p_{k+1} = z_{k+1} + β_k p_k

    where β_k is computed from residual norms only. This is the minimal-memory
    approach (O(n) storage) suitable for large problems.

    TruncatedOrthogonalDirection (not yet implemented) uses explicit Gram-Schmidt
    orthogonalization against a window of previous directions:
        p_i = z_i - Σ_{k=i-m}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

    This requires O(m * n) storage but provides better numerical stability
    for severely non-SPD preconditioners.

References:
    - Flexible CG: Notay (2000), "Flexible Conjugate Gradients"
    - Residual-only beta: Golub & van der Vorst (2000)
    - FCG orthogonalization: Saad (1993), "A Flexible Inner-Outer Preconditioned GMRES"
    - Non-SPD preconditioners: Saad (2003), "Iterative Methods for Sparse Linear Systems"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from dataclasses import replace
from scipy.linalg import norm

from ..constants import DEFAULT_BETA_MAX, DEFAULT_M_MAX, FCG_ORTHOG_EPSILON
from .helpers import beta_update, direction_update

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .state import DirectionHistory, IterationState


class DirectionStrategy(ABC):
    """Abstract base class for direction update strategies.

    A DirectionStrategy computes the search direction p_k in Flexible PCG
    using the preconditioned residual z_k and information from previous
    iterations (stored in DirectionHistory).

    Theory:
        The search direction p_k determines where the algorithm steps in
        the solution space. Different strategies make different trade-offs:

        1. Two-term recurrence (FlexibleCGDirection):
           - Memory: O(n) (only current vectors)
           - Cost: O(n) per iteration
           - Stability: Moderate (depends on restart frequency)

        2. Truncated orthogonalization (TruncatedOrthogonalDirection):
           - Memory: O(m * n) (history of m vectors)
           - Cost: O(m * n) per iteration
           - Stability: High (explicit orthogonalization)

    Methods:
        compute_direction: Compute p_k from z_k and history.

    Mathematical Requirements:
        A valid direction strategy must ensure:
        1. p_k is a descent direction: r_k^T A p_k > 0 (for SPD A)
        2. p_k does not cause overflow: ||p_k|| is bounded
        3. p_k remains linearly independent: not parallel to previous directions
    """

    @abstractmethod
    def compute_direction(
        self,
        r: NDArray,
        z: NDArray,
        state: IterationState,
        history: DirectionHistory,
        r_prev: NDArray | None = None,
    ) -> tuple[NDArray, IterationState]:
        """Compute search direction from residual, preconditioned residual, and history.

        This is the core method that defines how each strategy computes p_k.

        Args:
            r: Current residual r_k = b - A*x_k, shape (n,).
                Used for beta computation in residual-only formula.
            z: Current preconditioned residual z_k = M^{-1} r_k, shape (n,).
                This is the base steepest descent direction.
            state: Current iteration state. Used to check restart flag and
                update convergence/breakdown status.
            history: Direction history containing previous p_k, q_k, r_k, z_k
                vectors. May be empty on first iteration.
            r_prev: Previous residual r_{k-1} from the last iteration, shape (n,).
                Used for beta computation (optional, None on first iteration).

        Returns:
            Tuple of (p_new, updated_state):
            - p_new (ndarray): New search direction p_k, shape (n,)
            - updated_state (IterationState): Updated state with restart/breakdown flags

        Theory:
            The compute_direction method encapsulates the mathematical formula
            for constructing p_k. Different strategies implement different
            formulas:

            FlexibleCGDirection: p_k = z_k + β_{k-1} p_{k-1}
            TruncatedOrthogonalDirection: p_k = z_k - Σ_i [(z_k, q_i)/(p_i, q_i)] p_i

        Raises:
            NotImplementedError: If called on base class (must be overridden).
        """
        raise NotImplementedError


class FlexibleCGDirection(DirectionStrategy):
    """Classical flexible CG direction update with residual-only beta.

    This strategy implements the standard two-term recurrence:
        p_{k+1} = z_{k+1} + β_k p_k

    using the residual-only beta formula:
        β_k = ||r_{k+1}||^2 / ||r_k||^2

    Mathematical Justification:
        The two-term recurrence minimizes memory (O(n) storage) while maintaining
        descent properties. Unlike standard CG's beta formula:
            β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)

        the residual-only formula does NOT require M_k to be SPD. This allows
        Flexible PCG to work with neural network preconditioners, which are:
        1. Time-varying (M_k depends on iteration k)
        2. Non-symmetric (M_k^T ≠ M_k in general)
        3. Non-SPD (negative eigenvalues possible)

    Restart Mechanism:
        When β_k exceeds beta_max or becomes negative, the algorithm restarts:
            p_{k+1} = z_{k+1}  (steepest descent)

        This recovers from loss of conjugacy and prevents unbounded growth.

    Numerical Considerations:
        1. Beta clamping: β_k is clamped to [0, beta_max] range
        2. Restart on β_k < 0: Signals numerical error or non-SPD behavior
        3. Restart on β_k > beta_max: Signals loss of conjugacy
        4. Safe default: beta_max = 1e10 (typical value)

    Attributes:
        beta_max: Maximum allowed beta before restart. Default: 1e10.
        eps_breakdown: Minimum denominator for safe division. Default: 1e-14.

    Theory:
        Why use residual-only beta?
        - Standard CG beta requires (r_k^T z_k), which assumes M_k is SPD
        - Non-SPD M_k can cause (r_k^T z_k) to be negative or zero
        - Residual-only beta is always non-negative and well-defined

        Why restart on large beta?
        - β_k > 1 means residual increased (||r_{k+1}|| > ||r_k||)
        - β_k >> 1 indicates severe loss of conjugacy
        - Steepest descent (β = 0) guarantees descent for SPD systems

    References:
        - Notay (2000): "Flexible Conjugate Gradients"
        - Algorithm.md, Steps 8-9: beta_update and direction_update
    """

    def __init__(
        self,
        beta_max: float = DEFAULT_BETA_MAX,
        eps_breakdown: float = 1e-14,
    ) -> None:
        """Initialize FlexibleCGDirection strategy.

        Args:
            beta_max: Maximum allowed beta before restart. Typical: 1e10.
                Values larger than this indicate severe loss of conjugacy.
            eps_breakdown: Minimum denominator for safe division in beta formula.
                Typical: 1e-14 (two orders of magnitude above machine epsilon).

        Theory:
            The beta_max parameter controls restart frequency:
            - Smaller beta_max: More frequent restarts, slower convergence
            - Larger beta_max: Fewer restarts, risk of overflow
            - Default 1e10: Balance between stability and efficiency

            The eps_breakdown parameter prevents division by zero:
            - If ||r_k||^2 < eps_breakdown, beta computation is unsafe
            - This signals convergence (residual near zero) or breakdown
            - Default 1e-14: Safe for double precision (1e-16 machine epsilon)
        """
        self.beta_max = beta_max
        self.eps_breakdown = eps_breakdown

    def compute_direction(
        self,
        r: NDArray,
        z: NDArray,
        state: IterationState,
        history: DirectionHistory,
        r_prev: NDArray | None = None,
    ) -> tuple[NDArray, IterationState]:
        """Compute direction using two-term recurrence with residual-only beta.

        This method implements the classical flexible CG update:
            1. Compute β_k = ||r_{k+1}||^2 / ||r_k||^2
            2. Update p_{k+1} = z_{k+1} + β_k p_k

        On restart (triggered by beta_update if β_k out of bounds):
            p_{k+1} = z_{k+1}  (steepest descent)

        Args:
            r: Current residual r_k = b - A*x_k, shape (n,).
                Used for computing beta (residual-only formula).
            z: Current preconditioned residual z_k = M^{-1} r_k, shape (n,).
                This is the steepest descent direction in the M-norm.
            state: Current iteration state. Checked for restart flag.
            history: Direction history containing:
                - r_old: Previous residual r_{k-1} (needed for beta)
                - z_old: Previous preconditioned residual z_{k-1} (unused here)
                - p_vectors: Previous search directions (only p_{k-1} needed)

        Returns:
            Tuple of (p_new, updated_state):
            - p_new (ndarray): New search direction p_k, shape (n,)
            - updated_state (IterationState): With restart flag set if β_k out of bounds

        Theory:
            The two-term recurrence maintains a balance between:
            1. Steepest descent (β = 0): Stable but slow convergence
            2. Conjugate gradient (β from formula): Fast but requires conjugacy

            When β_k is bounded, we get fast convergence. When β_k exceeds bounds,
            we restart to steepest descent to recover stability.

        Mathematical Steps:
            1. Extract r_old from history (previous iteration's residual)
            2. Compute β_k = ||r_k||^2 / ||r_{k-1}||^2 via beta_update()
            3. Check if β_k ∈ [0, beta_max]; if not, set restart=True and β_k=0
            4. Extract p_old from history (previous iteration's direction)
            5. Compute p_k = z_k + β_k p_{k-1} via direction_update()
            6. If restart=True, override p_k = z_k (steepest descent)

        Implementation Note:
            This implementation uses the helper functions beta_update() and
            direction_update() from helpers.py to maintain consistency with
            the existing flexible_pcg implementation.
        """
        # First iteration: no history available, use steepest descent
        if r_prev is None or not history.p_vectors:
            # Initialize with steepest descent direction
            return z.copy(), state

        # Get previous direction from history
        p_old = history.p_vectors[-1]  # Most recent direction

        # Compute beta using residual-only formula
        # beta_update handles restart logic if beta is out of bounds
        # Use r (current residual r_{k+1}) and r_prev (previous residual r_k)
        beta, state = beta_update(r, r_prev, self.beta_max, state)

        # Update direction using beta and previous direction
        # direction_update handles restart case (state.restart=True)
        p_new, state = direction_update(z, p_old, beta, state)

        return p_new, state


class TruncatedOrthogonalDirection(DirectionStrategy):
    """FCG direction update with truncated Gram-Schmidt orthogonalization.

    This strategy implements the Flexible Conjugate Gradient (FCG) algorithm
    with explicit orthogonalization against a sliding window of previous
    search directions:

        p_i = z_i - Σ_{k=i-m_i}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

    where:
        - m_i = min(i, max(1, mod(i, m_max+1))) is the active history length
        - z_i = M^{-1} r_i is the preconditioned residual
        - q_k = A p_k is the matrix-vector product
        - (·,·) denotes the Euclidean inner product

    Mathematical Theory:

    A-Conjugacy vs Euclidean Orthogonality:
        In standard CG with SPD preconditioner M, search directions satisfy
        A-conjugacy (A-orthogonality):
            p_i^T A p_j = 0 for i ≠ j

        This is maintained implicitly by the two-term recurrence. However,
        with non-SPD preconditioners M_k, A-conjugacy can be lost.

        FCG restores approximate A-conjugacy by explicit Gram-Schmidt
        orthogonalization. The key insight is that we orthogonalize z_i
        against q_k (not against p_k):

            coeff_k = (z_i, q_k) / (p_k, q_k) = (z_i, A p_k) / (p_k^T A p_k)

        This is designed to enforce:
            p_i^T A p_k ≈ 0

        Why? Because after orthogonalization:
            p_i^T A p_k = (z_i - Σ coeff_j p_j)^T A p_k
                        ≈ z_i^T A p_k - coeff_k (p_k^T A p_k)
                        = (z_i, q_k) - [(z_i, q_k)/(p_k, q_k)] · (p_k, q_k)
                        = 0

        This is A-conjugacy, not Euclidean orthogonality. If we used (z_i, p_k)
        in the numerator, we would get Euclidean orthogonality instead, which
        does NOT accelerate convergence for linear systems.

    Periodic Reset Mechanism:
        The formula m_i = min(i, max(1, mod(i, m_max+1))) implements a
        periodic reset of the history length:

        - For i < m_max: m_i = i (accumulate all history)
        - For i = m_max: m_i = m_max (full window)
        - For i = m_max+1: m_i = 1 (reset to z_i only)
        - For i = m_max+2: m_i = 2 (rebuild history)
        - ...continues with period m_max+1

        This periodic reset:
        1. Prevents accumulation of rounding errors over long runs
        2. Bounds memory usage at O(m_max * n)
        3. Provides automatic recovery from loss of orthogonality

    Numerical Safety:

    The implementation includes comprehensive safety checks:

    1. Small Denominator Protection:
        If (p_k, q_k) < eps_orthog, skip the orthogonalization term for
        that k. This prevents division by near-zero values which would
        amplify rounding errors.

    2. NaN/Inf Detection:
        Check numerator, denominator, and coefficient for NaN/Inf at each
        orthogonalization step. Any invalid values trigger term skip.

    3. Breakdown Detection:
        After orthogonalization, check if ||p_i|| << ||z_i||. If the
        orthogonalized direction becomes too small, restart to steepest
        descent direction p_i = z_i and set state.restart=True.

        Why? Small ||p_i|| indicates near-linear dependence with previous
        directions, which causes:
        - Step length α → ∞ (curvature d_k → 0)
        - Numerical overflow in x update
        - Loss of descent property

    Attributes:
        m_max: Maximum history length for orthogonalization. Default: 10.
            Larger m_max → better orthogonality, higher cost.
            Typical range: 5-50.

        eps_orthog: Threshold for near-zero denominators in Gram-Schmidt.
            Default: 1e-14 (two orders above machine epsilon).
            Terms with (p_k, q_k) < eps_orthog are skipped.

    Performance Characteristics:
        - Memory: O(m_max * n) to store p_k and q_k vectors
        - Cost per iteration: O(m_i * n) for orthogonalization
            where m_i ≤ m_max
        - Typical m_max = 10 → 10x cost vs two-term recurrence

    When to Use This Strategy:
        Use TruncatedOrthogonalDirection when:
        1. Preconditioner M_k is severely non-SPD (e.g., neural networks)
        2. Two-term recurrence shows frequent restarts
        3. Memory allows O(m_max * n) storage
        4. Convergence improvements justify higher cost

        Use FlexibleCGDirection when:
        1. Preconditioner M_k is nearly SPD or symmetric
        2. Memory is constrained (large n)
        3. Cost per iteration is critical
        4. Occasional restarts are acceptable

    References:
        - Notay (2000): "Flexible Conjugate Gradients"
        - Saad (1993): "A Flexible Inner-Outer Preconditioned GMRES"
        - Axelsson & Vassilevski (1991): "A Black Box Generalized CG Solver"
    """

    def __init__(
        self,
        m_max: int = DEFAULT_M_MAX,
        eps_orthog: float = FCG_ORTHOG_EPSILON,
    ) -> None:
        """Initialize TruncatedOrthogonalDirection strategy.

        Args:
            m_max: Maximum history length for orthogonalization. Default: 10.
                Controls memory usage (O(m_max * n)) and orthogonalization
                cost (O(m_max * n) per iteration).

                Larger values:
                - Better numerical stability
                - Stronger orthogonality enforcement
                - Higher memory and computational cost

                Typical values:
                - m_max = 5: Minimal overhead, moderate stability
                - m_max = 10: Default, good balance
                - m_max = 50: High stability, higher cost

            eps_orthog: Threshold for near-zero denominators in Gram-Schmidt.
                Default: 1e-14 (FCG_ORTHOG_EPSILON).

                Terms with (p_k, q_k) < eps_orthog are skipped to prevent
                division by near-zero values. This is two orders of magnitude
                above machine epsilon (~1e-16 for float64), providing a safety
                buffer against rounding error amplification.

        Theory:
            The eps_orthog parameter trades off:
            - Too small: Risk of division by near-zero, numerical instability
            - Too large: Skip too many terms, loss of orthogonality

            The default 1e-14 is calibrated for double precision arithmetic
            and typical matrix condition numbers (κ < 1e10).

            The m_max parameter determines the orthogonalization window size.
            Saad (1993) recommends m_max = 5-20 for most applications. Larger
            values help with severely ill-conditioned or non-SPD preconditioners
            but increase cost proportionally.
        """
        self.m_max = m_max
        self.eps_orthog = eps_orthog

    def compute_direction(
        self,
        r: NDArray,
        z: NDArray,
        state: IterationState,
        history: DirectionHistory,
        r_prev: NDArray | None = None,
    ) -> tuple[NDArray, IterationState]:
        """Compute direction using truncated Gram-Schmidt orthogonalization.

        This method implements the FCG algorithm with periodic reset:

        1. Compute iteration number i from history length
        2. Compute active history length m_i = min(i, max(1, mod(i, m_max+1)))
        3. Initialize p_i = z_i (steepest descent)
        4. For each k in [i-m_i, i-1]:
            a. Compute numerator: (z_i, q_k)
            b. Compute denominator: (p_k, q_k)
            c. Check for NaN/Inf in numerator, denominator
            d. Skip if denominator < eps_orthog
            e. Compute coefficient: coeff = numerator / denominator
            f. Check for NaN/Inf in coefficient
            g. Orthogonalize: p_i = p_i - coeff * p_k
        5. Check breakdown: if ||p_i|| << ||z_i||, restart to z_i

        Args:
            r: Current residual r_i = b - A*x_i, shape (n,).
                Not used in orthogonalization (FCG uses z_i and q_k instead).
            z: Current preconditioned residual z_i = M^{-1} r_i, shape (n,).
                This is the starting point for orthogonalization.
            state: Current iteration state. Updated if breakdown occurs.
            history: Direction history containing:
                - p_vectors: Previous search directions [p_{i-m}, ..., p_{i-1}]
                - q_vectors: Previous products [q_{i-m}, ..., q_{i-1}]
                - r_old, z_old: Not used in FCG orthogonalization

        Returns:
            Tuple of (p_new, updated_state):
            - p_new (ndarray): New search direction p_i, shape (n,)
                After orthogonalization or restart to z_i if breakdown
            - updated_state (IterationState): With restart=True if breakdown detected

        Theory:
            First iteration (i=0):
            No history available, return p_0 = z_0 (steepest descent).

            Subsequent iterations (i>0):
            1. Determine m_i from periodic reset formula
            2. Orthogonalize z_i against last m_i directions
            3. Check for numerical breakdown
            4. Restart to z_i if breakdown detected

            Periodic Reset Example (m_max=3):
            - i=0: m_i=0, p_0 = z_0 (no history)
            - i=1: m_i=1, p_1 = z_1 - coeff_0 * p_0
            - i=2: m_i=2, p_2 = z_2 - Σ_{k=0}^{1} coeff_k * p_k
            - i=3: m_i=3, p_3 = z_3 - Σ_{k=0}^{2} coeff_k * p_k
            - i=4: m_i=1, p_4 = z_4 - coeff_3 * p_3 (RESET: forget p_0, p_1, p_2)
            - i=5: m_i=2, p_5 = z_5 - Σ_{k=4}^{5} coeff_k * p_k
            - ...repeats with period m_max+1 = 4

        Breakdown Detection:
            After orthogonalization, we check:
                ||p_i||_2 / ||z_i||_2 < eps_orthog

            If true, p_i is nearly zero despite z_i being non-zero. This indicates:
            - Near-linear dependence with previous directions
            - Accumulated rounding error in orthogonalization
            - Loss of numerical rank in search space

            Recovery: Set p_i = z_i (restart) and state.restart=True.
            This gives the algorithm a fresh start from steepest descent.

        Safety Checks:
            The implementation performs checks at each orthogonalization step:
            1. Check np.isfinite(numerator): Detect NaN/Inf in (z_i, q_k)
            2. Check np.isfinite(denominator): Detect NaN/Inf in (p_k, q_k)
            3. Check denominator >= eps_orthog: Prevent division by near-zero
            4. Check np.isfinite(coeff): Detect overflow in division
            5. After loop: Check ||p_i|| vs ||z_i||: Detect breakdown

            Any failure skips the term (checks 1-4) or triggers restart (check 5).

        Implementation Notes:
            - We use np.dot() for inner products (faster than linalg.norm)
            - We check NaN/Inf before division to prevent propagation
            - We copy z to initialize p_i to avoid modifying input
            - We use dataclasses.replace() to update state immutably
        """
        # First iteration: no history available, use steepest descent
        if not history.p_vectors or not history.q_vectors:
            return z.copy(), state

        # Compute iteration index from total updates (not truncated length)
        i = history.total_updates

        # Compute active history length m_i with periodic reset
        # Formula: m_i = min(i, max(1, mod(i, m_max+1)))
        m_i = min(i, max(1, i % (self.m_max + 1)))
        # Only use as many vectors as we kept in the truncated buffer
        m_i = min(m_i, len(history.p_vectors))

        # Initialize with steepest descent direction
        p_i = z.copy()

        # Orthogonalize against last m_i directions
        # Range: k ∈ [i-m_i, i-1]
        start_idx = len(history.p_vectors) - m_i
        for idx in range(start_idx, len(history.p_vectors)):
            # Get p_k and q_k from history
            # history vectors are indexed 0-based: [p_0, ..., p_{i-1}]
            p_k = history.p_vectors[idx]
            q_k = history.q_vectors[idx]

            # Compute numerator: (z_i, q_k) = z_i^T A p_k
            numerator = np.dot(z, q_k)

            # Check for NaN/Inf in numerator
            if not np.isfinite(numerator):
                # Skip this term if numerator is invalid
                continue

            # Compute denominator: (p_k, q_k) = p_k^T A p_k
            denominator = np.dot(p_k, q_k)

            # Check for NaN/Inf in denominator
            if not np.isfinite(denominator):
                # Skip this term if denominator is invalid
                continue

            # Check for near-zero denominator
            if denominator < self.eps_orthog:
                # Skip this term to prevent division by near-zero
                continue

            # Compute orthogonalization coefficient
            coeff = numerator / denominator

            # Check for NaN/Inf in coefficient
            if not np.isfinite(coeff):
                # Skip this term if coefficient is invalid
                continue

            # Orthogonalize: p_i = p_i - coeff * p_k
            p_i = p_i - coeff * p_k

        # Check for breakdown: ||p_i|| too small compared to ||z_i||
        # This indicates near-linear dependence or numerical error accumulation
        p_i_norm = norm(p_i)
        z_norm = norm(z)

        # Avoid division by zero in ratio check
        if z_norm < self.eps_orthog:
            # z is nearly zero; this shouldn't happen (would trigger convergence)
            # but handle it safely: return z as direction
            return z.copy(), state

        # Check if orthogonalized direction is too small
        # Threshold: ||p_i|| / ||z_i|| < eps_orthog
        if p_i_norm / z_norm < self.eps_orthog:
            # Breakdown detected: restart to steepest descent
            p_restart = z.copy()
            state_restart = replace(
                state,
                restart=True,
                num_restarts=state.num_restarts + 1,
            )
            return p_restart, state_restart

        # Normal case: return orthogonalized direction
        return p_i, state
