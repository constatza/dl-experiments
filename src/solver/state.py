"""Iteration state tracking for Flexible PCG with non-SPD preconditioners.

This module defines the IterationState dataclass that tracks all numerical
information throughout the PCG iteration. The state is immutable and updated
using dataclasses.replace(), enabling clean tracking of algorithm progress.

Theory:
    The IterationState maintains cumulative diagnostic information:
    - Convergence status and history
    - Restart tracking (for non-SPD preconditioner recovery)
    - Numerical diagnostics (breakdown/divergence detection)
    - Helper/residual management event history

    Each iteration returns an updated state via dataclasses.replace(),
    preserving functional programming principles while maintaining full
    diagnostic information for analysis and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IterationState:
    """Tracks numerical state and diagnostics across PCG iterations.

    This dataclass maintains complete information about the algorithm's progress,
    enabling robust recovery from preconditioner non-symmetry and detecting
    numerical breakdowns.

    Attributes:
        converged (bool): Whether convergence criterion has been satisfied.
            Convergence uses: ||r_k||_2 <= max(rtol * ||b||_2, atol)
            This is checked at each iteration regardless of restart status.

        residual_history (list[float]): Running list of ||r_k||_2 norms.
            Used to track residual reduction and detect divergence.
            Each iteration appends the current ||r_k||_2 after convergence check.

        restart (bool): Whether a restart occurred at current iteration.
            Restart triggers:
            1. Curvature negative or too small: d_k <= eps_curv * ||p_k||_2^2
            2. Beta out of bounds: β_k < 0 or β_k > beta_max
            3. Breakdown in preconditioner or step_length: NaN/Inf detected
            4. Divergence detected: ||r_k||_2 > gamma_div * ||b||_2

            On restart, p_k is replaced with z_k (steepest descent step).

        breakdown (bool): Whether a numerical breakdown has been detected.
            Breakdown indicators:
            - NaN or Inf in preconditioned residual z_k
            - NaN or Inf in step length calculation
            - Division by denominator < eps_breakdown in step_length
            - Preconditioner output with wrong shape/dtype

            Once triggered, breakdown persists. The solver should terminate.

        divergence (bool): Whether divergence has been detected.
            Divergence condition: ||r_k||_2 > gamma_div * ||b||_2
            where gamma_div is typically 1e10.

            If detected, residual_management() will recompute true residual:
            r_true = b - A*x_k and reinitialize the search direction.

        num_restarts (int): Count of restart events.
            Each restart increments this counter. Used for statistics/analysis.

        num_residual_replacements (int): Count of true residual recomputation events.
            Each call to residual_management that replaces r, z, p increments this.
            These occur approximately every m_replacement iterations.

        iterations (int): Count of iterations completed.
            Incremented after each iteration's 10 steps complete.
            Used to: check convergence flags, count restarts per N iterations,
            control periodic residual recomputation.

    Theory:
        The state follows immutable update patterns:
            S_{k+1} = dataclasses.replace(S_k, converged=True, ...)

        This preserves a functional programming style while maintaining
        diagnostic history. The full residual_history enables post-hoc
        analysis of convergence behavior, restart patterns, and numerical
        stability.

        Restart and breakdown are separate concerns:
        - Restart: Algorithm recovery for non-SPD preconditioners. Benign.
        - Breakdown: Numerical overflow/NaN. Terminal.
    """

    converged: bool = False
    """Whether convergence criterion satisfied: ||r||_2 <= max(rtol*||b||, atol)."""

    residual_history: list[float] = field(default_factory=list)
    """Running log of ||r_k||_2 at each iteration. Enables divergence detection."""

    restart: bool = False
    """Whether current iteration triggered a restart. Benign recovery mechanism."""

    breakdown: bool = False
    """Whether numerical breakdown occurred (NaN/Inf). Terminal condition."""

    divergence: bool = False
    """Whether ||r_k||_2 > gamma_div * ||b||_2. Triggers residual recomputation."""

    num_restarts: int = 0
    """Count of restart events. Updated when restart=True."""

    num_residual_replacements: int = 0
    """Count of true residual recomputation events. Tracks periodic maintenance."""

    iterations: int = 0
    """Count of completed iterations. Used for frequency-based decisions."""

    def __post_init__(self) -> None:
        """Validate state invariants.

        Raises:
            ValueError: If state has inconsistent flags.
        """
        # If breakdown is True, algorithm should terminate immediately
        if self.breakdown and self.iterations > 0:
            pass  # Breakdown can occur at any iteration

        # If converged, should check this after iteration
        if self.converged and not self.residual_history:
            raise ValueError("Cannot be converged with empty residual history")


@dataclass
class DirectionHistory:
    """History buffer for direction strategies that require previous vectors (e.g., FCG).

    This dataclass maintains buffers of search directions and matrix products
    needed for algorithms like Flexible Conjugate Gradient (FCG) that orthogonalize
    against a window of previous directions.

    Theory:
        FCG computes search directions using Gram-Schmidt orthogonalization:
            p_i = z_i - Σ_{k=i-m}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

        This requires storing the last m directions (p_k) and products (q_k = A p_k).

        Memory: O(m * n) where m is history length, n is problem size.
        Typical m = 10-50, which is manageable for most problems.

    Attributes:
        p_vectors: List of previous search directions [p_{i-m}, ..., p_{i-1}].
            Maximum length controlled by m_max parameter in direction strategy.

        q_vectors: List of previous matrix-vector products [q_{i-m}, ..., q_{i-1}].
            Each q_k = A @ p_k. Used in orthogonalization coefficients.

        r_old: Previous residual r_{i-1} (optional).
            Used by some strategies for beta computation.

        z_old: Previous preconditioned residual z_{i-1} (optional).
            Used by some strategies for beta computation.

    Usage:
        history = DirectionHistory()
        history = history.update(p, q, r, z, m_max=10)
    """

    p_vectors: list = field(default_factory=list)
    """Search directions from previous iterations."""

    q_vectors: list = field(default_factory=list)
    """Matrix-vector products q_k = A @ p_k from previous iterations."""

    r_old: None | object = None
    """Previous residual (optional, for beta computation)."""

    z_old: None | object = None
    """Previous preconditioned residual (optional, for beta computation)."""

    total_updates: int = 0
    """Total number of stored directions ever added (monotonic, not truncated)."""

    def update(
        self,
        p: object,
        q: object,
        r: object | None = None,
        z: object | None = None,
        *,
        m_max: int = 10,
    ) -> DirectionHistory:
        """Return new history with updated vectors, maintaining max length.

        This method follows immutable update pattern, returning a new DirectionHistory
        instance rather than modifying in place.

        Args:
            p: New search direction to add.
            q: New matrix-vector product (A @ p) to add.
            r: Current residual (becomes r_old in next iteration).
            z: Current preconditioned residual (becomes z_old in next iteration).
            m_max: Maximum history length. Oldest vectors dropped if exceeded.

        Returns:
            New DirectionHistory with updated buffers.

        Theory:
            Truncation strategy maintains only last m_max vectors to:
            1. Bound memory usage at O(m_max * n)
            2. Limit orthogonalization cost at O(m_max * n) per iteration
            3. Prevent numerical error accumulation in very long orthogonalization chains

        Example:
            history = DirectionHistory()
            for i in range(iterations):
                p, q = compute_direction(...)
                history = history.update(p, q, r, z, m_max=10)
        """
        import copy

        new_p = self.p_vectors + [copy.copy(p)]
        new_q = self.q_vectors + [copy.copy(q)]
        new_total = self.total_updates + 1

        # Truncate to m_max
        if len(new_p) > m_max:
            new_p = new_p[-m_max:]
        if len(new_q) > m_max:
            new_q = new_q[-m_max:]

        return DirectionHistory(
            p_vectors=new_p,
            q_vectors=new_q,
            r_old=copy.copy(r) if r is not None else None,
            z_old=copy.copy(z) if z is not None else None,
            total_updates=new_total,
        )
