"""Scheduled preconditioner for iteration-based switching."""

from __future__ import annotations

from numpy.typing import NDArray

from ..base import ContextualPreconditioner, Preconditioner, PreconditionerContext


class ScheduledPreconditioner(ContextualPreconditioner):
    """Preconditioner that switches based on iteration schedule.

    Enables strategies like:
    - Use expensive preconditioner for first N iterations
    - Apply preconditioner every K iterations
    - Switch to fallback after iteration limit

    Args:
        primary: Main preconditioner to apply
        fallback: Fallback preconditioner (default: Identity)
        limit_iters: Switch to fallback after this iteration
        apply_every: Apply primary every N iterations (1 = every iteration)

    Example:
        >>> # Use ILU for first 10 iterations, then Jacobi
        >>> from .ilu import ILUPreconditioner
        >>> from .jacobi import JacobiPreconditioner
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=ILUPreconditioner(A),
        ...     fallback=JacobiPreconditioner(A),
        ...     limit_iters=10
        ... )
        >>>
        >>> # Apply neural preconditioner every 5 iterations
        >>> from .neural import NeuralPreconditioner
        >>> from .identity import Identity
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=NeuralPreconditioner(ckpt),
        ...     fallback=Identity(),
        ...     apply_every=5
        ... )
    """

    def __init__(
        self,
        primary: Preconditioner,
        fallback: Preconditioner | None = None,
        limit_iters: int | None = None,
        apply_every: int = 1,
    ) -> None:
        """Initialize scheduled preconditioner.

        Args:
            primary: Main preconditioner to apply
            fallback: Fallback preconditioner (default: Identity)
            limit_iters: Switch to fallback after this iteration
            apply_every: Apply primary every N iterations (1 = every iteration)
        """
        self._primary = primary
        self._fallback = fallback
        self._limit_iters = limit_iters
        self._apply_every = apply_every

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply primary or fallback based on schedule.

        Args:
            residual: Current residual vector r_k
            context: Iteration state (required for scheduling)

        Returns:
            Preconditioned residual z_k

        Raises:
            ValueError: If context is None
        """
        if context is None:
            raise ValueError("ScheduledPreconditioner requires context for iteration tracking")

        # Lazy import to avoid circular dependency
        if self._fallback is None:
            from .identity import Identity
            self._fallback = Identity()

        iteration = context.iteration

        # Determine which preconditioner to use
        use_primary = True
        if self._limit_iters is not None and iteration >= self._limit_iters:
            use_primary = False
        if self._apply_every > 1 and iteration % self._apply_every != 0:
            use_primary = False

        precond = self._primary if use_primary else self._fallback

        # Apply chosen preconditioner (pass context through for nested contextual preconditioners)
        return precond.apply(residual, context)
