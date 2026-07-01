"""Scheduled preconditioner for iteration-based switching."""

from __future__ import annotations

from numpy.typing import NDArray

from ..base import BindableInputs, NonLinearPreconditioner, Preconditioner, PreconditionerContext


class ScheduledPreconditioner(NonLinearPreconditioner, BindableInputs):
    """Preconditioner that switches between primary and fallback by iteration.

    Use fallback before ``start_iter``, primary while the schedule is active,
    then fallback again after ``limit_iters`` if a limit is configured. This is
    useful for expensive preconditioners that should be delayed, bounded, or
    both.

    Args:
        primary: Main preconditioner to apply
        fallback: Fallback preconditioner (default: Identity)
        limit_iters: Iterations to apply primary after start_iter (None = unlimited)
        start_iter: Iteration at which primary becomes active

    Example:
        >>> # Use ILU from iterations 5 through 14, then Jacobi
        >>> from .ilu import ILUPreconditioner
        >>> from .jacobi import JacobiPreconditioner
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=ILUPreconditioner(A),
        ...     fallback=JacobiPreconditioner(A),
        ...     limit_iters=10,
        ...     start_iter=5,
        ... )
    """

    def __init__(
        self,
        primary: Preconditioner,
        fallback: Preconditioner | None = None,
        limit_iters: int | None = None,
        start_iter: int = 0,
    ) -> None:
        """Initialize scheduled preconditioner.

        Args:
            primary: Main preconditioner to apply
            fallback: Fallback preconditioner (default: Identity)
            limit_iters: Iterations to apply primary after start_iter
            start_iter: Iteration at which primary becomes active
        """
        if start_iter < 0:
            raise ValueError("start_iter must be non-negative")
        self._primary = primary
        self._fallback = fallback
        self._limit_iters = limit_iters
        self._start_iter = start_iter

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Aggregate extra input names from primary (and fallback if bindable).

        Returns:
            Union of extra_input_names from primary and fallback, de-duplicated.
        """
        names: list[str] = []
        if isinstance(self._primary, BindableInputs):
            names.extend(self._primary.extra_input_names)
        if isinstance(self._fallback, BindableInputs):
            names.extend(self._fallback.extra_input_names)
        return tuple(dict.fromkeys(names))

    def bind_inputs(self, **inputs: NDArray) -> None:
        """Propagate extra inputs to primary and fallback preconditioners.

        Args:
            **inputs: Named arrays to forward.
        """
        if isinstance(self._primary, BindableInputs):
            self._primary.bind_inputs(**inputs)
        if isinstance(self._fallback, BindableInputs):
            self._fallback.bind_inputs(**inputs)

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

        precond = self._primary if self._uses_primary(iteration) else self._fallback

        # Apply chosen preconditioner (pass context through for nested contextual preconditioners)
        return precond.apply(residual, context)

    def _uses_primary(self, iteration: int) -> bool:
        """Return whether primary is active for the given zero-based iteration."""
        if iteration < self._start_iter:
            return False
        if self._limit_iters is None:
            return True
        return iteration < self._start_iter + self._limit_iters
