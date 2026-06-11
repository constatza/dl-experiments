"""Scheduled preconditioner for iteration-based switching."""

from __future__ import annotations

from numpy.typing import NDArray

from ..base import BindableInputs, ContextualPreconditioner, Preconditioner, PreconditionerContext


class ScheduledPreconditioner(ContextualPreconditioner, BindableInputs):
    """Preconditioner that switches to fallback after iteration limit.

    Use primary preconditioner for first N iterations, then switch to fallback.
    This is useful for expensive preconditioners (e.g., neural networks) that
    provide most benefit in early iterations.

    Args:
        primary: Main preconditioner to apply
        fallback: Fallback preconditioner (default: Identity)
        limit_iters: Switch to fallback after this iteration (None = unlimited)

    Example:
        >>> # Use ILU for first 10 iterations, then Jacobi
        >>> from .ilu import ILUPreconditioner
        >>> from .jacobi import JacobiPreconditioner
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=ILUPreconditioner(A), fallback=JacobiPreconditioner(A), limit_iters=10
        ... )
    """

    def __init__(
        self,
        primary: Preconditioner,
        fallback: Preconditioner | None = None,
        limit_iters: int | None = None,
    ) -> None:
        """Initialize scheduled preconditioner.

        Args:
            primary: Main preconditioner to apply
            fallback: Fallback preconditioner (default: Identity)
            limit_iters: Switch to fallback after this iteration
        """
        self._primary = primary
        self._fallback = fallback
        self._limit_iters = limit_iters

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

        # Switch to fallback after iteration limit
        if self._limit_iters is not None and iteration >= self._limit_iters:
            precond = self._fallback
        else:
            precond = self._primary

        # Apply chosen preconditioner (pass context through for nested contextual preconditioners)
        return precond.apply(residual, context)
