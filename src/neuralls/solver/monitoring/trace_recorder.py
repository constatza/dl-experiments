"""Type-safe trace recording for solver iteration logging.

This module provides orthogonal trace recording using EventType enum for
complete type safety. Replaces string-based event names with strongly-typed
enumeration.

Design Principles:
    - Single Responsibility: Only handles logging, not solver logic
    - Type Safety: EventType enum prevents typos and enables validation
    - Immutable Events: All values copied to prevent mutation
    - Performance: Separate storage for scalars vs vectors

Theory:
    Event sourcing captures all state changes as immutable events.
    This enables:
    1. Full iteration history reconstruction
    2. Post-hoc analysis and debugging
    3. Type-safe event access
    4. Separation of logging from algorithm logic

Usage:
    >>> from neuralls.solver.monitoring.events import EventType
    >>> logger = TraceRecorder()
    >>> logger.log(EventType.RESIDUAL, r0)
    >>> logger.log(EventType.RESIDUAL_NORM, 1.5)
    >>>
    >>> residuals = logger.get_vectors(EventType.RESIDUAL)
    >>> norms = logger.get_scalars(EventType.RESIDUAL_NORM)

References:
    - Event Sourcing: Martin Fowler (2005)
    - SOLID Principles: Single Responsibility, Open/Closed
    - PEP 484: Type Hints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .events import EventType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ScalarHistory(np.ndarray):
    """Scalar history that compares cleanly in assertions.

    Custom ndarray subclass that implements value-based equality using
    np.allclose() instead of element-wise comparison.
    """

    def __new__(cls, input_array: list[float] | np.ndarray) -> ScalarHistory:
        """Create ScalarHistory from list or array."""
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        return obj

    def __array_finalize__(self, obj: object) -> None:  # pragma: no cover
        """Finalize array (required by ndarray protocol)."""
        if obj is None:
            return

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        """Value-based equality using np.allclose().

        Args:
            other: Object to compare

        Returns:
            True if arrays are approximately equal
        """
        try:
            a = np.asarray(self, dtype=np.float64)
            b = np.asarray(other, dtype=np.float64)
        except Exception:
            return False
        return bool(np.allclose(a, b))

    def __bool__(self) -> bool:
        """Boolean conversion (size > 0).

        Avoids numpy's ambiguous truth value error.
        """
        return self.size > 0


class TraceRecorder:
    """Type-safe trace recorder using EventType enum.

    Each log() call creates an immutable copy of the value. Events are
    stored separately for scalars (list[float]) and vectors (list[NDArray])
    for memory efficiency.

    Design Pattern:
        Event Sourcing with strong typing via EventType enum

    Attributes:
        _scalar_events: Dictionary mapping EventType to list of scalar values
        _vector_events: Dictionary mapping EventType to list of vector arrays

    Mathematical Background:
        Krylov solvers track scalar and vector quantities at each iteration:
        - Scalars: iteration k, residual norm ||r_k||, step length α_k
        - Vectors: solution x_k, residual r_k, direction p_k

    Memory Usage:
        - Scalars: ~8 bytes per value
        - Vectors: ~8n bytes per vector (n = problem size)

    Type Safety:
        - EventType enum prevents typos (compile-time check)
        - Separate methods for scalars vs vectors (runtime check)
        - Type annotations enable Mypy/Pyright validation

    Examples:
        >>> from neuralls.solver.monitoring.events import EventType
        >>> logger = TraceRecorder()
        >>> logger.log(EventType.RESIDUAL, np.array([1.0, 2.0]))
        >>> logger.log(EventType.RESIDUAL_NORM, 2.5)
        >>>
        >>> residuals = logger.get_vectors(EventType.RESIDUAL)
        >>> norms = logger.get_scalars(EventType.RESIDUAL_NORM)
        >>> assert len(residuals) == 1
        >>> assert norms == [2.5]
    """

    def __init__(self) -> None:
        """Initialize empty trace buffers."""
        self._scalar_events: dict[EventType, list[float]] = {}
        self._vector_events: dict[EventType, list[NDArray]] = {}

    def log_scalar(self, event: EventType | str, value: float | int | bool) -> None:
        """Log scalar event with type validation.

        Args:
            event: EventType or string (must satisfy event.is_scalar)
            value: Scalar value (float, int, or bool)

        Raises:
            ValueError: If event is not a scalar type

        Theory:
            Booleans are converted to float (0.0 or 1.0) for consistency.
            This allows convergence flags to be stored as numeric values.

        Examples:
            >>> logger.log_scalar(EventType.ITERATION, 5)
            >>> logger.log_scalar("iteration", 5)  # String also works
            >>> logger.log_scalar(EventType.CONVERGED, True)
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if not event.is_scalar:
            msg = f"{event} is not a scalar event type"
            raise ValueError(msg)

        # Convert bool to float (0.0 or 1.0) for numeric storage
        float_val = float(value)

        if event not in self._scalar_events:
            self._scalar_events[event] = []
        self._scalar_events[event].append(float_val)

    def log_vector(self, event: EventType | str, value: NDArray) -> None:
        """Log vector event with type validation.

        Args:
            event: EventType or string (must satisfy event.is_vector)
            value: Vector (NDArray)

        Raises:
            ValueError: If event is not a vector type

        Theory:
            Vectors are copied via np.asarray(...).copy() to ensure
            immutability. External mutations don't affect logged values.

        Examples:
            >>> logger.log_vector(EventType.RESIDUAL, np.array([1.0, 2.0]))
            >>> logger.log_vector("residual", np.array([1.0, 2.0]))  # String also works
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if not event.is_vector:
            msg = f"{event} is not a vector event type"
            raise ValueError(msg)

        if event not in self._vector_events:
            self._vector_events[event] = []

        # Immutable copy
        array_val = np.asarray(value, dtype=np.float64).copy()
        self._vector_events[event].append(array_val)

    def log(self, event: EventType | str, value: float | int | bool | NDArray) -> None:
        """Log any event type (dispatches to typed method).

        Convenience method that automatically routes to log_scalar() or
        log_vector() based on event.is_scalar property.

        Args:
            event: EventType or string
            value: Scalar (float/int/bool) or vector (NDArray)

        Raises:
            ValueError: If event type doesn't match value type

        Examples:
            >>> logger.log(EventType.ITERATION, 5)
            >>> logger.log("iteration", 5)  # String also works
            >>> logger.log(EventType.RESIDUAL, np.array([1.0, 2.0]))
            >>> logger.log("residual", np.array([1.0, 2.0]))  # String also works
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if event.is_scalar:
            self.log_scalar(event, value)  # type: ignore[arg-type]
        else:
            self.log_vector(event, value)  # type: ignore[arg-type]

    def get_scalars(self, event: EventType | str) -> list[float]:
        """Get scalar history for event.

        Args:
            event: EventType or string (must be scalar)

        Returns:
            List of float values (empty list if never logged)

        Raises:
            ValueError: If event is not a scalar type

        Examples:
            >>> norms = logger.get_scalars(EventType.RESIDUAL_NORM)
            >>> norms = logger.get_scalars("residual_norm")  # String also works
            >>> assert isinstance(norms, list)
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if not event.is_scalar:
            msg = f"{event} is not a scalar event type"
            raise ValueError(msg)
        return self._scalar_events.get(event, [])

    def get_vectors(self, event: EventType | str) -> NDArray:
        """Get vector history for event (stacked as 2D array).

        Args:
            event: EventType or string (must be vector)

        Returns:
            Stacked NDArray (shape: num_events × vector_size)
            Empty array if never logged

        Raises:
            ValueError: If event is not a vector type

        Theory:
            Vectors are stacked along axis 0:
            - Input: [r_0, r_1, r_2] where r_k has shape (n,)
            - Output: array with shape (3, n)
            - Output[k] = r_k

        Examples:
            >>> residuals = logger.get_vectors(EventType.RESIDUAL)
            >>> residuals = logger.get_vectors("residual")  # String also works
            >>> assert residuals.shape[0] == num_iterations
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if not event.is_vector:
            msg = f"{event} is not a vector event type"
            raise ValueError(msg)

        vectors = self._vector_events.get(event, [])
        if not vectors:
            return np.empty((0,), dtype=np.float64)
        return np.stack(vectors)

    def get_history(self, event: EventType | str) -> list[float] | NDArray:
        """Get history (dispatches to type-appropriate method).

        Convenience method for backward compatibility. Prefer using
        get_scalars() or get_vectors() directly for type safety.

        Args:
            event: EventType or string

        Returns:
            list[float] if scalar event, NDArray if vector event

        Examples:
            >>> history = logger.get_history(EventType.RESIDUAL_NORM)
            >>> history = logger.get_history("residual_norm")  # String also works
            >>> assert isinstance(history, list)
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if event.is_scalar:
            return self.get_scalars(event)
        else:
            return self.get_vectors(event)

    def get_latest(self, event: EventType | str) -> NDArray | float | None:
        """Get most recent value for event.

        Args:
            event: EventType or string

        Returns:
            Latest scalar (float) or vector (NDArray), or None if never logged

        Examples:
            >>> latest_norm = logger.get_latest(EventType.RESIDUAL_NORM)
            >>> latest_norm = logger.get_latest("residual_norm")  # String also works
            >>> assert isinstance(latest_norm, float) or latest_norm is None
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if event.is_scalar:
            scalars = self._scalar_events.get(event, [])
            return scalars[-1] if scalars else None
        else:
            vectors = self._vector_events.get(event, [])
            if not vectors:
                return None
            # Return copy to maintain immutability
            return vectors[-1].copy()

    def prepend(self, event: EventType | str, value: float | int | bool | NDArray) -> None:
        """Add value to beginning of history (for iteration 0 logging).

        Args:
            event: EventType or string
            value: Scalar or vector to prepend

        Theory:
            Used to add iteration 0 data after solver initialization
            completes. Iteration 0 residual computed after first state
            initialized, so prepended to history.

        Examples:
            >>> logger.prepend(EventType.RESIDUAL_NORM, initial_residual)
            >>> logger.prepend("residual_norm", initial_residual)  # String also works
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if event.is_scalar:
            if event not in self._scalar_events:
                self._scalar_events[event] = []
            self._scalar_events[event].insert(0, float(value))
        else:
            if event not in self._vector_events:
                self._vector_events[event] = []
            array_val = np.asarray(value, dtype=np.float64).copy()
            self._vector_events[event].insert(0, array_val)

    def clear(self) -> None:
        """Clear all logged events.

        Examples:
            >>> logger.clear()
            >>> assert logger.event_count(EventType.RESIDUAL_NORM) == 0
        """
        self._scalar_events.clear()
        self._vector_events.clear()

    def has_event(self, event: EventType | str) -> bool:
        """Check if event has been logged.

        Args:
            event: EventType or string to check

        Returns:
            True if event has at least one logged value

        Examples:
            >>> assert logger.has_event(EventType.RESIDUAL_NORM)
            >>> assert logger.has_event("residual_norm")  # String also works
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if event.is_scalar:
            return event in self._scalar_events and len(self._scalar_events[event]) > 0
        else:
            return event in self._vector_events and len(self._vector_events[event]) > 0

    def event_count(self, event: EventType | str) -> int:
        """Get number of logged values for event.

        Args:
            event: EventType or string

        Returns:
            Number of times event was logged

        Examples:
            >>> count = logger.event_count(EventType.ITERATION)
            >>> count = logger.event_count("iteration")  # String also works
            >>> assert count == num_iterations
        """
        # Convert string to EventType if needed
        event = EventType(event) if isinstance(event, str) else event
        if event.is_scalar:
            return len(self._scalar_events.get(event, []))
        else:
            return len(self._vector_events.get(event, []))

    def get_all_event_names(self) -> list[EventType]:
        """Get list of all logged event types.

        Returns:
            List of EventType values that have been logged

        Examples:
            >>> events = logger.get_all_event_names()
            >>> assert EventType.RESIDUAL_NORM in events
        """
        return list(self._scalar_events.keys()) + list(self._vector_events.keys())
