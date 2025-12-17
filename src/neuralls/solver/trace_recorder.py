"""Trace recording utilities for solver iteration logging.

This module provides orthogonal trace recording for solver iterations without
coupling to solver implementation. It stores immutable copies of iteration
artifacts (vectors and scalars) for diagnostics.

Design Principles:
    - Single Responsibility: Only handles logging, not solver logic
    - Orthogonal Design: Can be used or not used without affecting solver
    - Immutable Events: All values copied to prevent mutation
    - Open/Closed: Easy to extend with new event types

Theory:
    Event sourcing captures all state changes as a sequence of events.
    This enables:
    1. Full iteration history reconstruction
    2. Post-hoc analysis and debugging
    3. Zero overhead when not used (optional dependency)
    4. Separation of logging concerns from algorithm logic

Usage:
    >>> logger = TraceRecorder()
    >>> logger.log("residual", r0)
    >>> logger.log("residual", r1)
    >>> logger.log("solution", x0)
    >>>
    >>> residuals = logger.get_history("residual")  # stacked array
    >>> solutions = logger.get_history("solution")  # stacked array
    >>> latest_r = logger.get_latest("residual")     # r1

References:
    - Event Sourcing: Martin Fowler (2005)
    - SOLID Principles: Single Responsibility, Open/Closed
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ScalarHistory(np.ndarray):
    """Scalar history that compares cleanly in assertions."""

    def __new__(cls, input_array: list[float] | np.ndarray) -> "ScalarHistory":
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        return obj

    def __array_finalize__(self, obj: object) -> None:  # pragma: no cover - ndarray protocol
        if obj is None:
            return

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        try:
            a = np.asarray(self, dtype=np.float64)
            b = np.asarray(other, dtype=np.float64)
        except Exception:
            return False
        return bool(np.allclose(a, b))

    def __bool__(self) -> bool:  # Avoid numpy's ambiguous truth value
        return self.size > 0


class TraceRecorder:
    """Trace recorder for solver iteration values.

    Each log() call creates an immutable copy of the value. Supports both
    vector (NDArray) and scalar (float) entries.

    Design Pattern:
        Event Sourcing - Store all state changes as immutable events

    Attributes:
        residuals: List of residual vectors
        solutions: List of solution vectors
        residual_norms: List of residual norms (absolute)
        _events: Dictionary for other vector events
        _scalar_events: Dictionary for other scalar events

    Mathematical Background:
        For iterative solvers, we typically want to log:
        - Residual vectors: r_k = b - A x_k at each iteration k
        - Solution vectors: x_k at each iteration k
        - Search directions: p_k at each iteration k
        - Residual norms: ||r_k|| at each iteration k
        - Step lengths: α_k at each iteration k

    Usage Patterns:
        1. Log vectors (residuals, solutions):
            logger.log("residual", r_k)

        2. Log scalars (norms, step lengths):
            logger.log("residual_norm", norm(r_k))

        3. Retrieve full history:
            all_residuals = logger.get_history("residual")

        4. Retrieve latest value:
            latest_norm = logger.get_latest("residual_norm")

    Examples:
        >>> logger = TraceRecorder()
        >>> logger.log("residual", np.array([1.0, 2.0, 3.0]))
        >>> logger.log("residual", np.array([0.5, 1.0, 1.5]))
        >>> logger.log("residual_norm", 2.5)
        >>> logger.log("residual_norm", 1.25)
        >>>
        >>> residuals = logger.get_history("residual")
        >>> assert len(residuals) == 2
        >>> norms = logger.get_history("residual_norm")
        >>> assert norms == [2.5, 1.25]
    """

    def __init__(self) -> None:
        """Initialize empty trace buffers."""
        self.residuals: list[NDArray] = []
        self.solutions: list[NDArray] = []
        self.residual_norms: list[float] = []
        self._events: dict[str, list[NDArray]] = {}
        self._scalar_events: dict[str, list[float]] = {}

    def log(self, event_name: str, value: NDArray | float | int) -> None:
        """Log an iteration value as an immutable trace entry.

        Args:
            event_name: Name of the entry (e.g., "residual", "solution",
                "residual_norm"). Recommended naming:
                - Vectors: descriptive name ("residual", "solution")
                - Scalars: name with unit/type ("residual_norm", "step_length")
            value: Vector (NDArray) or scalar (float/int) to log.
                - Vectors are copied via np.asarray(...).copy()
                - Scalars are converted to float

        Theory:
            Immutability is enforced by copying:
            1. Vectors: np.asarray(value).copy() creates independent copy
            2. Scalars: float(value) creates new Python object
            3. External mutations to original value don't affect log

            Why separate storage for vectors vs scalars?
            - Memory efficiency: scalars stored as Python floats, not arrays
            - Type safety: get_history() returns appropriate type
            - Performance: avoid unnecessary array creation for scalars

        Examples:
            >>> logger = TraceRecorder()
            >>> r = np.array([1.0, 2.0])
            >>> logger.log("residual", r)
            >>> r[0] = 999.0  # Mutate original
            >>> logged = logger.get_history("residual")[0]
            >>> assert logged[0] == 1.0  # Log unchanged (immutable)
        """
        if isinstance(value, (int, float)):
            # Scalar event
            if event_name == "residual_norm":
                self.residual_norms.append(float(value))
                return
            if event_name not in self._scalar_events:
                self._scalar_events[event_name] = []
            self._scalar_events[event_name].append(float(value))
        else:
            # Vector event
            array_val = np.asarray(value, dtype=np.float64).copy()
            if event_name == "residual":
                self.residuals.append(array_val)
                return
            if event_name == "solution":
                self.solutions.append(array_val)
                return
            if event_name not in self._events:
                self._events[event_name] = []
            self._events[event_name].append(array_val)

    def prepend(self, event_name: str, value: NDArray | float | int) -> None:
        """Prepend an event to the beginning of the history.

        This is useful for adding the initial state (iteration 0) when the
        logging system starts recording after the first iteration.

        Args:
            event_name: Name of the event (e.g., "residual", "residual_norm")
            value: Vector (NDArray) or scalar (float/int) to prepend

        Theory:
            Maintains event sourcing guarantees by:
            1. Copying the value to ensure immutability
            2. Inserting at index 0 to preserve chronological order
            3. Following the same type detection as log()

            This method is needed because scipy's CG callback starts at iteration 1,
            not iteration 0. We compute the initial residual separately and prepend
            it to the history to get complete convergence data.

        Examples:
            >>> logger = TraceRecorder()
            >>> logger.log("residual_norm", 0.5)
            >>> logger.log("residual_norm", 0.25)
            >>> logger.prepend("residual_norm", 1.0)  # Add initial residual
            >>> assert logger.get_history("residual_norm") == [1.0, 0.5, 0.25]

            >>> # Vector prepending
            >>> logger.log("residual", np.array([0.5, 0.3]))
            >>> logger.prepend("residual", np.array([1.0, 0.8]))
            >>> residuals = logger.get_history("residual")
            >>> assert len(residuals) == 2
            >>> assert np.allclose(residuals[0], [1.0, 0.8])
        """
        if isinstance(value, (int, float)):
            # Scalar event
            if event_name == "residual_norm":
                self.residual_norms.insert(0, float(value))
                return
            if event_name not in self._scalar_events:
                self._scalar_events[event_name] = []
            self._scalar_events[event_name].insert(0, float(value))
        else:
            # Vector event
            array_val = np.asarray(value, dtype=np.float64).copy()
            if event_name == "residual":
                self.residuals.insert(0, array_val)
                return
            if event_name == "solution":
                self.solutions.insert(0, array_val)
                return
            if event_name not in self._events:
                self._events[event_name] = []
            self._events[event_name].insert(0, array_val)

    def get_history(self, event_name: str) -> NDArray:
        """Retrieve full history for an event.

        Returns:
            - For vector events: stacked ndarray (empty array if none logged)
            - For scalar events: list of floats
        """
        if event_name == "residual":
            return (
                np.stack(self.residuals)
                if self.residuals
                else np.empty((0,), dtype=np.float64)
            )
        if event_name == "solution":
            return (
                np.stack(self.solutions)
                if self.solutions
                else np.empty((0,), dtype=np.float64)
            )
        if event_name == "residual_norm":
            return ScalarHistory(self.residual_norms)
        if event_name in self._scalar_events:
            return ScalarHistory(self._scalar_events[event_name])
        if event_name not in self._events or not self._events[event_name]:
            return np.empty((0,), dtype=np.float64)
        return np.stack(self._events[event_name])

    def get_latest(self, event_name: str) -> NDArray | float | None:
        """Get most recent value for an event.

        Returns the last value logged for the given event name, or None
        if no values have been logged.

        Args:
            event_name: Name of the event to retrieve.

        Returns:
            - For vector events: Copy of latest NDArray
            - For scalar events: Latest float value
            - For non-existent events: None

        Theory:
            Returning the latest value is useful for checking current state
            without retrieving full history. Common use cases:
            - Check latest residual norm before next iteration
            - Retrieve final solution after solver completes
            - Get current step length for adaptive algorithms

        Examples:
        >>> logger = TraceRecorder()
            >>> logger.log("residual_norm", 1.0)
            >>> logger.log("residual_norm", 0.5)
            >>> latest = logger.get_latest("residual_norm")
            >>> assert latest == 0.5
            >>> assert logger.get_latest("nonexistent") is None
        """
        if event_name in self._scalar_events:
            history = self._scalar_events[event_name]
            return history[-1] if history else None
        if event_name == "residual_norm":
            return self.residual_norms[-1] if self.residual_norms else None
        if event_name == "residual":
            return self.residuals[-1].copy() if self.residuals else None
        if event_name == "solution":
            return self.solutions[-1].copy() if self.solutions else None
        history = self._events.get(event_name, [])
        return history[-1].copy() if history else None

    def clear(self) -> None:
        """Clear all logged events.

        Removes all events from both vector and scalar logs. Useful for
        reusing logger across multiple solver runs.

        Theory:
            Clearing the log enables logger reuse without creating new
            objects. This is memory-efficient for repeated solves.

        Examples:
            >>> logger = TraceRecorder()
            >>> logger.log("residual", np.array([1.0, 2.0]))
            >>> logger.clear()
            >>> assert len(logger.get_history("residual")) == 0
        """
        self.residuals.clear()
        self.solutions.clear()
        self.residual_norms.clear()
        self._events.clear()
        self._scalar_events.clear()

    def has_event(self, event_name: str) -> bool:
        """Check if event has been logged.

        Args:
            event_name: Name of the event to check.

        Returns:
            True if at least one value logged for this event, False otherwise.

        Examples:
            >>> logger = TraceRecorder()
            >>> assert not logger.has_event("residual")
            >>> logger.log("residual", np.array([1.0]))
            >>> assert logger.has_event("residual")
        """
        if event_name == "residual":
            return len(self.residuals) > 0
        if event_name == "solution":
            return len(self.solutions) > 0
        if event_name == "residual_norm":
            return len(self.residual_norms) > 0
        return (
            event_name in self._scalar_events
            and len(self._scalar_events[event_name]) > 0
        ) or (event_name in self._events and len(self._events[event_name]) > 0)

    def event_count(self, event_name: str) -> int:
        """Get number of times event has been logged.

        Args:
            event_name: Name of the event to count.

        Returns:
            Number of values logged for this event (0 if never logged).

        Examples:
            >>> logger = TraceRecorder()
            >>> logger.log("residual_norm", 1.0)
            >>> logger.log("residual_norm", 0.5)
            >>> assert logger.event_count("residual_norm") == 2
            >>> assert logger.event_count("nonexistent") == 0
        """
        if event_name == "residual":
            return len(self.residuals)
        if event_name == "solution":
            return len(self.solutions)
        if event_name == "residual_norm":
            return len(self.residual_norms)
        if event_name in self._scalar_events:
            return len(self._scalar_events[event_name])
        return len(self._events.get(event_name, []))

    def get_all_event_names(self) -> list[str]:
        """Get names of all logged events.

        Returns:
            List of event names that have been logged (both vector and scalar).

        Examples:
            >>> logger = TraceRecorder()
            >>> logger.log("residual", np.array([1.0]))
            >>> logger.log("residual_norm", 1.0)
            >>> names = logger.get_all_event_names()
            >>> assert "residual" in names
            >>> assert "residual_norm" in names
        """
        names = set(self._events.keys()) | set(self._scalar_events.keys())
        if self.residuals:
            names.add("residual")
        if self.solutions:
            names.add("solution")
        if self.residual_norms:
            names.add("residual_norm")
        return list(names)
