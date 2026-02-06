"""Solution validation utilities for iterative solvers.

This module provides reusable validation functions for checking solution
validity and recording breakdown events. These utilities are shared across
all solver implementations.

Design:
    - Pure functions (no state)
    - Reusable across solver types
    - Consistent breakdown event recording
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..monitoring.event_log import EventLog


def check_solution_validity(solution: NDArray) -> bool:
    """Check if solution contains only finite values.

    Args:
        solution: Solution vector to validate

    Returns:
        True if solution is valid (all finite), False otherwise

    Example:
        >>> import numpy as np
        >>> check_solution_validity(np.array([1.0, 2.0, 3.0]))
        True
        >>> check_solution_validity(np.array([1.0, np.nan, 3.0]))
        False
        >>> check_solution_validity(np.array([1.0, np.inf, 3.0]))
        False
    """
    return bool(np.all(np.isfinite(solution)))


def record_breakdown_event(
    event_log: EventLog | None,
    solution: NDArray,
    iteration: int,
) -> None:
    """Record breakdown event when solution contains NaN/Inf.

    Args:
        event_log: EventLog to record to (can be None)
        solution: Solution vector with invalid values
        iteration: Final iteration number

    Example:
        >>> import numpy as np
        >>> from neuralls.solver.monitoring.event_log import EventLog
        >>> from neuralls.solver.monitoring.events import EventType
        >>>
        >>> event_log = EventLog()
        >>> solution = np.array([1.0, np.nan, 3.0])
        >>> record_breakdown_event(event_log, solution, iteration=5)
        >>>
        >>> event = event_log.find_first(EventType.BREAKDOWN)
        >>> event.iteration
        5
        >>> event.metadata["reason"]
        'nan_in_solution'
    """
    if event_log is None:
        return

    from ..monitoring.events import EventType

    has_nan = bool(np.any(np.isnan(solution)))
    has_inf = bool(np.any(np.isinf(solution)))

    reason_parts = []
    if has_nan:
        reason_parts.append("nan_in_solution")
    if has_inf:
        reason_parts.append("inf_in_solution")

    event_log.record(
        EventType.BREAKDOWN,
        iteration=iteration,
        reason=", ".join(reason_parts),
    )