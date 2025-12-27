"""Monitoring and callback components for solver execution.

This package provides components for tracking solver execution:
- trace_recorder: Type-safe event logging (TraceRecorder)
- events: EventType enum for type-safe event names
- trace_mode: TraceMode enum for controlling logging granularity
- history_tracker: Residual and solution history tracking (HistoryTracker)
- callbacks: SciPy callback adaptation (SciPyCallbackAdapter, InitialStateComputer)

Design Principles:
    - Single Responsibility: Each class has one clear purpose
    - Type Safety: EventType enum eliminates magic strings
    - Composition: Components compose via dependency injection
    - Optional: All monitoring is optional (zero overhead when unused)
"""

from .callbacks import InitialStateComputer, SciPyCallbackAdapter
from .events import EventType
from .history_tracker import HistoryTracker
from .trace_mode import TraceMode
from .trace_recorder import TraceRecorder

__all__ = [
    "EventType",
    "TraceMode",
    "TraceRecorder",
    "HistoryTracker",
    "SciPyCallbackAdapter",
    "InitialStateComputer",
]
