# Solver Monitoring System

This module provides comprehensive monitoring for iterative solvers through two complementary systems following industry-standard patterns.

## Architecture Overview

The monitoring system separates **continuous monitoring** from **discrete events**:

```
Monitoring System
├── IterationHistory (Continuous Data)
│   ├── ScalarHistory (residual norms)
│   └── VectorHistory (residuals, solutions, directions)
│
├── EventLog (Discrete Events)
│   └── SolverEvent (CONVERGED, BREAKDOWN, ORTHO_BREAKDOWN)
│
└── ResidualHistoryTracker (SciPy Integration)
```

## Design Principles

### 1. Separation of Concerns

**Continuous Monitoring** (IterationHistory):
- Data collected at **every iteration**
- Example: Residual norms `[1.0, 0.5, 0.25, 0.125, ...]`
- Pattern: **Telemetry/Metrics** (like Prometheus, StatsD, OpenTelemetry)

**Discrete Events** (EventLog):
- Events that **occur once**
- Example: Converged at iteration 42, breakdown at iteration 10
- Pattern: **Event Sourcing** (immutable event records)

### 2. Direct Attribute Access

**New API** (Direct access):
```python
residual_norms = iteration_history.residual_norms.to_list()
residuals = iteration_history.residuals.to_array()
```

**Old API** (Enum indexing - removed):
```python
# DON'T USE - this pattern has been replaced
residual_norms = event_log.get_scalars(EventType.RESIDUAL_NORM)
```

### 3. Immutable Storage

All history classes use frozen dataclasses with functional updates:

```python
# Immutable updates
history = ScalarHistory.empty()
history = history.add(1.0)  # Returns NEW history
history = history.add(0.5)  # Returns NEW history
```

---

## Core Classes

### 1. IterationHistory

**Purpose**: Track continuous data collected at every iteration.

**Attributes**:
- `residual_norms`: `ScalarHistory` - Always tracked
- `residuals`: `VectorHistory | None` - FULL mode only
- `solutions`: `VectorHistory | None` - FULL mode only
- `directions`: `VectorHistory | None` - FULL mode only (for debugging, NOT orthogonalization)

**Note**: DirectionHistory (for orthogonalization) lives in CGState, NOT here.

**Example**:
```python
from neuralls.solver.monitoring import IterationHistory, TraceMode

# Create history
history = IterationHistory(mode=TraceMode.MINIMAL)

# Log iteration data
history.log_iteration(residual_norm=1.0)
history.log_iteration(residual_norm=0.5)

# Access data
norms = history.residual_norms.to_list()  # [1.0, 0.5]
count = history.iteration_count()  # 2
```

**FULL Mode Example**:
```python
import numpy as np
from neuralls.solver.monitoring import IterationHistory, TraceMode

history = IterationHistory(mode=TraceMode.FULL)

r = np.array([1.0, 2.0])
x = np.array([0.1, 0.2])
p = np.array([0.5, 0.6])

history.log_iteration(
    residual_norm=2.236,
    residual=r,
    solution=x,
    direction=p
)

# Access vectors
residuals = history.residuals.to_array()  # Shape: (1, 2)
solutions = history.solutions.to_array()  # Shape: (1, 2)
```

---

### 2. EventLog

**Purpose**: Track discrete events that occur during solving.

**Event Types**:
- `EventType.CONVERGED`: Convergence criterion satisfied
- `EventType.BREAKDOWN`: Numerical breakdown detected
- `EventType.ORTHO_BREAKDOWN`: Orthogonalization breakdown

**Example**:
```python
from neuralls.solver.monitoring import EventLog, EventType

log = EventLog()

# Record events
log.record(EventType.CONVERGED, iteration=42, final_residual=1e-8)
log.record(EventType.ORTHO_BREAKDOWN, iteration=25)

# Query events
converged_at = log.converged_at()  # 42
breakdown_at = log.ortho_breakdown_at()  # 25

# Get all events
events = log.all_events()
for event in events:
    print(f"{event.type} at iteration {event.iteration}")
```

**Key Insight**:
```python
# OLD WAY (inefficient): Store boolean flags every iteration
# [0, 0, 0, 1, 1, 1, ...] - wastes memory

# NEW WAY (efficient): Store event with iteration number
# converged_at = 3 - minimal storage
```

---

### 3. ScalarHistory & VectorHistory

**Purpose**: Immutable storage for iteration data.

**ScalarHistory Example**:
```python
from neuralls.solver.monitoring import ScalarHistory

history = ScalarHistory.empty()
history = history.add(1.0)
history = history.add(0.5)
history = history.prepend(2.0)  # Add at beginning

norms = history.to_list()  # [2.0, 1.0, 0.5]
length = len(history)  # 3
value = history[0]  # 2.0
```

**VectorHistory Example**:
```python
import numpy as np
from neuralls.solver.monitoring import VectorHistory

history = VectorHistory.empty()
history = history.add(np.array([1.0, 2.0]))
history = history.add(np.array([3.0, 4.0]))

array = history.to_array()  # Shape: (2, 2)
vec = history[0]  # array([1.0, 2.0]) - returns copy
```

---

### 4. ResidualHistoryTracker

**Purpose**: SciPy-specific callback integration (scipy.sparse.linalg.cg).

**Usage**:
```python
from neuralls.solver.monitoring import ResidualHistoryTracker

tracker = ResidualHistoryTracker()
tracker.record_residual(norm_abs=1.0, norm_rel=0.1)
tracker.record_residual(norm_abs=0.5, norm_rel=0.05)

abs_norms = tracker.residual_norms_abs  # [1.0, 0.5]
rel_norms = tracker.residual_norms_rel  # [0.1, 0.05]
```

**Note**: Only used by `SciPyCGSolver`. Native solvers (PCG, FCG) use `IterationHistory` directly.

---

## TraceMode Settings

Control monitoring granularity:

| Mode | Memory | Data Collected |
|------|--------|----------------|
| `DISABLED` | 0 bytes | No monitoring |
| `MINIMAL` | ~KB | Residual norms only |
| `FULL` | ~MB-GB | Norms + all vectors |

**Example**:
```python
from neuralls.solver.monitoring import TraceMode

# Minimal (default)
history = IterationHistory(mode=TraceMode.MINIMAL)
history.log_iteration(residual_norm=1.0)
assert history.residuals is None  # Vectors not collected

# Full (debugging)
history = IterationHistory(mode=TraceMode.FULL)
history.log_iteration(residual_norm=1.0, residual=r)
assert history.residuals is not None  # Vectors collected
```

**Memory Impact**:

For problem size n=10,000, 100 iterations:
- **MINIMAL**: ~4 KB (just scalar norms)
- **FULL**: ~24 MB (norms + residuals + solutions + directions)

---

## Usage Patterns

### Pattern 1: Factory Functions

Factories automatically create monitoring objects based on trace mode:

```python
from neuralls.solver.factories import flexible_cg
from neuralls.solver.monitoring import TraceMode

# Minimal monitoring (default)
x, result = flexible_cg(A, b, trace_mode=TraceMode.MINIMAL)
norms = result.iteration_history.residual_norms.to_list()

# Full debugging
x, result = flexible_cg(A, b, trace_mode=TraceMode.FULL)
residuals = result.iteration_history.residuals.to_array()
solutions = result.iteration_history.solutions.to_array()

# Check convergence
if result.event_log.converged_at():
    print(f"Converged at iteration {result.event_log.converged_at()}")
```

### Pattern 2: Manual Solver Construction

For custom monitoring setup:

```python
from neuralls.solver.solvers import FlexibleCGSolver
from neuralls.solver.monitoring import IterationHistory, EventLog, TraceMode

# Create monitoring objects
iteration_history = IterationHistory(mode=TraceMode.FULL)
event_log = EventLog()

# Create solver with monitoring
solver = FlexibleCGSolver(
    iteration_history=iteration_history,
    event_log=event_log,
    trace_mode=TraceMode.FULL
)

# Solve
x, result = solver.solve(A, b, rtol=1e-6)

# Access monitoring data
norms = result.iteration_history.residual_norms.to_list()
converged = result.event_log.converged_at()
```

### Pattern 3: SciPy Integration

SciPyCGSolver uses a different pattern due to callback-based API:

```python
from neuralls.solver.solvers import SciPyCGSolver
from neuralls.solver.monitoring import IterationHistory, EventLog

solver = SciPyCGSolver(
    iteration_history=IterationHistory(mode=TraceMode.MINIMAL),
    event_log=EventLog()
)

x, result = solver.solve(A, b)
norms = result.iteration_history.residual_norms.to_list()
```

---

## Migration Guide

### Old API → New API

**Accessing Residual Norms**:
```python
# OLD
norms = result.event_log.get_scalars(EventType.RESIDUAL_NORM)

# NEW
norms = result.iteration_history.residual_norms.to_list()
```

**Accessing Vectors**:
```python
# OLD
residuals = result.event_log.get_vectors(EventType.RESIDUAL)

# NEW
residuals = result.iteration_history.residuals.to_array()
```

**Checking Convergence**:
```python
# OLD
converged_flags = result.event_log.get_scalars(EventType.CONVERGED)
if converged_flags[-1] == 1.0:
    ...

# NEW
if result.event_log.converged_at() is not None:
    iteration = result.event_log.converged_at()
```

**Creating Monitoring**:
```python
# OLD
from neuralls.solver.monitoring import TraceRecorder
event_log = TraceRecorder()

# NEW
from neuralls.solver.monitoring import IterationHistory, EventLog, TraceMode
iteration_history = IterationHistory(mode=TraceMode.MINIMAL)
event_log = EventLog()
```

---

## Implementation Notes

### Why Two Systems?

**Problem with old design**:
- Mixed continuous data (residual norms every iteration) with discrete events (breakdown at iteration 42)
- Used EventType enum to index dict of histories - wasteful
- Stored CONVERGED as `[0, 0, 0, 1, 1, 1, ...]` instead of just `converged_at = 3`

**Solution**:
- **IterationHistory**: Continuous metrics (Telemetry pattern)
- **EventLog**: Discrete events (Event Sourcing pattern)
- **Direct access**: `history.residual_norms` not `tracker.get(EventType.RESIDUAL_NORM)`

### Orthogonalization Monitoring

**Important**: Orthogonalization diagnostics are NOT exposed to solver monitoring.

- **DirectionHistory** (d_k, q_k window): Lives in CGState (internal to solver)
- **IterationHistory.directions**: For visualization/debugging only
- **EventLog.ORTHO_BREAKDOWN**: Only breakdown events, no coefficients

This separation keeps orthogonalization as an internal implementation detail.

### Thread Safety

All history classes are immutable (frozen dataclasses), making them thread-safe for reading. Updates return new instances:

```python
# Thread-safe pattern
history = ScalarHistory.empty()
history = history.add(1.0)  # Returns NEW instance
```

---

## References

### Design Patterns

1. **Telemetry/Metrics Pattern**: Prometheus, StatsD, OpenTelemetry
   - Direct attribute access for different metric types
   - Clear separation of metric categories

2. **Event Sourcing Pattern**: Martin Fowler, Greg Young
   - Immutable event records
   - Append-only log
   - Query by event type

### Related Documentation

- [Solver README](../README.md) - Solver implementations
- [CLAUDE.md](../../../../CLAUDE.md) - Project coding standards
