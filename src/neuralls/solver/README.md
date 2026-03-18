# Solver Implementations

This directory contains iterative solver implementations for sparse linear systems.

## Solver Hierarchy

```
IterativeSolverBase (base.py)
    ↓ inherits
ConjugateGradientSolver (conjugate_gradient.py) - Unified CG implementation
    ↑ composes
    ├─ DirectionStrategy (strategies/direction.py)
    │   ├─ TwoTermRecurrenceStrategy (PCG behavior)
    │   ├─ OrthogonalizationDirectionStrategy (FCG behavior)
    │   └─ CompositeDirectionStrategy (PCG + reorthog)
    ├─ Preconditioner
    ├─ ConvergenceCriterion
    └─ OrthogonalizationStrategy (optional)

SciPyCGSolver (scipy_wrapper.py) - Separate wrapper
```

## Design Principles

- **Strategy Pattern**: Direction computation via DirectionStrategy (composition over inheritance)
- **Template Method**: IterativeSolverBase defines iteration skeleton
- **Single Responsibility**: ConjugateGradientSolver implements unified CG algorithm only
- **Composition**: Strategies injected, not inherited

---

## Unified CG Algorithm

All CG variants are implemented through a single `ConjugateGradientSolver` class using the Strategy pattern.

### Algorithm (Standard CG Iteration)

```
Initialize: u_0, r_0 = b - A*u_0

For k = 0, 1, ..., until convergence:
    1. w_k = M^{-1}(r_k)                        # Apply preconditioner
    2. d_k = strategy.compute_direction(w_k, state)  # STRATEGY PATTERN
    3. q_k = A @ d_k                            # Matrix-vector product
    4. α_k = (d_k, r_k) / (d_k, A d_k)          # Step length
    5. u_{k+1} = u_k + α_k * d_k                # Update solution
    6. r_{k+1} = r_k - α_k * q_k                # Update residual
```

**Step 2 is the ONLY difference between CG variants.** All other steps are identical.

### Configuring CG Variants

The `ConjugateGradientSolver` supports all CG variants through `DirectionStrategy`:

| Variant | DirectionStrategy | Use Case |
|---------|-------------------|----------|
| **PCG** | `TwoTermRecurrenceStrategy` | Fixed SPD preconditioners |
| **FCG** | `OrthogonalizationDirectionStrategy` | Variable, non-SPD, or non-linear preconditioners |
| **PCG+Reorthog** | `CompositeDirectionStrategy` | Ill-conditioned problems |

---

## Direction Strategies

The unified `ConjugateGradientSolver` uses `DirectionStrategy` to implement different CG variants through composition.

### 1. TwoTermRecurrenceStrategy (Standard PCG)

Standard two-term recurrence for preconditioned CG.

**Algorithm:**
```
d_k = w_k + β_k * d_{k-1}
where β_k = (r_k, w_k) / (r_{k-1}, w_{k-1})
```

**Properties:**
- Memory: O(1) - only current direction
- Use for: Classical SPD preconditioners
- Stateless: Stores previous values in CGState (reusable across solves)

**Example:**
```python
from neuralls.solver import ConjugateGradientSolver, TwoTermRecurrenceStrategy

direction_strategy = TwoTermRecurrenceStrategy()
solver = ConjugateGradientSolver(direction_strategy=direction_strategy)
x, result = solver.solve(A, b)
```

**Code Variable Mapping:**
- `u_k` ↔ `u` (solution iterate)
- `r_k` ↔ `r` (residual)
- `w_k` ↔ `w` (preconditioned residual)
- `d_k` ↔ `d` (search direction)
- `q_k` ↔ `q` (matrix-vector product A @ d)

---

### 2. OrthogonalizationDirectionStrategy (Flexible CG)

Flexible CG with explicit orthogonalization for variable preconditioners.

**Algorithm:**
```
d_k = w_k - Σ_j [(w_k, A d_j) / (d_j, A d_j)] d_j
```

**Properties:**
- Memory: O(m*n) for window size m
- Use for: Variable/non-SPD preconditioners
- Stateless: Operates on history stored in CGState

**Example:**
```python
from neuralls.solver import (
    ConjugateGradientSolver,
    OrthogonalizationDirectionStrategy,
    PeriodicRestartOrthogonalization,
)

orthog = PeriodicRestartOrthogonalization(m_max=10)
direction_strategy = OrthogonalizationDirectionStrategy(orthog)
solver = ConjugateGradientSolver(direction_strategy=direction_strategy)
x, result = solver.solve(A, b)
```

---

### 3. CompositeDirectionStrategy (PCG + Reorthogonalization)

Combines two-term recurrence with reorthogonalization for extra stability.

**Algorithm:**
```
Phase 1: d_k = w_k + β_k * d_{k-1}              # Two-term recurrence
Phase 2: d_k := orthogonalize(d_k, history)     # Reorthogonalization
```

**Properties:**
- Memory: O(m*n) for window size m
- Use for: Ill-conditioned problems needing extra stability
- Stateless: Both base strategy and reorthogonalization use CGState

**Example:**
```python
from neuralls.solver import (
    ConjugateGradientSolver,
    TwoTermRecurrenceStrategy,
    CompositeDirectionStrategy,
    TruncatedGramSchmidt,
)

base = TwoTermRecurrenceStrategy()
reorthog = TruncatedGramSchmidt(window_size=20)
direction_strategy = CompositeDirectionStrategy(base, reorthog)
solver = ConjugateGradientSolver(direction_strategy=direction_strategy)
x, result = solver.solve(A, b)
```

---

## Factory Functions (Recommended)

The easiest way to use the solver is through factory functions, which handle strategy configuration automatically.

### flexible_cg()

**Use for:** Neural, variable, or other non-linear preconditioners

**Example:**
```python
from neuralls.solver import flexible_cg

x, result = flexible_cg(A, b, m_max=10, rtol=1e-6)
```

**Under the hood:**
```python
orthog = PeriodicRestartOrthogonalization(m_max=10)
direction_strategy = OrthogonalizationDirectionStrategy(orthog)
solver = ConjugateGradientSolver(direction_strategy=direction_strategy)
```

---

### preconditioned_cg()

**Use for:** Classical preconditioners, standard PCG

**Example:**

```python
from neuralls.solver import pcg

x, result = pcg(A, b, preconditioner=jacobi, rtol=1e-6)
```

**With reorthogonalization:**
```python
x, result = preconditioned_cg(A, b, m_max=20, rtol=1e-6)
```

**Under the hood:**
```python
base = TwoTermRecurrenceStrategy()
if m_max is not None:
    reorthog = create_fcg_orthogonalization(m_max)
    direction_strategy = CompositeDirectionStrategy(base, reorthog)
else:
    direction_strategy = base
solver = ConjugateGradientSolver(direction_strategy=direction_strategy)
```

---

## scipy.cg Compatibility

Our `ConjugateGradientSolver` closely matches `scipy.sparse.linalg.cg` behavior:

| Feature | scipy.cg | ConjugateGradientSolver |
|---------|----------|------------------------|
| Beta formula | Direct division | Direct division (matches) |
| Alpha formula | `rho / dotprod(p, q)` | `(d,r) / (d,q)` (equivalent) |
| Zero checks | None | None (matches) |
| NaN/Inf detection | None (lets propagate) | Checked in solution only |
| Breakdown detection | None | Yes (NaN/Inf in solution) |

**Design Philosophy:** We match scipy.cg's minimal-guard approach for numerical operations (no zero checks, direct division). For NaN/Inf handling:
- Convergence criterion returns `False` if residual norm or RHS norm is non-finite
- Final solution is checked for NaN/Inf in `_build_result()` → `converged=False` if present
- No per-iteration checks - we verify the final state only (simpler and more efficient)

---

## SciPyCGSolver (scipy.sparse.linalg.cg Wrapper)

Wrapper around scipy's CG implementation with callback-based monitoring.

**Use Cases:**
- Comparison baseline against custom implementations
- Validation against reference scipy implementation
- When scipy's implementation is preferred

**Features:**
- Callback integration via `SciPyCallbackAdapter`
- History tracking via `ResidualHistoryTracker` (scipy-specific)
- Continuous monitoring via `IterationHistory`
- Compatible `SolverResult` output format

**Example:**
```python
from neuralls.solver import SciPyCGSolver

solver = SciPyCGSolver()
x, result = solver.solve(A, b, rtol=1e-6)
```

---

## Monitoring System

All solvers support optional monitoring of iteration data through two complementary systems:

### 1. IterationHistory (Continuous Monitoring)

Tracks continuous data collected at every iteration following the **Telemetry/Metrics pattern**.

**Features:**
- Direct attribute access: `iteration_history.residual_norms` (no enum indexing)
- Mode-aware collection (MINIMAL vs FULL)
- Immutable history objects (ScalarHistory, VectorHistory)
- Functional updates for thread safety

**Data Collected:**
- **MINIMAL mode**: Residual norms only (low memory overhead)
- **FULL mode**: Residual norms + residual/solution/direction vectors (debugging)

**Example:**
```python
from neuralls.solver.factories import flexible_cg
from neuralls.solver.monitoring import TraceMode

x, result = flexible_cg(A, b, trace_mode=TraceMode.FULL)

# Direct attribute access
residual_norms = result.iteration_history.residual_norms.to_list()
residuals = result.iteration_history.residuals.to_array()  # FULL mode only
solutions = result.iteration_history.solutions.to_array()  # FULL mode only
```

### 2. EventLog (Discrete Events)

Tracks discrete occurrences during solving following the **Event Sourcing pattern**.

**Event Types:**
- `CONVERGED`: Convergence criterion satisfied
- `BREAKDOWN`: Numerical breakdown detected
- `ORTHO_BREAKDOWN`: Orthogonalization breakdown

**Features:**
- Immutable event records with iteration numbers
- Query methods: `converged_at()`, `breakdown_at()`, `ortho_breakdown_at()`
- Efficient storage (events as occurrences, not boolean flags)

**Example:**
```python
x, result = flexible_cg(A, b)

# Query discrete events
if result.event_log.converged_at() is not None:
    print(f"Converged at iteration {result.event_log.converged_at()}")

# Check for breakdowns
if result.event_log.ortho_breakdown_at() is not None:
    print(f"Orthogonalization breakdown at {result.event_log.ortho_breakdown_at()}")
```

### 3. TraceMode Settings

Control monitoring granularity:

| Mode | Memory | Data Collected |
|------|--------|----------------|
| `DISABLED` | 0 bytes | None |
| `MINIMAL` | ~KB | Residual norms only |
| `FULL` | ~MB-GB | Norms + all vectors |

**Example:**
```python
from neuralls.solver.monitoring import TraceMode

# Minimal monitoring (default)
x, result = flexible_cg(A, b, trace_mode=TraceMode.MINIMAL)

# Full debugging
x, result = flexible_cg(A, b, trace_mode=TraceMode.FULL)

# No monitoring
x, result = flexible_cg(A, b, trace_mode=TraceMode.DISABLED)
```

### Key Design Principles

1. **Separation of Concerns**: Continuous data (IterationHistory) vs discrete events (EventLog)
2. **Direct Access**: `history.residual_norms` not `tracker.get_scalars(EventType.RESIDUAL_NORM)`
3. **Industry Patterns**: Follows Telemetry/Metrics + Event Sourcing standards
4. **Immutability**: All histories use frozen dataclasses with functional updates
5. **Type Safety**: Direct attributes more type-safe than enum-indexed dictionaries

---

## Orthogonalization Strategies

Three orthogonalization strategies are available for FlexibleCG, implementing different variants from the literature:

### 1. PeriodicRestartOrthogonalization (FCG with Periodic Restart)

**Algorithm:** FCG(m_max) from Notay (2000) - Flexible Conjugate Gradient with periodic restart

Implements the paper's recommended cyclic restart formula:
```
m_i = max(1, i mod (m_max + 1))
```

**Example with m_max = 10:**

| Iteration i | m_i = max(1, i mod 11) | Behavior |
|-------------|------------------------|----------|
| 0           | max(1, 0) = 1          | 1 direction |
| 1-10        | 1-10                   | accumulate history |
| 11          | max(1, 0) = **1**      | **RESTART** - discard old directions |
| 12-21       | 2-11 → clamp to 10     | rebuild history |

**Properties:**
- Periodic reset prevents accumulation of rounding errors
- Generally more cost-efficient per Notay (2000) Table 1 benchmarks
- Memory: O(m_max × n)
- **This is the paper's original FCG algorithm**

**Reference:** Notay (2000) Section 2, Algorithm 2.1

---

### 2. PeriodicRestartOrthogonalization with m_max=∞ (FCG(∞))

**Algorithm:** FCG(∞) - Full reorthogonalization against all history

Implements FCG with infinite orthogonalization window by setting `m_max=np.inf`. This is mathematically equivalent to setting m_max → ∞ in the periodic restart formula, which gives m_i = i.

**Properties:**
- Orthogonalizes against all previous directions
- Exact A-conjugacy (within numerical precision)
- Memory: O(k × n) where k = iteration count
- Used for ill-conditioned problems (e.g., Notay 2000 Table 1, Case 3: κ=1100)
- Uses Modified Gram-Schmidt for numerical stability

**Usage:**
```python
orthog = PeriodicRestartOrthogonalization(m_max=np.inf)
# Or via factory:
orthog = create_fcg_orthogonalization(m_max=-1)  # Converts -1 to np.inf
```

**Reference:** Notay (2000) Section 5.1, Table 1 Case 3

---

### 3. TruncatedGramSchmidt (Tr-FCG with Sliding Window)

**Algorithm:** Tr-FCG - Truncated FCG with pure sliding window (no periodic restart)

Implements truncated Gram-Schmidt with fixed window size:
```
m_i = min(i, m_max)
```

| Iteration i | m_i = min(i, 10) | Behavior |
|-------------|------------------|----------|
| 0           | 0                | none     |
| 1-10        | 1-10             | accumulate history |
| 11+         | 10               | sliding window, keeps last m_max |

**Properties:**
- Simpler implementation (no restart logic)
- Always orthogonalizes against last m_max directions
- Memory: O(m_max × n)
- **This is NOT the original FCG from the paper** (no periodic restart)

**Note:** This is a variant sometimes called "Tr-FCG" in the literature.

---

## Parameters

### ConjugateGradientSolver

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction_strategy` | `DirectionStrategy` | (required) | Strategy for computing search direction |
| `preconditioner` | `Preconditioner` | Identity | M⁻¹ operator |
| `convergence_criterion` | `IConvergenceCriterion` | Combined(rtol, atol) | Stopping criterion |
| `iteration_history` | `IterationHistory` | None | Continuous monitoring |
| `event_log` | `EventLog` | None | Discrete event logging |
| `trace_mode` | `TraceMode` | MINIMAL | Logging granularity |

### PeriodicRestartOrthogonalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m_max` | `int` | 10 | Maximum window size before periodic restart |
| `epsilon` | `float` | 1e-14 | Small denominator threshold |

### TruncatedGramSchmidt

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | 10 | Maximum sliding window size |
| `epsilon` | `float` | 1e-14 | Small denominator threshold |

### Solve Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rtol` | `float` | 1e-6 | Relative tolerance: ‖r‖/‖b‖ < rtol |
| `atol` | `float` | 1e-14 | Absolute tolerance: ‖r‖ < atol |
| `maxiter` | `int` | 10 * N | Maximum iterations (N = system size) |

---

## Migration from Old API

If you were using the old classes, here's how to migrate:

### Old: PreconditionedCGSolver

```python
# OLD (no longer works)
from neuralls.solver import PreconditionedCGSolver
solver = PreconditionedCGSolver(preconditioner=my_precond)
```

**New:**

```python
# NEW - Option 1: Use factory (recommended)
from neuralls.solver import pcg

x, result = pcg(A, b, preconditioner=my_precond)

# NEW - Option 2: Direct instantiation
from neuralls.solver import ConjugateGradientSolver, TwoTermRecurrenceStrategy

direction_strategy = TwoTermRecurrenceStrategy()
solver = ConjugateGradientSolver(
   direction_strategy=direction_strategy,
   preconditioner=my_precond,
)
```

### Old: FlexibleCGSolver

```python
# OLD (no longer works)
from neuralls.solver import FlexibleCGSolver
solver = FlexibleCGSolver(orthogonalization=orthog_strategy)
```

**New:**
```python
# NEW - Option 1: Use factory (recommended)
from neuralls.solver import flexible_cg
x, result = flexible_cg(A, b, m_max=10)

# NEW - Option 2: Direct instantiation
from neuralls.solver import ConjugateGradientSolver, OrthogonalizationDirectionStrategy
direction_strategy = OrthogonalizationDirectionStrategy(orthog_strategy)
solver = ConjugateGradientSolver(direction_strategy=direction_strategy)
```

See `MIGRATION.md` for complete migration guide.

---

## References

### Primary References

1. **Notay, Y. (2000)**. Flexible Conjugate Gradients. *SIAM Journal on Scientific Computing*, 22(4), 1444-1460.
   - **DOI:** [10.1137/S1064827599362314](https://doi.org/10.1137/S1064827599362314)
   - **Original FCG paper** - Defines Algorithm 2.1 (Section 2) with periodic restart formula
   - Analyzes truncation strategies and provides benchmark comparisons (Table 1)
   - Recommends periodic restart (m_i = max(1, i mod (m_max + 1))) as "generally more cost efficient"
   - Section 5.1 discusses FCG(∞) for ill-conditioned problems

2. **Hestenes, M. R., & Stiefel, E. (1952)**. Methods of Conjugate Gradients for Solving Linear Systems. *Journal of Research of the National Bureau of Standards*, 49(6), 409-436.
   - **DOI:** [10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)
   - Original conjugate gradient paper

### Comprehensive References

3. **Saad, Y. (2003)**. *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
   - **ISBN:** 978-0-89871-534-7
   - Chapter 9 covers CG methods and preconditioning
   - Section 9.5 discusses flexible variants

4. **Greenbaum, A. (1997)**. *Iterative Methods for Solving Linear Systems*. SIAM.
   - **ISBN:** 978-0-89871-396-1
   - Theoretical foundations of iterative methods
   - Analysis of A-conjugacy and orthogonalization

### Implementation Notes

- Our `ConjugateGradientSolver` implements a unified CG algorithm using Strategy pattern
- Our `TwoTermRecurrenceStrategy` matches scipy.cg's two-term recurrence behavior
- Our `OrthogonalizationDirectionStrategy` implements Notay (2000) Algorithm 2.1 via composition
- Our `PeriodicRestartOrthogonalization` implements the periodic restart formula from Section 2
- Our `PeriodicRestartOrthogonalization(m_max=np.inf)` implements FCG(∞) from Section 5.1
- Test cases in `tests/solver/conftest.py` reproduce Notay (2000) Table 1 conditions
