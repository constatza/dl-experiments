# Solver Architecture

This package implements Flexible CG for non-SPD and time-varying preconditioners plus SciPy-backed CG baselines, organized into small SOLID components.

## Entry Points
- `graph-cg/src/solver/factories.py:flexible_cg(...)` - truncated Gram-Schmidt FCG with automatic event logging, callable or `LinearOperator` preconditioners, optional reorthogonalization, and trace capture.
- `graph-cg/src/solver/factories.py:preconditioned_cg(...)` - SciPy CG wrapper with matching `SolverResult`, residual histories, and `event_log` (preconditioner=None covers the baseline).
- `graph-cg/src/solver/comparison.py:run_cg_comparison(...)` - sweep helpers for scheduled preconditioners (limits, cadence, fallbacks).

## Architecture
- **Core solver** (`graph-cg/src/solver/fcg_solver.py`): `FlexibleConjugateGradientSolver` orchestrates the algorithm using `IterationState` plus `DirectionHistory`, restart logic, residual replacement, divergence detection, and optional trace capture (`capture_traces`, `capture_search_directions` store vectors in `SolverResult`).
- **Direction strategies** (`graph-cg/src/solver/direction_strategies.py`): `TruncatedOrthogonalDirection` (default) performs periodic Gram-Schmidt with `m_max` windowing and `history_limit`; `FlexibleCGDirection` keeps the two-term recurrence for low-memory variants. Reorthogonalization hooks live in `graph-cg/src/solver/reorthogonalization.py` (`Full`, `Partial`, `Selective`) and are pluggable via the `reorthogonalize` parameter.
- **Preconditioners** (`graph-cg/src/solver/preconditioners.py`): `Preconditioner` base plus wrappers. `CallablePreconditioner` adapts `f(r)`, `f(r, ctx)`, or `f(ctx)` signatures; `LinearOperatorPreconditioner` wraps SciPy operators; `IdentityPreconditioner` is the default. `IterationContext` (in `graph-cg/src/solver/info.py`) exposes iteration, residual, solution, matrix, and rhs for context-aware preconditioners or helpers.
- **Diagnostics/helpers** (`graph-cg/src/solver/helpers.py`, `graph-cg/src/solver/state.py`): `convergence_check`, `curvature`, `step_length`, `beta_update`, `direction_update`, and `residual_management` enforce restart/breakdown rules, periodic true residual recomputation (`m_replacement=50` default), and divergence recovery. `IterationState` tracks residual history, restarts, breakdown/divergence flags, and maintenance counters; `DirectionHistory` holds p/q buffers with windowed truncation.
- **Logging/results** (`graph-cg/src/solver/event_logging.py`, `graph-cg/src/solver/info.py`): `VectorLogger` event-sources residuals, solutions, search directions, and norms. Factories attach a logger by default and surface it as `SolverResult.event_log`; `SolverResult` mirrors SciPy fields (`info`, `status`, `niter`, `residual_history[_abs]`, `breakdown`, `stopping_criterion`) plus optional vector traces.

## Default behavior and tuning
- Convergence: `||r|| <= max(rtol * ||b||, atol)` with `rtol=1e-6`, `atol=1e-14` (`stopping_criterion="fixed_iterations"` bypasses it).
- FCG: truncated Gram-Schmidt with periodic reset (`m_max` default 10, `history_limit` 200), restarts on small or negative curvature or beta spikes, residual recomputation every `m_replacement` iterations, divergence guard `gamma_div`.
- SciPy CG wrapper: `preconditioned_cg` forwards `tol`/`max_iter` aliases, wraps callables into `LinearOperator`s (or `None` baseline), and records histories through `ResidualRecorder` while returning the same `SolverResult` surface.

## Quick usage
```python
import numpy as np
from numpy.typing import NDArray

from src.solver.factories import flexible_cg
from src.solver.info import IterationContext
from src.solver.preconditioners import CallablePreconditioner

def neural_precond(residual: NDArray, ctx: IterationContext) -> NDArray:
    features = build_features(residual, ctx.solution)
    return model.predict(features)

x, result = flexible_cg(
    A,
    b,
    preconditioner=CallablePreconditioner(neural_precond),
    m_max=10,
    reorthogonalize=None,  # swap in create_reorthogonalization_strategy(...) if needed
    capture_traces=True,
)
residual_norms = result.event_log.get_history("residual_norm")
```
