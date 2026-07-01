# Solver Module

The solver package contains the CG algorithms, preconditioners, monitoring
tools, and typed result models used across the repo. It is the cleanest
independent subsystem in the codebase.

## Start With The Public Entry Points

Most callers should begin with the factory functions:

```python
from neuralls.domain.solver import flexible_cg, pcg, scipy_cg
```

Use them in this order of complexity:

| Level | Entry point | Use when |
| --- | --- | --- |
| Basic | `pcg(...)` | the preconditioner is fixed and SPD |
| Basic | `scipy_cg(...)` | you want SciPy behavior with repo monitoring hooks |
| Advanced | `flexible_cg(...)` | the preconditioner is neural, variable, or non-linear |

## Practical Choice Guide

### `pcg`

Choose `pcg` for classical preconditioners such as:

- identity
- jacobi
- ilu
- icholesky or ic0

### `flexible_cg`

Choose `flexible_cg` when:

- the preconditioner changes per iteration
- the preconditioner is not safely modeled as fixed SPD
- you are benchmarking neural preconditioners

The current docs and examples use `m_max = 20` as the default flexible-CG
history setting.

## Package Map

- `factories.py`: recommended public entry points
- `conjugate_gradient.py`: unified CG solver classes
- `preconditioners/`: classical (Jacobi, ILU, IC0, ICholesky), AMG, neural, and scheduled preconditioners
- `strategies/`: direction, convergence, norms, orthogonalization
- `monitoring/`: iteration history, events, callback adapters
- `models/`: typed config, state, history, and result objects
- `utils/`: numerical helpers and export utilities

## Advanced Usage

Drop below the factory layer only when you need one of these:

- direct control over direction strategies
- explicit monitoring objects
- low-level preconditioner construction
- custom orthogonalization behavior

Typical advanced imports:

```python
from neuralls.domain.solver import ConjugateGradientSolver
from neuralls.domain.solver.strategies.direction import OrthogonalizationDirectionStrategy
from neuralls.domain.solver.strategies.orthogonalization import PeriodicRestartOrthogonalization
```

## Where The Solver Is Used

The rest of the repo consumes solver functionality through:

- generation trace strategies
- comparison workflows
- neural-preconditioner adapters

The DLKit-backed predictor adapter now lives under
`neuralls.platform.dlkit.predictor_adapter`. The neural preconditioner itself
still lives in
`neuralls.domain.solver.preconditioners.implementations.neural`, but it is not
re-exported from the aggregate preconditioner package to keep the solver layer
from depending on platform exports. That adapter speaks directly to the current
DLKit inference runtime under `dlkit.engine.inference`, so solver code remains
isolated from DLKit's infrastructure and configuration packages.

## Preconditioner Context

All preconditioners receive a `PreconditionerContext` (iteration, residual norm,
RHS norm) on every `apply()` call. There is no `isinstance` dispatch — the
context is always provided. `ContextualPreconditioner` was removed; use
`NonLinearPreconditioner` for preconditioners that change per iteration.

`ScheduledPreconditioner` uses that context to choose between primary and
fallback preconditioners. `start_iter` delays activation of the primary
preconditioner, and `limit_iters` bounds how many iterations the primary stays
active after that start. Before activation and after the optional limit, the
fallback preconditioner is used.

## AMG Preconditioner

`AMGPreconditioner` lives under
`neuralls.domain.solver.preconditioners.implementations.amg` and implements
`Preconditioner` directly (lazy hierarchy build). The extension points are:

- `CoarseningStrategy`: implement to add Ruge-Stüben or neural P/R coarsening
- `MultigridCycle`: implement to add W or F cycles
- `MultigridSmoother`: implement to add Gauss-Seidel or other smoothers

`extra_input_names` on `ExtraInputPredictorPort` is the single source of truth
for which arrays a neural predictor needs. `NeuralPreconditionerConfig` still
accepts an override in the comparison TOML for backward compatibility, but new
configs should omit it and let the model config declare its own inputs.
