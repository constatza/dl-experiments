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
- `preconditioners/`: classical, neural, and scheduled preconditioners
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
from depending on platform exports.
