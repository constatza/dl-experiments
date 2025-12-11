# Solver Configuration

Solver parameters now live outside model configs. Each TOML file in this directory describes:

- `[general]`: global tolerances and iteration limits shared by all solvers (`rtol`, `atol`, `max_iterations`, `stopping_criterion`). Optional paths `matrix` and `rhs` let you override the system used for comparisons without touching CLI flags.
- `[[solvers]]`: the set of solvers/preconditioners to run. `name` is a display label; `type` must map to an implementation in the codebase (e.g., `none`, `identity`, `jacobi`, `ilu`, `pca`, `neural`). Any additional keys are treated as arguments for that solver (`limit_iters`, `apply_every`, `first_n`, `fallback`, etc.).
- `[data_generation]`: optional normalization settings for generated systems.

Example:

```toml
[general]
rtol = 1e-6
atol = 1e-14
max_iterations = 100
stopping_criterion = "residual_norm"
# Optional overrides for comparison input
# matrix = "/abs/path/to/normalized.npz"
# rhs = "/abs/path/to/normalized.npz"

[[solvers]]
name = "none"
type = "none"       # baseline SciPy CG without preconditioning

[[solvers]]
name = "jacobi"
type = "jacobi"

[[solvers]]
name = "neural"
type = "neural"
limit_iters = 5     # apply neural preconditioner for first 5 iterations
fallback = "jacobi" # fallback preconditioner after limit_iters
apply_every = 1

[data_generation]
normalize = "matrix"
```

Usage:
- `compare_methods.py` accepts `--solver-config` (defaults to `solver-configs/default.toml`).
- Workflow comparison tasks pass the same solver config to decide which solvers are executed and how they are labeled in reports.
