# dl-experiments

This monorepo hosts experimentation sandboxes for structured linear solvers and
bio-signal models. Most day-to-day work currently lives under `graph-cg/`, which
contains tooling to collect datasets, train neural preconditioners, and benchmark
conjugate-gradient variants.

## Key entry points

- `graph-cg/collect_data.py`: materialise datasets defined via TOML templates in
  `graph-cg/data-configs/`.
- `graph-cg/generate_data.py`: synthesise matrices / RHS pairs for training and
  evaluation.
- `graph-cg/train_model.py`: train FFNN or GNN checkpoints specified by configs in
  `graph-cg/configs/`.
- `graph-cg/compare_methods.py`: run flexible CG comparisons across classical and
  neural configurations. Defaults spin up Jacobi / ILU baselines, neural-only
  preconditioning, neural warm-starts, and the combined warm-start + neural
  preconditioner while seeding CG with the warm-start output.

All orchestration scripts should be launched through `uv run python …` to ensure we
share the same dependency environment.

## Neural comparison workflow

`compare_methods.py` now builds a configurable combination plan:

1. Warm-start only runs (`neural_warm_start`, etc.)
2. Classical preconditioners without warm-starts (Jacobi, ILU)
3. Neural preconditioner alone (`neural`)
4. Neural warm-start + neural preconditioner

Additional tuples can be added at runtime with repeated `--combo` arguments in the
form `WARM:PRECONDITIONER[:HELPER]`.

When a data config provides `test.solutions_path`, the script derives the RHS via
`b_test = A @ x_test` before running comparisons, ensuring all methods evaluate the
same system. Each run records its initial guess so downstream consumers can inspect
the warm-start seed actually used by CG.

## Repository hygiene

Large artifacts (datasets, checkpoints, figures) remain under `/data/projects/graph-cg`
and per-project `output/` folders. Do not modify `.venv/` or `.ruff_cache/`, and keep
changes focused within the workspace roots provided by the harness.
