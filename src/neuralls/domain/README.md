# Domain Module

The domain package contains the pure computational core of `neuralls`.

## Package Map

- `generation/`: strategy-driven dataset payload generation
- `solver/`: CG solvers, monitoring, strategies, and preconditioners
- `analysis/`: pure numerical diagnostics used by higher layers

## Semantic Difference

Domain code explains the mathematics and workflow invariants of the project.
If a module needs only arrays, pure data models, and algorithmic rules, it
belongs here. The moment it starts parsing config files, opening MLflow runs,
or deciding artifact locations, it has crossed into `platform` or
`composition`.

## Boundary

Domain code depends only on `neuralls.shared`. It does not load configs,
resolve filesystem layout, start MLflow runs, or persist artifacts. When
domain output must be written, it is returned as typed payloads and handled by
`neuralls.composition` plus `neuralls.platform`.
