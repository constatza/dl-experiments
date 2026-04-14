# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: config models (including workspace and preconditioner), registry resolution, and TOML loaders
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `tracking/`: MLflow run and client helpers
- `reporting/`: plotting, artifact staging, and inference output adapters
- `dlkit/`: DLKit-backed predictor adapter over `dlkit.engine.inference`
- `caching.py`: directory hashing for workflow cache invalidation

## Semantic Difference

Platform code owns concrete integrations and side effects. A module belongs
here when it speaks a third-party API, reads or writes files, configures
runtime services, or serializes artifacts. It should not decide workflow order
or contain domain algorithms.

The DLKit integration now maps cleanly onto DLKit's current layer split:
`dlkit.infrastructure.*` for config and IO, `dlkit.engine.*` for runtime
training and inference entrypoints, and `dlkit.domain.*` for model, metric,
and loss namespaces referenced by checked-in configs.

## Boundary

Platform code may depend on domain protocols and domain data structures, but it
should not own business rules or experiment orchestration.
