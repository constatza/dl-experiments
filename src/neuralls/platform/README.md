# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, workflow config models, registry resolution, TOML loaders, and DLKit workflow bridging
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `tracking/`: MLflow run and client helpers
- `reporting/`: plotting, artifact staging, and inference output adapters
- `dlkit/`: DLKit-backed adapters for solver preconditioners and batch inference
- `caching.py`: directory hashing for workflow cache invalidation

## Semantic Difference

Platform code owns concrete integrations and side effects. A module belongs
here when it speaks a third-party API, reads or writes files, configures
runtime services, or serializes artifacts. It should not decide workflow order
or contain domain algorithms.

Platform also owns translation between repository-facing configuration data and
external runtimes. That includes workflow-model validation, path and tracking
resolution, DLKit-specific config bridging, and MLflow client operations.
These concerns stay here so composition can assemble workflows without owning
filesystem, environment, or third-party policy details.

The `dlkit/` package is the runtime adapter boundary. It hides predictor and
inference integration details behind local abstractions so solver and
application code depend on structural contracts rather than DLKit return-shape
quirks or registry helpers.

## Boundary

Platform code may depend on domain protocols and domain data structures, but it
should not own business rules or experiment orchestration.

Tracking helpers treat DLKit as the authoritative checkpoint artifact logger.
Workspace uploads therefore exclude the local `checkpoints/` tree and only
forward staged diagnostics/config artifacts, avoiding duplicate MLflow artifact
layouts such as `checkpoints/checkpoints/...`.
