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

The DLKit integration now maps cleanly onto DLKit's current layer split:
`dlkit.infrastructure.*` for config and IO, `dlkit.engine.*` for runtime
training and inference entrypoints, and `dlkit.domain.*` for model, metric,
and loss namespaces referenced by workflow TOMLs.

Workflow config path expansion now lives at the config-model boundary.
`NeurallsSettings` provides required raw, processed, and output roots after
profile resolution from `~/.config/neuralls/config.toml` plus any explicit env
overrides, and `ConfigContext` passes those roots into Pydantic validators so
workflow TOMLs can use `${NEURALLS_*}` placeholders without loader-side
normalization. User-home expansion for workflow config paths, profile roots,
and runtime root overrides explicitly prefers `HOME` before platform defaults
so `~` resolves consistently across Windows and POSIX test environments.
DLKit-specific workflow assembly is isolated in `config/dlkit_bridge.py`.

Path and MLflow resolution now have a single policy home:
`config/resolution.py`. That module is pure and owns user-home expansion,
config-relative path resolution, sqlite/file URI normalization, MLflow
topology derivation, and the immutable `PathContext` / `MlflowPaths` value
objects. Path normalization delegates to DLKit where possible: local
filesystem roots use DLKit's resolver stack, while SQLite/file URI handling
uses DLKit's local URI resolver. `neuralls` keeps only repo-specific policy
such as placing MLflow artifacts under `mlartifacts/`.
On Windows, mapped-drive and UNC roots are preserved as authored rather than
resolved through `Path.resolve()`, so profiles and `${NEURALLS_*}` placeholders
do not collapse `M:\...` roots into UNC server paths during config loading.
On POSIX hosts those Windows-only roots are still rejected early.

Runtime side effects stay out of the resolution hub. Temporary MLflow env-var
scoping lives in `tracking/environment.py`, while `tracking/mlflow.py` owns
MLflow client calls and run lifecycle operations. Model-config-specific MLflow
validation lives separately in `config/model_mlflow.py` so path policy does not
accumulate unrelated responsibilities.

Top-level case configs now accept only `[[experiments]]` for registry-backed
entries. The singular `[[experiment]]` table is rejected instead of being
silently normalized.

Current DLKit compatibility note:

- model configs still use `dlkit.nn` as the public-facing namespace
- advanced model configs may target concrete DLKit submodules such as
  `dlkit.domain.nn.spectral` when the installed top-level namespace does not
  re-export those classes
- local loaders normalize model TOML into workflow-specific DLKit config models
- MLflow logged-model URI construction is handled locally in `tracking/`
  because the installed DLKit package no longer exports those registry
  convenience helpers
- DLKit output-shape normalization is private to `platform/dlkit`, where both
  solver and inference adapters depend only on a structural contract:
  prediction results must expose a `predictions` tensor. No legacy tensor or
  tuple return shapes are supported in `neuralls`.

## Boundary

Platform code may depend on domain protocols and domain data structures, but it
should not own business rules or experiment orchestration.
