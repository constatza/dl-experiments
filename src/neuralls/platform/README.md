# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, repo-owned config models, registry resolution, TOML loaders, and DLKit workflow bridging
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `tracking/`: MLflow run and client helpers
- `reporting/`: plotting, artifact staging, and inference output adapters
- `dlkit/`: DLKit-backed predictor adapter
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

Repo-owned config path expansion now lives at the config-model boundary.
`NeurallsSettings` provides required raw, processed, and output roots after
profile resolution from `~/.config/neuralls/config.toml` plus any explicit env
overrides, and `ConfigContext` passes those roots into Pydantic validators so
checked-in TOML files can use `${NEURALLS_*}` placeholders without loader-side
normalization. DLKit-specific workflow assembly is isolated in
`config/dlkit_bridge.py`.

Path normalization now also delegates to DLKit where possible: local
filesystem roots use `dlkit.infrastructure.io.PathResolver`, while
SQLite/file URI handling uses DLKit's local URI resolver. `neuralls` keeps
only repo-specific policy such as placing MLflow artifacts under
`mlartifacts/`.

Top-level case configs now accept only `[[experiments]]` for registry-backed
entries. The singular `[[experiment]]` table is rejected instead of being
silently normalized.

Current DLKit compatibility note:

- model configs still use `dlkit.nn` as the public-facing namespace
- local loaders normalize model TOML into workflow-specific DLKit config models
- MLflow logged-model URI construction is handled locally in `tracking/`
  because the installed DLKit package no longer exports those registry
  convenience helpers

## Boundary

Platform code may depend on domain protocols and domain data structures, but it
should not own business rules or experiment orchestration.
