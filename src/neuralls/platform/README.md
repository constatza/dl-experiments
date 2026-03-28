# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: config models, registry resolution, and TOML loaders
- `storage/`: filesystem, workspaces, and storage-facing helpers
- `tracking/`: MLflow run and client helpers
- `reporting/`: plotting, artifact staging, and inference output adapters
- `dlkit/`: DLKit-backed predictor adapter

## Semantic Difference

Platform code owns concrete integrations and side effects. A module belongs
here when it speaks a third-party API, reads or writes files, configures
runtime services, or serializes artifacts. It should not decide workflow order
or contain domain algorithms.

## Boundary

Platform code may depend on domain protocols and domain data structures, but it
should not own business rules or experiment orchestration.
