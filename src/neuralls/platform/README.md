# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, workflow config models, registry resolution, TOML loaders, and DLKit workflow bridging
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `tracking/`: MLflow run helpers, naming/query policy, workflow topology resolution, and client adapters
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

The DLKit dataset bridge stays generic. Platform helpers construct and patch
DLKit-native dataset entries with the names supplied by composition, but
platform does not own the canonical runtime naming policy itself. Storage-layer
artifact names such as `rhs.npy`, `solutions.npy`, and `matrix_coo/` stay in
platform storage, while the composition dataset contract decides which runtime
entry names those artifacts map to.

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

MLflow-specific policy also belongs here: safe metric-key sanitization, search
filter escaping, workflow tracking-environment resolution, and comparison-run
metric logging all stay under `platform.tracking` so orchestration code does
not reimplement third-party rules.
When runtime `MLFLOW_TRACKING_URI` or `MLFLOW_ARTIFACT_URI` values are already
exported, platform tracking helpers preserve them verbatim instead of
re-normalizing them against the local operating system.

Storage validation owns concrete dataset-layout checks. Comparison matrix/RHS
preflight belongs under `platform.storage` because it depends on manifest and
sparse-pack layout knowledge rather than workflow sequencing.

Training artifact persistence also stays generic: platform storage writes the
already-normalized numpy payload it receives from composition without
reintroducing DLKit prediction-key fallback logic.
