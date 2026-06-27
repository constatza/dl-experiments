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

Case-config auto naming also lives at this boundary. Auto-generated experiment
ids are model-first (`{model_id}-{dataset_id}`), and auto-generated experiment
display names follow the same order (`{model_label} | {dataset_label}`).
Auto-generated comparison display names stay dataset-defined: they resolve to
the matrix dataset label alone when matrix and RHS come from the same dataset,
or to `{matrix_label} | {rhs_label}` when they differ. Explicitly configured
comparison display names still override the generated label.

The DLKit dataset bridge stays generic. Platform helpers construct and patch
DLKit-native dataset entries with the names supplied by composition, but
platform does not own the canonical runtime naming policy itself. Storage-layer
artifact families such as `rhs.zarr`/`rhs.npy`, `solutions.zarr`/`solutions.npy`,
and `matrix.zarr`/`matrix.npy` stay in platform storage, while the composition
dataset contract decides which runtime entry names those artifacts map to.
Training artifact resolution preserves those on-disk sources as path-backed
dataset inputs, and platform adapters translate resolved entry specs into
concrete `NpyEntry` / `ZarrEntry` objects instead of eagerly converting whole
datasets into `ValueEntry` payloads.

That change improves separation of concerns and keeps format-specific loading
policy inside DLKit, including `NpyEntry` support for `mmap_mode`. It does not
by itself guarantee fully lazy `.npy` training because DLKit's current
`FlexibleDataset` still materializes non-lazy path entries during dataset
construction.

The `dlkit/` package is the runtime adapter boundary. It hides predictor and
inference integration details behind local abstractions so solver and
application code depend on structural contracts rather than DLKit return-shape
quirks or registry helpers.
Solver-side DLKit predictors must preserve fitted checkpoint transforms during
load so transform-aware models such as PCA-preprocessed preconditioners receive
inputs in the feature space they were trained on.

`DLKitPredictor` exposes a `required_inputs: tuple[str, ...]` property so that
the solver layer can derive which extra arrays a neural model needs without
consulting the comparison TOML. The DLKit model config is the single source of
truth; the comparison TOML's `extra_input_names` field is an optional override
kept for backward compatibility.

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
artifact-layout knowledge rather than workflow sequencing.
Generated-RHS workflows therefore validate only the matrix artifact they
actually consume, while dataset-backed RHS workflows validate both matrix and
RHS artifacts.
The same boundary owns filesystem and write-failure enrichment for dataset
artifacts so CLI callers receive operation- and path-specific diagnostics
without importing storage policy into composition.

Dataset storage is split by responsibility:
- `storage/manifest.py`: typed dataset manifest dataclasses and JSON serialization
- `storage/generation_formats.py`: generation-time `zarr` and `npy` writers/accumulators
- `storage/dataset_readers.py`: manifest-driven read helpers and explicit resolved dataset contracts

Safe comparison selection relies on manifest-declared metadata artifacts stored
in the dataset's native format. Mature datasets may expose:

- `rhs_kind`
- `target_kind`
- `matrix_sample_index`

Readers derive safe RHS candidates from `rhs_kind`; they do not trust ad hoc
filenames or stored allowlists. Storage owns persistence and artifact
resolution, while the compact integer encoding/decoding boundary lives in
shared pure codecs so domain and composition code do not depend on platform.
When datasets also expose `matrix_sample_index`, platform persists and reads
that binding as provenance metadata only. Comparison workflows may still mix
any selected matrix sample with any semantically allowed RHS row unless a
higher-level workflow adds a stricter pairing policy.

Case-driven comparison sample selection stays explicit and deterministic.
`ComparisonRegistryEntry.matrix_index` and `rhs_index` default to `0`, and the
workflow uses those indices directly when selecting one matrix/RHS system from
the comparison dataset. Platform does not infer held-out semantics from
training runs or MLflow split artifacts.

Training artifact persistence also stays generic: platform storage writes the
already-normalized numpy payload it receives from composition without
reintroducing DLKit prediction-key fallback logic.
