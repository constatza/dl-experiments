# Composition Module

The composition package owns wiring and config-driven assembly.

## Package Map

- `assignments/`: registry loading, assignment wiring, training, comparison, inference
- `comparison/`: single-run comparison assembly around application/domain logic
- `generation/`: config-driven dataset orchestration, dataset persistence wiring, and default tracing services
- `inference/`: inference data-loading composition helpers
- `preconditioners/`: config-to-preconditioner factory wiring (Identity, Jacobi, ILU, IC0, ICholesky, AMG, NeuralAMG, Neural)
- `tracking/`: tracking tag and run-spec assembly

## Semantic Difference

Composition is the only layer allowed to know both the abstract workflow shape
and the concrete adapter set needed to run it. If a module mostly wires config
models, ports, and runtime collaborators together, it belongs here. If it
starts doing reusable numerical work, it belongs in `domain`. If it starts
owning concrete IO or MLflow mechanics, that code belongs in `platform`.

## Boundary

Composition is where config models, platform adapters, workflow DTOs, and
domain services are connected. Entry modules that still assemble concrete
collaborators belong here rather than under `application`.

Composition assumes that platform has already resolved configuration and
runtime concerns into concrete values and adapters. Its job is to select the
right collaborators, construct workflow-local DTOs, and hand execution to
application or platform entrypoints without re-implementing path, environment,
or client policy.

Preconditioner assembly follows the same boundary. Platform config validates
shared schedule fields such as `start_iter`, `limit_iters`, and `fallback`;
composition maps those values into the domain `ScheduledPreconditioner` without
owning the iteration-switching policy.

For DLKit training, composition now consumes already-loaded lower-case DLKit
jobs. DLKit owns TOML parsing, profile references under `[run]`, section merge
order, and typed validation. Composition owns only runtime materialization:
dataset entry injection, workspace path patching, and tracking enablement on
top of the validated job object.

That materialization is split on purpose. `job_loader.py` is the only local
boundary that asks platform to load one typed DLKit job. `assembler.py` then
applies only the minimal mode-specific startup patching needed to create a
runnable assignment workspace, while `job_materializer.py` owns the later
training-only runtime patch sequence for dataset entries, dataloader defaults,
workspace callback wiring, and tracking enablement. Composition no longer
duplicates that patch policy by re-parsing TOML or by open-coding workspace
and tracking mutations in multiple call sites.

For DLKit-supervised training, composition is also the only layer allowed to
translate repository storage names into runtime dataset-entry names. That
translation is owned by a composition-level runtime dataset contract. The
current supervised bridge maps dataset `rhs` artifacts to feature entry `x`,
dataset `solutions` artifacts to target entry `y`, and the dataset `matrix`
artifact to auxiliary feature entry `matrix`. The storage format (`zarr` or
`npy`) is resolved in platform storage before composition turns those artifacts
into format-neutral resolved dataset-entry specs. Platform adapters then
translate those specs into concrete DLKit entries such as `NpyEntry` or
`ZarrEntry`. Domain terms such as `solutions` remain valid on disk, but
composition does not expose `solutions` as a runtime target alias.

Additional runtime model inputs are opt-in and come from exactly one source:
extra `[[data.features]]` declarations in the DLKit job or data profile. Composition
preserves their declaration order and binds them positionally to persisted
`parameters_0`, `parameters_1`, ... dataset artifacts. Model hyperparameters in
`[model]` are never treated as dataset inputs.

This bridge is intentionally declarative at the config boundary. DLKit job and
profile TOMLs remain the source of truth for runtime entry names, transforms,
and supported entry-routing metadata, while composition only patches in the
resolved on-disk paths and neutral runtime semantics at execution time.
DLKit-specific entry construction and optional metadata application stay in
platform code.

Training assembly keeps feature specs path-backed instead of eagerly opening
dataset arrays inside `neuralls`, but that does not guarantee fully lazy `.npy`
training. DLKit's current
`FlexibleDataset` still materializes non-lazy path entries during dataset
construction, even though `NpyEntry` can forward format-specific load kwargs
such as `mmap_mode`.

Prediction payload normalization follows the same rule. Composition accepts
DLKit's raw boundary output once, normalizes it into the canonical prediction
key `y_pred`, and then hands the normalized payload to diagnostics and artifact
staging. Downstream reporting code does not keep a fallback list of prediction
aliases.

Training, inference, comparison, and generation assembly all follow the same
rule: composition may decide which collaborators participate in a workflow, but
it must not absorb low-level IO mechanics, config normalization policy, or
tensor-level runtime behavior.

For MLflow naming, composition only propagates the resolved case-config names.
Training uses `names.training` and comparison uses `names.comparison`, with the
defaults owned by the case-config Pydantic models rather than composition-layer
constants.

Batch training adds one session-scoped parent run per case-config invocation.
That parent is identified by case-config path plus launch time and exists only
to group the per-assignment child runs. Comparison workflows do not add an
equivalent wrapper: their parent run identity stays defined by the dataset
selection encoded in each comparison entry.

Comparison parent-run summary metrics sanitize each preconditioner label into an
MLflow-safe metric-key segment before appending it under namespaced keys such as
`iterations/<preconditioner>`. Composition does not implement that MLflow policy
itself: it delegates metric-key sanitization, comparison-run metric logging, and
nested child-run writes to `platform.tracking`, while only assembling child tag
payloads and deciding when the logging happens.

Case-driven comparison selection is intentionally simple: one comparison entry
loads one system from its required `rhs_source`. Generated and raw sources use
the configured matrix dataset plus `matrix_index`, while dataset sources
delegate to platform storage to resolve one canonical `(matrix, rhs, lhs)`
triplet. When a dataset source omits `sample_index`, platform selects the first
STANDARD row and follows persisted `matrix_sample_index` to load the matching
matrix. Comparison does not inspect training/test splits or infer held-out
semantics from MLflow artifacts.

Comparison input preflight also stays out of composition. Workflow code invokes
platform-owned validation for matrix/RHS inputs rather than inspecting dataset
manifests directly. Preflight validates only the concrete artifacts required by
the chosen comparison mode: matrix-only validation for generated-RHS runs, and
matrix-plus-RHS validation for raw or dataset-backed RHS runs.
Likewise, enriched infrastructure failures from storage and tracking propagate
through composition unchanged; composition does not reformat low-level I/O
errors into user-facing strings.

Resolved comparison inputs are treated as first-class artifacts. Composition
stages the final `(A, b)` pair plus provenance metadata and delegates MLflow
logging of those artifacts to platform tracking helpers so future runs can
reconstruct the exact system that was compared.

Comparison model resolution treats one resolved MLflow `run_id` as the hard
boundary for checkpoint discovery. When downloaded run artifacts contain
multiple `.ckpt` files, composition canonicalizes byte-identical duplicate
copies and raises on distinct candidates instead of silently picking the first
path.
