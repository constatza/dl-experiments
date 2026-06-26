# Configuration Guide

The public CLI is case-oriented. One case config drives the batch workflows for
dataset generation, training, full execution, and comparison.

This checkout is pinned to CUDA 13.0. Run `uv sync` once for the project
environment, then use plain `uv run neuralls ...` commands.

`SESSION.precision` must use Lightning-compatible precision names. Use `64`,
`32`, `16`, `bf16`, `16-mixed`, or `bf16-mixed`; do not use dtype strings such
as `float64`.

## Start With The Right File

Choose the config type that matches the case you want to run:

- `datasets/train/<system>/*.toml`: training dataset configs (gaussian-cg{N}, solutions-cg{N}, and any single-matrix variants)
- `datasets/test/<system>/*.toml`: comparison/test dataset configs (scaled-solutions, sparse-rhs, and any non-Gaussian benchmarks)
- `models/<family>/*.toml`: model architecture and trainer setup (shared across systems)
- `cases/<system>/*.toml`: case configs that tie datasets, models, comparisons,
  MLflow, and experiment ids together

Current systems: `45x15`, `45x15randomE`, and `93x31`. Each system has case
configs named `{model}-{dataset}.toml` with an optional `-pca` suffix for PCA-transform
variants where that family exists. `evaluate-all.toml` runs the curated candidate models;
`ffnn-mixed.toml` runs FFNN on all dataset variants; `films.toml` and
`deeponets.toml` split the `45x15randomE` parameter-conditioned cases by FiLM
and DeepONet family; `solutions-pca.toml` remains specific to the plain
`45x15` system. `factorized-single-matrix.toml` adds a dedicated
`45x15randomE` case for the one-matrix CG-100 dataset and keeps only the
scale-equivariant factorized models. Its benchmark datasets also use that same
fixed `45x15randomE` matrix for `scaled_solutions` and `sparse_rhs` tables.
The plain `45x15randomE/factorized.toml` case now uses `45x15randomE` benchmark
datasets too, so every explicit comparison matrix in the non-conditional
factorized family stays within the `45x15randomE` training matrix family.
Gaussian benchmark comparisons now reuse the corresponding `gaussian-cg1`
training dataset instead of carrying duplicate `gaussian-rhs` TOMLs. The
`45x15randomE` case family also reuses the existing `45x15`
`scaled-solutions` benchmark dataset rather than defining a separate solution
archive. Its training datasets also read per-sample E vectors from
`${NEURALLS_RAW_DIR}/SpectralData/45x15randomE/stiffness/*_YoungModuli_E1_E2_E3_E4.txt`.
When you need a one-matrix `45x15randomE` training dataset, use a dataset TOML
with a concrete `source.matrix_path` and omit both `enumerate_by` and
`parameters_paths`; `configs/datasets/train/45x15randomE/gaussian-cg100-single-matrix.toml`
is the reference example.

Model families (all `ScaleEquivariant*`, `module_path = "dlkit.nn"`):
- `ffnn/`: `ScaleEquivariantFFNN` — plain skip residual FFNN
- `film/`: `ScaleEquivariantFiLM{,Embedded}FFNN` — parameter-conditioned FiLM FFNN variants using `hidden_size` and `num_layers`
- `deeponet/`: `FFNNDeepONet`, `EmbeddedDeepONet` — branch/trunk operator models driven by `query`
- `embedded/`: `ScaleEquivariantEmbedded{Factorized,SPD,SPDFactorized}FFNN` — embedded square-output; current factorized case configs use the regular embedded-factorized variant
- `symmetric/`: `ScaleEquivariant{SPD,Factorized}FFNN` — non-embedded square-output; current factorized case configs do not select the SPD-factorized variants
- `spectral/`: `ScaleEquivariant{Siren,FourierFeatureNetwork,ModifiedMLP}` — sine/Fourier networks

## Recommended Progression

### 1. Configure one machine profile

```bash
uv run neuralls config init
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
```

### 2. Generate datasets

```bash
uv run neuralls generate /path/to/case.toml
uv run neuralls generate-single /path/to/dataset.toml \
  --case-config /path/to/case.toml
```

`neuralls generate <case.toml>` only builds the case `[[datasets]]` entries.
It does not infer or materialize standalone benchmark datasets referenced in
`[[comparisons]]` entries.

### 3. Train the case batch

```bash
uv run neuralls train /path/to/case.toml
```

### 4. Run or compare the same case

```bash
uv run neuralls run /path/to/case.toml
uv run neuralls compare /path/to/case.toml
```

If one comparison profile depends on a test dataset, build it explicitly first:

```bash
uv run neuralls generate-single configs/datasets/train/45x15/gaussian-cg1.toml \
  --case-config configs/cases/45x15/evaluate-all.toml
```

When a benchmark dataset is missing, `neuralls compare <case.toml>` now fails
that comparison before opening an MLflow run, reports the missing processed
path, and continues the rest of the batch.

## Case Anatomy

Each `case-*.toml` file is a case config for one experiment family.

```toml
[mlflow]
tracking_uri = "http://localhost:5000"

[names]
training = "Train"
comparison = "Comparisons"

[[datasets]]
id = "my-dataset"
path = "datasets/my-dataset.toml"

[[datasets]]
id = "my-bench-dataset"
path = "datasets/my-bench-dataset.toml"

[[models]]
id = "my-model"
path = "models/<family>/my-model.toml"

[[comparisons]]
id             = "my-comparison"
matrix_dataset = "my-bench-dataset"
rhs_dataset    = "my-bench-dataset"

[[experiments]]
id      = "my-dataset-my-model"
dataset = "my-dataset"
model   = "my-model"
```

`[names].training` controls the MLflow experiment bucket used for training
runs, and `[names].comparison` controls the MLflow experiment bucket used for
comparison runs. If you omit `[names]`, the defaults come from the case-config
model: `"Train"` for training and `"Comparisons"` for comparison.

## What Lives In Each Config

### Dataset configs

Dataset configs define:

- raw matrix and optional archive paths
- generation strategy blocks
- normalization and output settings
- optional test-set metadata

Comparison-matrix invariant:
- For every `cases/<problem>/` family, `[[comparisons]].matrix_dataset` must reference a training dataset from that same case/problem family.
- Benchmark/test datasets may still be used as `rhs_dataset`, but they must not be the source of the comparison matrix.
- If a case trains on a single fixed matrix, every comparison in that case must use that same fixed-matrix training dataset as `matrix_dataset`.

### Model configs

Model configs define:

- DLKit model module and hyperparameters
- trainer, loss, and optimizer-policy settings
- the `[DATASET]` dataset class name only; do not add removed legacy keys such as `memmap_cache`
- either `TRAINING.optimizer.default_optimizer` / `default_scheduler` or
  staged optimization under `TRAINING.optimizer.stages`
- checkpoint callback naming

Model configs keep `module_path = "dlkit.nn"` as the user-facing entrypoint.
`MODEL.name` must match the target class name exactly; kwargs stay flat under
`[MODEL]` and are forwarded through DLKit's `ModelComponentSettings`
filtering, so they must match the target constructor signature exactly.
`TRAINING.loss_function.name` and each `TRAINING.metrics[].name` must match
the current DLKit export names exactly, for example
`relative_vector_norm_loss` and `RelativeVectorNormError`.
For constant-width FFNN variants, "constant width" means the hidden body uses
the model's built-in width policy. Do not add an explicit `size` or
`hidden_size` override unless the DLKit constructor for that exact class
accepts it.

Optimizer sections may omit per-optimizer kwargs when DLKit defaults are
acceptable. Scheduler settings should still be written explicitly unless the
case intentionally wants DLKit's scheduler defaults.

Model configs use canonical DLKit syntax: the default scheduler lives
under `TRAINING.optimizer.default_scheduler`, and any staged program lives
under `TRAINING.optimizer.stages`.

For the local supervised workflows, model configs must route the label through
`TRAINING.loss_function.target_key = "targets.y"`. When `[[DATASET.targets]]`
placeholders are present, use the canonical runtime target name `y`. Keep
domain/storage names such as `solutions` for dataset files and dataset ids, not
for runtime DLKit target aliases. Runtime naming comes from the composition
dataset contract, while on-disk artifact names come from `[output].dataset_format`
and resolve through the dataset manifest as `rhs.{npy|zarr}`,
`solutions.{npy|zarr}`, `matrix.{npy|zarr}`, and optional `parameters_*.{npy|zarr}`.
Training diagnostics also consume one canonical prediction key, `y_pred`, after
composition normalizes DLKit's raw prediction payload once at the training
boundary.

### Comparison entries

`[[comparisons]]` entries in a case config define:

- `matrix_dataset` and `rhs_dataset`: ids from the case `[[datasets]]` registry
- optional `method` path to a methodology override TOML with solver params and preconditioners
- optional `display_name`, `rhs_index`, `matrix_index`

Shared solver parameters live in `[comparison_defaults]`. A `method` override
can selectively replace preconditioners or solver tolerances for a single entry.

Datasets referenced by `[[comparisons]]` must be materialised with
`neuralls generate-single` before comparison runs if they are not part of the
training dataset batch.

Case-driven comparison sample selection is explicit. Each `[[comparisons]]`
entry uses `matrix_index` and `rhs_index` to choose one system from the
comparison datasets; both default to `0` when omitted. The comparison workflow
does not infer held-out-only evaluation from training runs or split artifacts.
If a specific sample matters, set the indices directly in the case config.

## MLflow And Paths

Model configs do not define their own `[MLFLOW]` block. Runtime tracking
settings come from the selected case config or the execution environment.

Model configs must use DLKit's canonical workflow syntax directly. `neuralls`
does not translate top-level `[OPTIMIZATION]` or infer optimization mode from
`OPTUNA.enabled`. To run Optuna, set `SESSION.workflow = "optimize"` explicitly.
Keeping `workflow = "train"` with `OPTUNA.enabled = true` still executes a
normal training run.

Optuna search spaces live under `[OPTUNA.model]`. Numeric ranges use
`{low=..., high=..., step=...}` and categorical sweeps use
`{choices=[...]}`. For example, layer-count candidates `1`, `6`, and `9`
must be written as `num_layers = {choices = [1, 6, 9]}`.

## Machine Roots

Case configs contain experiment structure only. Machine-specific roots live in
the user config directory and are managed with `neuralls config`.

Set up once per machine:

```bash
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
```

Select a named profile at runtime with `--profile NAME` or
`NEURALLS_PROFILE=NAME`.

Profiles provide three roots:

- `raw_dir`: raw matrix and archive inputs
- `processed_dir`: processed datasets used by training and comparison
- `output_dir`: MLflow, checkpoints, figures, and reports

Explicit `--env-file` or `NEURALLS_ENV_FILE` still override the active profile
for one invocation. `neuralls` does not auto-discover `.env` or `.env.local`.

Local path normalization and sqlite URI handling delegate to DLKit's
`PathResolver` and local URI helpers. Relative paths are resolved against the
config file directory instead of assuming a Unix-only working directory layout.

## What To Read Next

- [`../README.md`](../README.md) for the end-to-end workflow
