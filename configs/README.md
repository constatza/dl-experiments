# Configuration Guide

The public CLI is case-oriented. One case config binds datasets, jobs,
comparisons, and experiment batches.

## Warning

The runnable job contract is now fully lower-case DLKit-native.

Hard-cut rules:

- no legacy monolithic job configs
- no `[[models]]` in case configs
- no uppercase runnable sections such as `[JOB]`, `[SESSION]`, `[TRAINING]`, `[MODEL]`, `[DATASET]`, or `[DATAMODULE]`

## Config Roles

### `configs/datasets/**/*.toml`

Dataset configs own:

- raw input paths
- generation strategy
- normalization and output settings
- processed dataset identity

There should be as many dataset configs as there are real dataset variants.
Datasets stay dataset-only; they do not own model architecture or training
policy.

### `configs/profiles/model/**/*.toml`

Model profiles are reusable lower-case DLKit model fragments.

`module_path` may be omitted when the model lives under the default `dlkit.nn`
surface.

They may contain:

- `[model]`
- optional `[data]` / `[data.module]` / `[[data.features]]` / `[[data.targets]]`
  — only when the data shape is genuinely coupled to the model architecture

They must not contain:

- `[run]`
- `[training]`
- `[tracking]`
- `[search]`

Most model profiles should stay thin (`[model]` only) and reference a shared
data profile from `configs/profiles/data/**` via `run.data` instead of
repeating the default data block. Only special model families should embed
custom runtime input structure directly in the model profile, for example:

- DeepONet branch/trunk inputs (`configs/profiles/data/deeponet-branch-trunk.toml`)
- conditional or FiLM-style models with multiple feature inputs (`configs/profiles/data/film-condition.toml`)
- PCA-specific runtime transforms used by only one model family

When a model family derives only shape-driven kwargs automatically, keep the
remaining architecture kwargs explicit in the profile. DeepONet profiles are the
main example: `branch_in_features`, `trunk_dim`, and `out_features` come from
data shapes, but `basis_dim` and the branch/trunk hidden widths belong in
`[model]`.

The DeepONet data profile uses `MinMaxScaler(dim = [0, 1])` before `Unsqueeze`
for vector-valued entries. That reduces value scale with one shared scalar range
per entry instead of per-coordinate normalization, so branch and target vectors
keep their internal relative magnitudes.

The checked-in model profiles pin `activation = "gelu"` wherever the target
network exposes an activation kwarg. `ScaleEquivariantSiren` is the exception:
its constructor does not accept a configurable activation.

### `configs/profiles/data/**/*.toml`

Shared data profiles for the common case: any number of model families with
identical data shape should point `run.data` at the same file here instead of
duplicating `[data]` in every model profile.

Current checked-in example:

- `array-default.toml` — the default `FlexibleDataset` + `ArrayDataModule`
  shape with no custom features/targets

### `configs/profiles/training/**/*.toml`

Training profiles are the small shared optimization-policy layer.

`default.toml` is the baseline. Every other profile should deviate from it
in exactly one named way — that's what the filename documents. Don't stack
multiple deviations into one profile; add a new single-deviation profile
instead.

They may contain:

- `[training.*]`

Current checked-in examples:

- `default.toml` — baseline: 300 epochs, `max_lr = 1e-3`, early stopping
  patience 30 / min_delta 1e-4
- `extended.toml` — deviates by epoch count: 600 epochs
- `high-max-lr.toml` — deviates by LR tuner ceiling: `max_lr = 1e-2`
- `strict-early-stopping.toml` — deviates by early stopping: patience 10,
  min_delta 1e-3

### `configs/jobs/**/*.toml`

Jobs are thin runnable entrypoints.

They own:

- `run.type`
- references to one model profile and one training profile
- optional inline `[search]` for search jobs
- optional `[experiment]` when a job needs a stable explicit name

They should not duplicate the shared training block unless a job has a genuine
job-specific override.

### `configs/cases/**/*.toml`

Case configs bind:

- dataset ids
- job ids
- experiment ids
- comparison ids
- case-level MLflow topology

## Case Anatomy

```toml
[mlflow]
tracking_uri = "http://localhost:5000"

[names]
training = "Train"
comparison = "Comparisons"

[[datasets]]
id = "train-dataset"
path = "../../datasets/train/45x15/gaussian-cg100.toml"

[[datasets]]
id = "benchmark-dataset"
path = "../../datasets/test/45x15/scaled-solutions.toml"

[[jobs]]
id = "factorized"
path = "../../jobs/ffnn/factorized.toml"

[[comparisons]]
id = "scaled"
matrix_dataset = "train-dataset"
rhs_dataset = "benchmark-dataset"

[[experiments]]
dataset = "train-dataset"
job = "factorized"
```

`[names].training` controls the MLflow experiment bucket for training runs.
`[names].comparison` controls the comparison bucket.

## Thin Job Example

```toml
[run]
type = "train"
seed = 42
precision = "64"
model = "../../profiles/model/ffnn/factorized.toml"
data = "../../profiles/data/array-default.toml"
training = "../../profiles/training/default.toml"
```

## Search Job Example

```toml
[run]
type = "search"
seed = 42
precision = "64"
model = "../../profiles/model/ffnn/ffnn.toml"
data = "../../profiles/data/array-default.toml"
training = "../../profiles/training/default.toml"

[search]
objective = "val_loss"
space."training.optimizer.default_optimizer.lr" = { type = "log_float", low = 1e-5, high = 1e-2 }
```

Search stays coupled to jobs, not training profiles. The job is the runnable
entrypoint, so the inline `[search]` block belongs there.

## Model Profile Example

```toml
[model]
name = "ScaleEquivariantFactorizedFFNN"
num_layers = 1

[data]
name = "FlexibleDataset"
batch_size = 256
pin_memory = true
shuffle = true

[data.module]
name = "ArrayDataModule"
```

## Composition Boundary

`neuralls` composes the final executable DLKit workflow by combining:

1. model defaults from `configs/profiles/model` and shared data defaults from `configs/profiles/data`
2. shared optimization policy from `configs/profiles/training`
3. run-mode and search intent from `configs/jobs`
4. runtime dataset injection from the selected dataset config
5. runtime-only workspace and tracking values

DLKit remains the validator for the final workflow object.

## Practical Rules

- create one dataset config per real dataset variant
- keep shared training policy small and reusable
- keep jobs thin
- only special model families should define custom `data.features` / `data.targets`
- keep case configs as registries and bindings, not payload containers

## MLflow And Paths

Case configs may define `[mlflow].tracking_uri` and optional artifacts
destination. Jobs and model profiles must not embed machine-specific
infrastructure paths.

Machine roots still come from the active neuralls profile:

```bash
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
```
