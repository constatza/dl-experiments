# Configuration Guide

The public CLI is case-oriented. One case config binds datasets, jobs,
comparisons, and assignment batches.

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
- `array-small-batch.toml` — the same array data shape with `batch_size = 32`
- `deeponet-branch-trunk.toml` — maps the primary dataset feature to
  `branch` and the first parameter stream to `trunk`
- `film-condition.toml` — maps the primary dataset feature to `x` and the first
  parameter stream to `condition`

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
  patience 30 / min_delta 1e-4, LR plateau patience 7 / threshold 1e-3
- `conservative-plateau.toml` — AdamW (`lr=1e-3`, `weight_decay=1e-2`)
  with batch size 32 jobs and LR plateau patience 10 / cooldown 5 / threshold 1e-4
- `extended.toml` — deviates by epoch count: 400 epochs
- `extended-conservative-plateau.toml` — 400 epochs with LR plateau patience
  10 / cooldown 5 / threshold 1e-4
- `limited-200.toml` — deviates by epoch count: 200 epochs
- `high-max-lr.toml` — deviates by LR tuner ceiling: `max_lr = 1e-2`
- `debug-overfit.toml` — 500 epochs, checkpointing disabled, `overfit_batches = 1`,
  and no early-stopping callback for expected validation divergence
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
- assignment ids, inferred when omitted
- comparison ids, inferred when omitted
- case-level MLflow topology

The 45x15 `default.toml` case compares the pure CG-N training dataset across Scale
Equivariant Embedded FFNN and Scale Equivariant Constant Width FFNN variants.
The 93x31 `default.toml` case uses the newer embedded FFNN families: embedded
factorized, embedded hyper, and embedded MoE. The 45x15randomE `default.toml`
case is parametric — the underlying problem is a family of ~100 stiffness
matrices with randomized Young's moduli (E1-E4), not one fixed matrix — and
uses DeepONet variants so the residual/RHS stream is the `branch` input and the
Young-modulus parameter stream is the `trunk` input. These CG cases include
classical identity/Jacobi/IC0, AMG, and dataset-backed POD-2G preconditioners,
with `gaussian-cg1`, `gaussian-cg10`, and `gaussian-cg50` datasets serving as
POD snapshot sources. Every randomE dataset uses the matrix glob
`45x15randomE/stiffness/*_subdomain_1_Kaa.txt`, `enumerate_by = "name"`, and
the matching Young-modulus parameter glob.

**45x15randomE train/eval matrix split**: because the same ~100-matrix family
would otherwise back both training (and the pod2g-\* POD snapshot datasets)
and the comparisons, every train dataset in `datasets/train/45x15randomE/` sets
`[source].exclude_indices` to the last 15 matrix ids (85-99, by
`enumerate_by = "name"` order), and `datasets/test/45x15randomE/gaussian-eval.toml`
sets `[source].include_indices` to that same list — so comparisons only ever run
against matrices no neural preconditioner or POD basis has seen. `include_indices`/
`exclude_indices` are plain `[source]`-level id lists (see
`domain/generation/README.md`); they don't require computing anything — the two
lists are just the same 15 ids, used as an exclude on the train side and an
include on the eval side. The eval dataset's `[[generation.strategy]]` sample
count is set to exactly 15 (one per included matrix) so `matrix_index` in
`[[comparisons]]` addresses genuinely distinct matrices — `matrix_index` is a
*row position in that dataset*, not a raw family id, and only maps 1:1 to
distinct matrices when the strategy emits exactly one row per matrix (see
`_allocate_strategy_counts_across_bindings` in `domain/generation/orchestration.py`).
`gaussian-eval.toml` uses the cheap `gaussian_forward` strategy rather than
`gaussian_residuals`/CG trajectories because comparisons only read the *matrix*
from this dataset (RHS is synthesized fresh per `[[comparisons]]` entry) — unlike
the training datasets, nothing ever reads this dataset's own RHS/solution rows,
so there's no reason to pay for CG solves here. Its comparisons mirror the 45x15
RHS modes on 3 of the 15 eval matrix positions (0, 7, and 14 — first/middle/last):
random RHS, sparse RHS, and raw-LHS comparisons. The companion 45x15
`default-search.toml` case binds one Optuna search job per network variant. Those
search jobs tune learning rate, layer count, activation, bias, dropout, and
scale-equivariant initialization/gain parameters.

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

[[jobs]]
id = "scale-equivariant-embedded-factorized"
path = "../../jobs/ffnn/scale-equivariant-embedded-factorized.toml"

[[comparisons]]
id = "scaled"
matrix_dataset = "train-dataset"
rhs_source = { kind = "raw_lhs", path = "${NEURALLS_RAW_DIR}/SpectralData/45x15-displacements/UaVectorsFromSpectral/ua_vector from_spectral_no_realization_0.txt", row_kind = "standard", scale = 5.0 }

[[assignments]]
dataset = "train-dataset"
job = "scale-equivariant-embedded-factorized"
```

`[names].training` controls the MLflow experiment bucket for training runs.
`[names].comparison` controls the comparison bucket.

## Thin Job Example

```toml
[run]
type = "train"
seed = 42
precision = "64"
model = "../../profiles/model/ffnn/scale-eq-full/identity-4L.toml"
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
objective = "val/loss"
space."training.optimizer.default_optimizer.lr" = { type = "log_float", low = 1e-5, high = 1e-2 }
```

Search stays coupled to jobs, not training profiles. The job is the runnable
entrypoint, so the inline `[search]` block belongs there.

## Model Profile Example

```toml
[model]
name = "ScaleEquivariantEmbeddedFactorizedFFNN"
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

## Eval-Only Runs

Use `neuralls eval <case.toml>` after training to score completed assignment
checkpoints on their original logged test splits:

```bash
uv run neuralls eval configs/cases/93x31/evaluate-all.toml --metric mae
```

The eval workflow reads existing `[[assignments]]`; it does not define a
separate registry. For each assignment it finds the latest FINISHED training
run, downloads its checkpoint and `splits/*.json` artifact from MLflow, and
passes that split file to DLKit via `data.splits.filepath`. Runs without split
artifacts are not evaluable until the training run is rerun or repaired.

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
