# Configuration Guide

The public CLI is case-oriented. One case config drives the batch workflows for
dataset generation, training, full execution, and comparison.

This checkout is pinned to CUDA 13.0. Run `uv sync` once for the project
environment, then use plain `uv run neuralls ...` commands.

## Start With The Right File

Choose the config type that matches the case you want to run:

- `datasets/*.toml`: define one dataset source or generation strategy
- `models/*.toml`: define one model architecture and trainer setup
- `comparison/*.toml`: define one solver benchmark profile
- `case-*.toml`: case configs that tie datasets, models, comparisons,
  MLflow, and experiment ids together

The repository also includes focused sweep cases such as
`case-constant-width.toml`, which binds one model to a family of related
datasets.

Current model families include `linear/`, `skip-ffnn/`, `gnn/`, and
`advanced/`. The `advanced/` family now targets the scale-equivariant DLKit
models `ScaleEquivariantFourierFeatureNetwork`,
`ScaleEquivariantSiren`, and `ScaleEquivariantModifiedMLP`.

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
It does not infer or materialize standalone benchmark datasets referenced by
`comparison/*.toml`.

### 3. Train the case batch

```bash
uv run neuralls train /path/to/case.toml
```

### 4. Run or compare the same case

```bash
uv run neuralls run /path/to/case.toml
uv run neuralls compare /path/to/case.toml
```

If one comparison profile depends on an extra benchmark dataset such as
`configs/datasets/solutions.toml`, build it explicitly first:

```bash
uv run neuralls generate-single configs/datasets/solutions.toml \
  --case-config configs/case-ffnn.toml
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

[[models]]
id = "my-model"
path = "models/<family>/my-model.toml"

[[comparisons]]
id = "my-solver"
path = "comparison/my-solver.toml"

[[experiments]]
id = "my-dataset-my-model"
dataset = "my-dataset"
model = "my-model"
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

### Model configs

Model configs define:

- DLKit model module and hyperparameters
- trainer, loss, and optimizer-policy settings
- either `TRAINING.optimizer.default_optimizer` / `default_scheduler` or
  staged optimization under `TRAINING.optimizer.stages`
- checkpoint callback naming

Model configs keep `module_path = "dlkit.nn"` as the user-facing entrypoint.
`MODEL.name` must match the target class name exactly; kwargs stay flat under
`[MODEL]` and are forwarded through DLKit's `ModelComponentSettings`
filtering, so they must match the target constructor signature exactly.
For constant-width FFNN variants, "constant width" means the hidden body uses
the same width across layers; the width itself is still configured explicitly
with the constructor's `size` or `hidden_size` kwarg.

Optimizer sections may omit per-optimizer kwargs when DLKit defaults are
acceptable. Scheduler settings should still be written explicitly unless the
case intentionally wants DLKit's scheduler defaults.

Model configs use canonical DLKit syntax: the default scheduler lives
under `TRAINING.optimizer.default_scheduler`, and any staged program lives
under `TRAINING.optimizer.stages`.

### Comparison configs

Comparison configs define:

- solver tolerances and iteration limits
- matrix and RHS inputs for benchmarking
- explicit `[[preconditioners]]` blocks

Comparison configs may point at processed benchmark datasets that are separate
from the case `[[datasets]]` registry. Those benchmark datasets must be
materialized with `neuralls generate-single` before comparison runs.

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

- [`datasets/README.md`](datasets/README.md) for strategy details and schema examples
- [`../README.md`](../README.md) for the end-to-end workflow
