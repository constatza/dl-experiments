# Configuration Guide

The repo uses one case config to move users from a single local run to a full
experiment batch.

## Start With The Right File

Choose the config type that matches the task:

- `datasets/*.toml`: build one processed dataset
- `models/*.toml`: train one model
- `comparison/*.toml`: define one solver benchmark profile
- `experiments-*.toml`: case configs that tie datasets, models, comparisons,
  MLflow, and experiment ids together

## Recommended Progression

### 1. Validate one dataset

```bash
uv run process-data configs/datasets/residuals-100.toml \
  --case-config configs/experiments-ffnn.toml
```

### 2. Validate one model

```bash
uv run train-model configs/models/ffnn-l2.toml \
  --data-config configs/datasets/residuals-100.toml \
  --case-config configs/experiments-ffnn.toml
```

### 3. Move to a case config

```bash
uv run run-experiments --config configs/experiments-ffnn.toml
```

### 4. Benchmark with the same case config

```bash
uv run compare-all configs/experiments-ffnn.toml
```

## Case Anatomy

Each `experiments-*.toml` file is a case config for one experiment family.

```toml
raw_dir = "${NEURALLS_RAW_DIR}"
processed_dir = "${NEURALLS_PROCESSED_DIR}"
project_root = ".."
output_dir = "${NEURALLS_OUTPUT_DIR}"

[names]
training = "Train"
comparison = "Comparisons"

[[datasets]]
id = "residuals-100"
path = "datasets/residuals-100.toml"

[[models]]
id = "scaleequivariant-residual-ffnn-l2"
path = "models/ffnn-l2.toml"

[[comparisons]]
id = "gaussian"
path = "comparison/gaussian.toml"

[[experiments]]
id = "residuals-100-scaleequivariant-residual-ffnn-l2"
dataset = "residuals-100"
model = "scaleequivariant-residual-ffnn-l2"
```

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
- staged optimization under `TRAINING.optimizer.stages`
- checkpoint callback naming

Checked-in model configs keep `module_path = "dlkit.nn"` as the user-facing
entrypoint and now define a repo-wide two-stage optimizer program:

- stage 1: AdamW
- switch: epoch `200`
- stage 2: LBFGS

Checked-in configs use canonical DLKit syntax: the default scheduler lives
under `TRAINING.optimizer.default_scheduler`, and any staged program lives
under `TRAINING.optimizer.stages`.

### Comparison configs

Comparison configs define:

- solver tolerances and iteration limits
- matrix and RHS inputs for benchmarking
- explicit `[[preconditioners]]` blocks

## MLflow And Paths

Model configs do not define their own `[MLFLOW]` block. Runtime tracking
settings come from the selected case config or the execution environment.

Model configs must use DLKit's canonical workflow syntax directly. `neuralls`
does not translate top-level `[OPTIMIZATION]` or infer optimization mode from
`OPTUNA.enabled`.

The two important roots are:

- `processed_root` for datasets
- `output_root` for MLflow, checkpoints, and reports

Checked-in configs are portable by convention and use `NEURALLS_*` environment
variables. Set these in one explicit env file passed with `--env-file`, expose
them through `NEURALLS_ENV_FILE`, or export them in the shell. `.env.example`
documents the supported keys and the explicit-loading model.

- `NEURALLS_RAW_DIR`: root for raw matrix/archive inputs
- `NEURALLS_PROCESSED_DIR`: root for processed datasets used by training and comparison
- `NEURALLS_OUTPUT_DIR`: root for MLflow, checkpoints, and reports

Case configs are the authoritative persisted source. Env files are optional
override layers for machine-specific roots rather than ambient cwd state.
`neuralls` does not auto-discover `.env` or `.env.local`.

Local path normalization and sqlite URI handling delegate to DLKit's
`PathResolver` and local URI helpers. Relative paths are resolved against the
config file directory instead of assuming a Unix-only working directory layout.

## What To Read Next

- [`configs/datasets/README.md`](datasets/README.md) for strategy details
- [`../README.md`](../README.md) for the end-to-end workflow
