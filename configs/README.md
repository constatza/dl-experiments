# Configuration Guide

The repo uses one registry to move users from a single local run to a full
experiment batch.

## Start With The Right File

Choose the config type that matches the task:

- `datasets/*.toml`: build one processed dataset
- `models/*.toml`: train one model
- `comparison/*.toml`: define one solver benchmark profile
- `experiments-*.toml`: tie datasets, models, comparisons, MLflow, and
  experiment ids together

## Recommended Progression

### 1. Validate one dataset

```bash
uv run process-data configs/datasets/residuals-100.toml
```

### 2. Validate one model

```bash
uv run train-model configs/models/ffnn-l2.toml \
  --data-config configs/datasets/residuals-100.toml
```

### 3. Move to a registry

```bash
uv run run-experiments --config configs/experiments-ffnn.toml
```

### 4. Benchmark with the same registry

```bash
uv run compare-all configs/experiments-ffnn.toml
```

## Registry Anatomy

Each `experiments-*.toml` file is the discoverability layer for one experiment
family.

```toml
project_root = ".."
output_dir = "${GRAPH_CG_OUTPUT_DIR}"

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
settings come from the selected registry or the execution environment.

Model configs must use DLKit's canonical workflow syntax directly. `neuralls`
does not translate top-level `[OPTIMIZATION]` or infer optimization mode from
`OPTUNA.enabled`.

The two important roots are:

- `processed_root` for datasets
- `output_root` for MLflow, checkpoints, and reports

Checked-in configs are now portable by convention:

- `GRAPH_CG_RAW_DIR`: root for raw matrix/archive inputs
- `GRAPH_CG_PROCESSED_DIR`: root for processed datasets used by training and comparison
- `GRAPH_CG_OUTPUT_DIR`: root for MLflow, checkpoints, and reports

Local path normalization and sqlite URI handling delegate to DLKit's
`PathResolver` and local URI helpers. Relative paths are resolved against the
config file directory instead of assuming a Unix-only working directory layout.

## What To Read Next

- [`configs/datasets/README.md`](datasets/README.md) for strategy details
- [`../README.md`](../README.md) for the end-to-end workflow
