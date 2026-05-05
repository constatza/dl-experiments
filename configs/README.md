# Configuration Guide

The public CLI is case-oriented. One case config drives the batch workflows for
dataset generation, training, full execution, and comparison.

## Start With The Right File

Choose the config type that matches the case you want to run:

- `datasets/*.toml`: define one dataset source or generation strategy
- `models/*.toml`: define one model architecture and trainer setup
- `comparison/*.toml`: define one solver benchmark profile
- `experiments-*.toml`: case configs that tie datasets, models, comparisons,
  MLflow, and experiment ids together

## Recommended Progression

### 1. Configure one machine profile

```bash
uv run neuralls config init
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
```

### 2. Generate datasets

```bash
uv run neuralls generate configs/experiments-ffnn.toml
uv run neuralls generate-single configs/datasets/residuals-100.toml \
  --case-config configs/experiments-ffnn.toml
```

### 3. Train the case batch

```bash
uv run neuralls train configs/experiments-ffnn.toml
```

### 4. Run or compare the same case

```bash
uv run neuralls run configs/experiments-ffnn.toml
uv run neuralls compare configs/experiments-ffnn.toml
```

## Case Anatomy

Each `experiments-*.toml` file is a case config for one experiment family.

```toml
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

- [`configs/datasets/README.md`](datasets/README.md) for strategy details
- [`../README.md`](../README.md) for the end-to-end workflow
