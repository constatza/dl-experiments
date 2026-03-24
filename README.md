# neuralls

neuralls trains neural models that help Conjugate Gradient solve graph-structured
linear systems faster. The repo is organized around one practical workflow:

1. build or collect a dataset
2. train one model or a registry of models
3. compare neural and classical preconditioners on benchmark systems

## Start Here

Install dependencies with `uv`, then use the commands below in order.

### 1. Build one dataset

```bash
uv run process-data configs/datasets/residuals-100.toml
```

Use this first when you want to validate one dataset config and inspect the
processed output before training.

### 2. Train one model on that dataset

```bash
uv run train-model configs/models/ffnn-residual-l2.toml \
  --data-config configs/datasets/residuals-100.toml
```

This is the fastest path for checking whether one model and one dataset fit
together cleanly.

### 3. Scale up to a full registry run

```bash
uv run run-experiments --config configs/experiments-ffnn.toml
```

This command builds the datasets referenced by the registry, reuses cached
outputs where possible, and trains every declared experiment.

### 4. Compare preconditioners

```bash
uv run compare-all configs/experiments-ffnn.toml
```

This runs the comparison profiles declared in the selected registry and reports
how neural and classical preconditioners behave on the configured systems.

## Command Ladder

Use the commands in this order as you move from basic to advanced work:

| Level | Goal | Command |
| --- | --- | --- |
| Basic | Build one dataset | `uv run process-data <dataset.toml>` |
| Basic | Train one model | `uv run train-model <model.toml> --data-config <dataset.toml>` |
| Intermediate | Build every dataset in a registry | `uv run generate-all <registry.toml>` |
| Intermediate | Train every experiment in a registry | `uv run run-experiments --config <registry.toml>` |
| Intermediate | Train a batch and compare one metric | `uv run train-all <registry.toml>` |
| Advanced | Run solver comparisons | `uv run compare-all <registry.toml>` |
| Advanced | Run inference or synthetic evaluation | `uv run predict --config <model.toml> --data-config <dataset.toml>` |

## Configuration Map

The repo uses three config layers plus one registry:

- `configs/datasets/*.toml`: how to build or collect processed datasets
- `configs/models/*.toml`: DLKit model and training settings
- `configs/comparison/*.toml`: solver-comparison inputs and preconditioners
- `configs/experiments-*.toml`: the registry that ties datasets, models,
  comparisons, MLflow settings, and experiment ids together

Example registry shape:

```toml
[[datasets]]
id = "residuals-100"
path = "datasets/residuals-100.toml"

[[models]]
id = "normscaled-residual-ffnn-l2"
path = "models/ffnn-residual-l2.toml"

[[comparisons]]
id = "gaussian"
path = "comparison/gaussian.toml"

[[experiments]]
id = "residuals-100-normscaled-residual-ffnn-l2"
dataset = "residuals-100"
model = "normscaled-residual-ffnn-l2"
```

Read more in:

- [`configs/README.md`](configs/README.md)
- [`configs/datasets/README.md`](configs/datasets/README.md)

## Basic To Advanced Concepts

### Basic: single dataset, single model

Work with one dataset config and one model config until:

- the dataset builds successfully
- the model trains successfully
- the checkpoint lands in the expected output root

### Intermediate: registry-driven experimentation

Move to a registry when you want:

- repeatable experiment ids
- one place for MLflow topology
- batch dataset generation
- batch training across multiple model and dataset combinations

### Advanced: solver and workflow internals

Once the outer workflow is stable, the repo opens into three deeper packages:

- [`src/neuralls/generation/README.md`](src/neuralls/generation/README.md):
  dataset strategy internals
- [`src/neuralls/solver/README.md`](src/neuralls/solver/README.md):
  PCG, FCG, monitoring, and preconditioners
- [`src/neuralls/workflows/README.md`](src/neuralls/workflows/README.md):
  batch orchestration and comparison pipelines

## Output Layout

Two roots matter:

- `processed_root`: generated datasets
- `output_root`: MLflow tracking, checkpoints, reports, and artifacts

The selected registry and path helpers in `src/neuralls/configuration/paths.py`
determine those locations. `output_root` is the source of truth for training and
comparison artifacts.

## Testing

```bash
uv run pytest tests/generation -v
uv run pytest tests/configuration -v
uv run pytest tests/solver -v
uv run pytest tests/workflows tests/cli -v
uv run pyright src/neuralls
```

## Repo Guide

- `src/neuralls/cli/`: user-facing commands
- `src/neuralls/configuration/`: typed config loading and workspace resolution
- `src/neuralls/generation/`: dataset creation strategies and processing
- `src/neuralls/solver/`: CG solvers, monitoring, and preconditioners
- `src/neuralls/workflows/`: orchestration for training, prediction, and comparison

If you are new to the repo, stay on the command ladder above. If you are
changing internals, start with the package README that matches the layer you are
touching.
