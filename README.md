# neuralls

neuralls trains neural models that help Conjugate Gradient solve graph-structured
linear systems faster. The repo is organized around one practical workflow:

1. build or collect a dataset
2. train one model or a registry of models
3. compare neural and classical preconditioners on benchmark systems

## Start Here

Install dependencies with `uv`, selecting exactly one PyTorch backend extra:

```bash
uv sync --extra cpu
uv sync --extra cu128
uv sync --extra cu130
```

The `cpu`, `cu128`, and `cu130` extras map to the matching PyTorch wheel
indexes and are mutually exclusive. After syncing one of them, use the commands
below in order.

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
- [`docs/README.md`](docs/README.md)

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
uv run ruff check --fix .
uv run ruff format .
uv run pytest tests/generation -v
uv run pytest tests/configuration -v
uv run pytest tests/solver -v
uv run pytest tests/workflows tests/cli -v
uv run ty check
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

## Contributor Setup

Contributor workflows expect local hooks, Tach, and the pinned Python dev
toolchain.

```bash
uv tool install prek
uv sync --dev
prek install -t pre-commit -t pre-push
```

You can also install `prek` with your OS package manager instead of `uv tool`.

Contributor setup is not complete until `prek` is installed and the git hooks
are registered. Developer workflows are expected to use local `pre-commit` and
`pre-push` hooks; CI remains an independent enforcement layer.

`prek` is only the hook runner here. Python tool versions are pinned in this
repo's `uv` dev dependencies, and Python hooks run through `uv run ...`.

Current hook contract:

- `pre-commit`: `uv-lock` on dependency metadata changes, then Ruff lint and format autofix across `src/`, `tests/`, and `scripts/`, plus ty and Tach checks
- `pre-push`: the fast pytest smoke suite

For editor diagnostics and language-server features, use ty's language server
instead of Pyright or Pylance-specific type checking.
