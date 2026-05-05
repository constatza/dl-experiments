# neuralls

`neuralls` is a research and experimentation toolkit for learning neural
preconditioners for Conjugate Gradient on graph-structured linear systems.
It supports three common workflows:

- build processed datasets from raw matrices and archives
- train one model or a full case-defined batch of experiments
- compare neural and classical preconditioners under a shared benchmark setup

The project is organized around a **case config**: one top-level TOML file that
declares the datasets, models, comparisons, runtime roots, and experiment
registrations for a single experiment family.

## What You Need

- Python managed with [`uv`](https://docs.astral.sh/uv/)
- one PyTorch backend extra selected at install time
- access to your own raw matrix data, processed dataset root, and output root

Install the environment with exactly one backend:

```bash
uv sync --extra cpu
uv sync --extra cu128
uv sync --extra cu130
```

## Quickstart

The shortest useful path for a new user is:

1. select or create one case config
2. point the case at your machine-specific data roots
3. build one dataset
4. train one model
5. scale up to a full case run

### 1. Choose a case config

Checked-in examples currently live under names such as:

- `configs/experiments-ffnn.toml`
- `configs/experiments-linear.toml`
- `configs/experiments-parametrized.toml`

Those files are **case configs** even though the filenames still say
`experiments-*`. The generic conceptual name is `case.toml`.

### 2. Set machine-specific roots

Every case config must resolve three runtime roots:

- `raw_dir`
- `processed_dir`
- `output_dir`

Checked-in examples keep those fields portable by using placeholders:

```toml
raw_dir = "${NEURALLS_RAW_DIR}"
processed_dir = "${NEURALLS_PROCESSED_DIR}"
output_dir = "${NEURALLS_OUTPUT_DIR}"
```

You can provide those values in one of three explicit ways:

1. export `NEURALLS_RAW_DIR`, `NEURALLS_PROCESSED_DIR`, and `NEURALLS_OUTPUT_DIR`
2. pass `--env-file <path>`
3. set `NEURALLS_ENV_FILE` to an env file path

`neuralls` does **not** auto-discover `.env`, `.env.local`, or a repo root.
There is no hidden cwd-based configuration search.

Example env file:

```dotenv
NEURALLS_RAW_DIR=D:/neuralls/raw
NEURALLS_PROCESSED_DIR=D:/neuralls/processed
NEURALLS_OUTPUT_DIR=D:/neuralls/output
```

On Windows, prefer forward slashes in env-file paths.

### 3. Build one dataset

Commands that start from a dataset or model config must also be told which case
to use. Pass it explicitly with `--case-config`, or set
`NEURALLS_CASE_CONFIG`.

```bash
uv run process-data configs/datasets/residuals-100.toml \
  --case-config configs/experiments-ffnn.toml \
  --env-file .env.windows
```

This validates one dataset config and writes the processed dataset under the
resolved processed root.

### 4. Train one model

```bash
uv run train-model configs/models/ffnn-l2.toml \
  --data-config configs/datasets/residuals-100.toml \
  --case-config configs/experiments-ffnn.toml \
  --env-file .env.windows
```

Use this path first when validating a model, a dataset, and one training setup
before running a larger batch.

### 5. Run a full case

```bash
uv run run-experiments --config configs/experiments-ffnn.toml \
  --env-file .env.windows
```

This executes the experiments declared in the selected case config, reusing
processed datasets and writing outputs under the resolved output root.

## Case Configs

A case config is the authoritative persisted config source for a run family.
It can contain:

- runtime roots: `raw_dir`, `processed_dir`, `output_dir`
- dataset registry entries under `[[datasets]]`
- model registry entries under `[[models]]`
- comparison registry entries under `[[comparisons]]`
- experiment registrations under `[[experiments]]`
- optional `[mlflow]` topology
- optional display names under `[names]`

Minimal example:

```toml
raw_dir = "${NEURALLS_RAW_DIR}"
processed_dir = "${NEURALLS_PROCESSED_DIR}"
output_dir = "${NEURALLS_OUTPUT_DIR}"

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

Important behavior:

- `[[experiments]]` remains the table name for per-run registrations
- relative paths inside the case config resolve against the case file location
- `${NEURALLS_*}` placeholders are expanded from resolved settings
- if `[mlflow]` is omitted, local SQLite tracking and local artifact paths are
  derived from `output_dir`

## Command Reference

| Goal | Command |
| --- | --- |
| Build one dataset | `uv run process-data <dataset.toml> --case-config <case.toml>` |
| Build all datasets in one case | `uv run generate-all <case.toml>` |
| Train one model | `uv run train-model <model.toml> --data-config <dataset.toml> --case-config <case.toml>` |
| Train all experiments in one case | `uv run run-experiments --config <case.toml>` |
| Train a batch workflow | `uv run train-all <case.toml>` |
| Compare solver setups for one case | `uv run compare-all <case.toml>` |
| Run inference | `uv run predict --config <model.toml> --data-config <dataset.toml> --case-config <case.toml>` |

Case-selection rules are intentionally narrow:

1. explicit `--case-config`
2. `NEURALLS_CASE_CONFIG`
3. otherwise fail

Root-resolution rules are also explicit:

1. process env vars
2. `--env-file`
3. `NEURALLS_ENV_FILE`
4. the case config's `raw_dir`, `processed_dir`, and `output_dir`
5. otherwise fail

There are no other fallbacks.

## Configuration Layout

The repo uses one case layer and three lower-level config families:

- `configs/datasets/*.toml`: dataset generation and input-source definitions
- `configs/models/*.toml`: DLKit model, trainer, loss, and optimizer settings
- `configs/comparison/*.toml`: solver comparison inputs and preconditioners
- `configs/experiments-*.toml`: case configs tying all of the above together

Use the narrowest config that matches the task:

- debugging data generation: start with a dataset config
- validating one architecture: add one model config
- running repeatable experiment batches: move to a case config

Additional guidance:

- [Configuration Guide](configs/README.md)
- [Dataset Config Guide](configs/datasets/README.md)
- [Architecture Docs](docs/README.md)

## Outputs

Two roots matter operationally:

- `processed_dir`: generated datasets used by training and comparison
- `output_dir`: MLflow tracking, model checkpoints, figures, reports, and
  artifacts

`neuralls` does not migrate external data for you. Moving to a new machine
means you are responsible for copying any required raw datasets, processed
datasets, checkpoints, and MLflow state into the new roots.

## Repository Guide

If you need to work below the CLI layer:

- `src/neuralls/cli/`: Typer entry points
- `src/neuralls/composition/`: workflow assembly and orchestration
- `src/neuralls/application/`: use-case logic
- `src/neuralls/platform/`: config, storage, tracking, DLKit integration
- `src/neuralls/domain/`: generation, solver logic, analysis, normalization
- `src/neuralls/shared/`: constants, shared types, functional helpers

Module-level architecture notes live alongside the code under the corresponding
package directories.

## Development

Install the development toolchain:

```bash
uv tool install prek
uv sync --dev
prek install -t pre-commit -t pre-push
```

Useful verification commands:

```bash
uv run ruff check src tests
uv run ty check src/ tests/
uv run pytest
```

The repository uses local hooks plus CI. Ruff is the linting baseline, `ty` is
the type-checking baseline, and `uv run ...` is the expected entrypoint for
Python tooling.
