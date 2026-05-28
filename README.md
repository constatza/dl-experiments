# neuralls

`neuralls` is a research and experimentation toolkit for learning neural
preconditioners for Conjugate Gradient on graph-structured linear systems.
It supports three common workflows:

- build processed datasets from raw matrices and archives
- train one model or a full case-defined batch of experiments
- compare neural and classical preconditioners under a shared benchmark setup

The project is organized around a **case config**: one top-level TOML file that
declares the datasets, models, comparisons, and experiment registrations for a
single experiment family. Machine-specific roots live outside the repo in the
user config directory.

This checkout is pinned to CUDA 13.0. To change backends later, edit
`pyproject.toml`, then re-run `uv lock` and `uv sync`.

## What You Need

- Python managed with [`uv`](https://docs.astral.sh/uv/)
- CUDA 13.0 available on the machine for the pinned PyTorch build
- access to your own raw matrix data, processed dataset root, and output root

Install the project environment:

```bash
uv sync
```

The project is packaged with `uv_build`, so editable installs and local builds
use the native uv backend instead of the legacy setuptools fallback.

## Quickstart

The shortest useful path for a new user is:

1. select or create one case config
2. configure machine-specific data roots once
3. generate one case's datasets
4. train one case batch
5. run or compare a full case

### 1. Choose a case config

Checked-in examples currently live under names such as:

- `configs/case-ffnn.toml`
- `configs/case-linear.toml`
- `configs/case-parametrized.toml`

### 2. Set machine-specific roots

Profiles are defined in the user config file:

```text
~/.config/neuralls/config.toml
```

That file is the persistent machine-specific config store. `neuralls config`
is the CLI for creating, updating, listing, and selecting profiles inside it.
You can also edit `~/.config/neuralls/config.toml` manually as long as it
follows the expected TOML structure.

Set up the active machine profile once:

```bash
uv run neuralls config init
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
```

Example profile file:

```toml
[default]
raw_dir = "/data/raw"
processed_dir = "/data/processed"
output_dir = "/data/output"

[profiles.laptop]
raw_dir = "/mnt/external/raw"
processed_dir = "/mnt/external/processed"
output_dir = "/home/archer/laptop-output"
```

Profile format rules:

- `[default]` defines the fallback profile used when no named profile is selected
- `[profiles.<name>]` defines a named profile such as `laptop` or `windows`
- every profile must define `raw_dir`, `processed_dir`, and `output_dir`
- paths are expanded with `~` and normalized to absolute paths at load time

You can manage the same file either way:

```bash
uv run neuralls config path
uv run neuralls config list
uv run neuralls config show
uv run neuralls config create laptop --raw-dir /mnt/external/raw --processed-dir /mnt/external/processed --output-dir /home/archer/laptop-output
uv run neuralls config set output-dir /new/output laptop
uv run neuralls config delete laptop
```

or by editing `~/.config/neuralls/config.toml` directly.

If you want a starter file instead of answering prompts immediately, use:

```bash
uv run neuralls config init
```

That writes a commented template to `~/.config/neuralls/config.toml`. `config create`
is non-interactive and requires explicit `--raw-dir`, `--processed-dir`, and
`--output-dir` flags.

Profiles provide:

- `raw_dir`
- `processed_dir`
- `output_dir`

Profile selection works like this:

1. `--profile <name>` picks a named profile for one command
2. `NEURALLS_PROFILE=<name>` picks a named profile from the environment
3. otherwise `default` is used

Root overrides work after profile selection:

1. process env vars such as `NEURALLS_OUTPUT_DIR`
2. `--env-file <path>`
3. `NEURALLS_ENV_FILE=<path>`
4. the selected profile in `~/.config/neuralls/config.toml`

`config set` overwrites the existing field value in place for an existing
profile. `config delete NAME` removes a named profile; `default` cannot be
deleted.

`neuralls` does **not** auto-discover `.env`, `.env.local`, or a repo root.
There is no hidden cwd-based configuration search.

Example env file override:

```dotenv
NEURALLS_RAW_DIR=D:/neuralls/raw
NEURALLS_PROCESSED_DIR=D:/neuralls/processed
NEURALLS_OUTPUT_DIR=D:/neuralls/output
```

On Windows, prefer forward slashes in env-file paths.

### 3. Generate datasets

Use the batch form for a full case, or `generate-single` when you want to
inspect one dataset config directly.

```bash
uv run neuralls generate configs/case-<name>.toml --env-file .env.windows
uv run neuralls generate-single configs/datasets/<dataset>.toml \
  --case-config configs/case-<name>.toml \
  --env-file .env.windows
```

The batch form materializes every dataset referenced by the case config under
the resolved processed root. The `generate-single` form restores the one-dataset path
when you want to validate one dataset config in isolation.

### 4. Train one case batch

```bash
uv run neuralls train configs/case-<name>.toml --env-file .env.windows
```

This trains every experiment declared in the case config and writes aggregate
training outputs under the resolved output root.

### 5. Run or compare a full case

```bash
uv run neuralls run configs/case-<name>.toml --env-file .env.windows
uv run neuralls compare configs/case-<name>.toml --env-file .env.windows
```

`neuralls run` generates datasets as needed and trains the full experiment
matrix. `neuralls compare` benchmarks the configured solver setups for the same
case.

## Case Configs

A case config is the authoritative persisted config source for a run family.
It can contain:

- dataset registry entries under `[[datasets]]`
- model registry entries under `[[models]]`
- comparison registry entries under `[[comparisons]]`
- experiment registrations under `[[experiments]]`
- optional `[mlflow]` topology
- optional display names under `[names]`

Minimal example:

```toml
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

Important behavior:

- `[[experiments]]` remains the table name for per-run registrations
- relative paths inside the case config resolve against the case file location
- `${NEURALLS_*}` placeholders are expanded from resolved settings
- if `[mlflow]` is omitted, local SQLite tracking and local artifact paths are
  derived from the active settings `output_dir`

## Command Reference

| Goal | Command |
| --- | --- |
| Manage machine profiles | `uv run neuralls config ...` |
| Generate all datasets in one case | `uv run neuralls generate <case.toml>` |
| Generate one dataset config | `uv run neuralls generate-single <dataset.toml> --case-config <case.toml>` |
| Train all experiments in one case | `uv run neuralls train <case.toml>` |
| Generate datasets and train the full case | `uv run neuralls run <case.toml>` |
| Compare solver setups for one case | `uv run neuralls compare <case.toml>` |

Root-resolution rules are also explicit:

1. process env vars
2. `--env-file`
3. `NEURALLS_ENV_FILE`
4. the selected profile from `~/.config/neuralls/config.toml`
5. otherwise fail

There are no other fallbacks.

## Configuration Layout

The repo uses one case layer and three lower-level config families:

- `configs/datasets/*.toml`: dataset generation and input-source definitions
- `configs/models/*.toml`: DLKit model, trainer, loss, and optimizer settings
- `configs/comparison/*.toml`: solver comparison inputs and preconditioners
- `configs/case-*.toml`: case configs tying all of the above together

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

- `src/neuralls/cli/`: the `neuralls` root CLI plus config and case-batch commands
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
