# Configuration System

This project now uses a strict split between:
- training configs (`experiments.toml` + `models/*` + `datasets/*`)
- comparison configs (`comparison/*`, `schema_version = 3`)

## Directory Layout

```text
configs/
  experiments.toml          # Training experiment matrix
  experiments-ffnn.toml     # Alternate training matrix
  models/                   # Model/training settings
  datasets/                 # Dataset generation settings
  comparison/
    linear.toml             # Comparison settings for NormScaledLinearFFNN
    ffnn.toml               # Comparison settings for NormScaledConstantWidthFFNN
```

## Training Protocol

`experiments.toml` defines training runs via `[[experiment]]` (or `[[run]]` in direct-path mode).

Example (`[[experiment]]` form):

```toml
project_root = ".."
output_dir = "/data/projects/graph-cg/data/output"

[mlflow]
tracking_uri = "sqlite:////data/projects/graph-cg/data/output/mlruns/mlflow.db"
# optional; if omitted for sqlite URIs, defaults to sibling `mlartifacts/`
# artifacts_destination = "/data/projects/graph-cg/data/output/mlartifacts"

[comparisons]
tracking_uri = "sqlite:////data/projects/graph-cg/data/comparisons/mlflow.db"
artifact_location = "/data/projects/graph-cg/data/comparisons/mlartifacts"

[[experiment]]
id = "linear_test_solutions"
dataset = "solutions"
model = "linear"
```

Notes:
- `dataset` and `model` are IDs, not file paths.
- The loader resolves actual config files from the active config root.
- Checkpoints are training outputs; they are not declared in `experiments.toml`.

### MLflow Configuration (Flat Only)

Model configs now use a flat MLflow section:

```toml
[MLFLOW]
enabled = true
experiment_name = "optional-user-override"
run_name = "optional-user-override"
```

Nested `[MLFLOW.client]` / `[MLFLOW.server]` keys are no longer supported.
Model configs must not define `tracking_uri` or `artifacts_destination`.
Those infrastructure values come from `experiments.toml` or runtime env.

## Comparison Protocol (Strict)

Comparison configs must use `schema_version = 3` and `[[preconditioners]]`.

```toml
schema_version = 3
run_name = "comparison-default"

[general]

[general.params]
rtol = 1e-6
atol = 1e-14
max_iterations = 200
stopping_criterion = "residual_norm"
m_max = 10
breakdown_tol = 1e-14

[general.data]
matrix_path = "/data/processed/solutions"
rhs_path = "/data/processed/gaussian-rhs"
rhs_index = 0
dataset_alias = "solutions"
normalize_system = "matrix"

[general.tracking]
tracking_uri = "sqlite:////data/projects/graph-cg/data/comparisons/mlflow.db"
artifact_location = "/data/projects/graph-cg/data/comparisons/mlartifacts"
experiment_name = "neuralls-comparisons"

[general.model_store]
tracking_uri = "sqlite:////data/projects/graph-cg/data/output/mlruns/mlflow.db"

[[preconditioners]]
name = "jacobi"
type = "jacobi"

[[preconditioners]]
name = "neural_linear"
type = "neural"
fallback = "identity"
model_ref = { source = "registered", name = "NormScaledLinearFFNN", alias = "@dataset" }
```

Validation rules:
- `schema_version` must be `3`.
- `preconditioners` must be non-empty.
- Neural preconditioners must use `model_ref`.
- `general.model_store` is required if any neural preconditioner uses `model_ref`.
- If any alias is `"@dataset"`, `general.data.dataset_alias` is required.
- Unknown keys are rejected.

## Model Resolution

Neural preconditioners resolve from MLflow via `model_ref`:
- `source = "registered"`: resolve by model name + alias/version.
- `source = "logged"`: resolve by run/model filters (or explicit `run_id`).

Checkpoint selection for comparison is deterministic:
- resolve MLflow run from `model_ref`
- download checkpoint artifact for that run
- use downloaded checkpoint path in the preconditioner

## CLI Usage

Training:

```bash
uv run train-multiple configs/experiments.toml
```

Comparison (standalone):

```bash
uv run compare-preconditioners --comparison-config configs/comparison/linear.toml
```

Comparison (pipeline tagging/grouping):

```bash
uv run compare-preconditioners \
  --comparison-config configs/comparison/ffnn.toml \
  --comparison-run /path/to/comparison_run.json
```
