# Configuration System

This project now uses a strict split between:
- checked-in experiments registries (`experiments-*.toml`)
- a preserved legacy linear registry (`experiments-linear.toml`)
- a structured-linear registry (`experiments-parametrized.toml`)
- training configs (`models/*` + `datasets/*`)
- comparison configs (`comparison/*`)

## Directory Layout

```text
configs/
  experiments-parametrized.toml # Structured-linear registry
  experiments-linear.toml   # Preserved linear-focused registry
  experiments-ffnn.toml     # Constant-width FFNN registry
  models/                   # Model/training settings
  datasets/                 # Dataset generation settings
  comparison/
    linear.toml             # Comparison settings for NormScaledLinearFFNN
    ffnn.toml               # Comparison settings for NormScaledConstantWidthFFNN
```

## Training Protocol

Each selected `experiments-*.toml` file acts as the discoverability layer for its experiment family. The checked-in registries currently cover structured-linear, preserved linear, and constant-width FFNN setups. All use explicit registries for datasets, models, comparisons, and experiment bindings.

Example:

```toml
project_root = ".."
output_dir = "/data/projects/graph-cg/data/output"

[mlflow]
tracking_uri = "sqlite:////data/projects/graph-cg/data/output/mlruns/mlflow.db"
# optional; if omitted for sqlite URIs, defaults to sibling `mlartifacts/`
# artifacts_destination = "/data/projects/graph-cg/data/output/mlartifacts"

[[datasets]]
id = "solutions"
path = "datasets/solutions.toml"

[[models]]
id = "linear"
path = "models/linear.toml"

[[comparisons]]
id = "linear"
path = "comparison/linear.toml"

[[experiments]]
id = "linear_test_solutions"
dataset = "solutions"
model = "linear"
```

Notes:
- Registry `id` values are stable lookup handles, not runtime dataset identity.
- Dataset runtime identity comes from the dataset config top-level `id`.
- The loader resolves actual config files from the registry path entries.
- `[[run]]` direct-path mode is still supported for ad hoc execution.
- Checkpoints are training outputs; optional `checkpoint_path` overrides remain experiment-local metadata.

### MLflow Configuration

Model configs should not define an `[MLFLOW]` section.
MLflow settings come from defaults plus the master `experiments*.toml` file or runtime env.
Model configs must not define `tracking_uri` or `artifacts_destination`.

Training MLflow naming is controlled at runtime:
- `[names].training` in `experiments.toml` defaults to `"Training"`
- training run names are built as `{experiment_display_name}-{timestamp}`
- comparison runs use timestamped display names plus structured tags

## Comparison Protocol (Strict)

Comparison configs use `[[preconditioners]]`.

```toml
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
normalize_system = "matrix"

[[preconditioners]]
name = "jacobi"
type = "jacobi"

[[preconditioners]]
name = "neural_linear"
type = "neural"
experiment = "linear_test_solutions"
fallback = "identity"
model_ref = { source = "registered", alias = "@dataset" }
```

Validation rules:
- `preconditioners` must be non-empty.
- Neural preconditioners must use `model_ref`.
- Neural preconditioner names must be unique.
- If a registered `model_ref` omits `name`, the neural preconditioner must define `experiment`.
- If any alias is `"@dataset"`, the dataset alias must come from either:
  - `general.data.dataset_alias`, or
  - the neural preconditioner `experiment` binding resolved from `experiments.toml`
- Unknown keys are rejected.

Runtime behavior:
- If a neural preconditioner cannot be resolved from MLflow, comparison logs a warning and skips that preconditioner.
- If every configured preconditioner is skipped, the comparison run fails.

## Model Resolution

Neural preconditioners resolve from MLflow via `model_ref`:
- `source = "registered"`: resolve by model name + alias/version.
- `source = "logged"`: resolve by run/model filters (or explicit `run_id`).

Experiment-bound neural resolution is deterministic:
- resolve the explicit `experiment` id
- derive dataset identity from the referenced dataset config `id`
- derive registered model name from the referenced model config when `model_ref.name` is omitted
- resolve MLflow run from `model_ref`
- download checkpoint artifact for that run
- use downloaded checkpoint path in the preconditioner

## CLI Usage

Dataset generation:

```bash
uv run generate-all configs/experiments.toml
```

Training:

```bash
uv run train-all configs/experiments.toml
```

Comparison (standalone):

```bash
uv run compare-all configs/experiments.toml
```
