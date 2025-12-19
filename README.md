# Graph-CG: Neural Preconditioners for Conjugate Gradient Solvers

Graph-CG explores neural networks as preconditioners and warm-starts for Conjugate Gradient (CG) on graph-structured systems. The repository is organized around a unified experiment configuration system and a modular data-generation pipeline.

## Project Layout

- `configs/` – Configuration directory containing:
  - `experiments.toml` – Master experiment matrix linking model, data, and solver configs
  - `experiments/` – Experiment-specific config bundles (each with `model.toml`, `data.toml`, optional `solver.toml`)
  - `datasets/` – Dataset specifications (collection/generation parameters, test sets)
- `src/neuralls/` – Library code organized by functionality:
  - `cli/` – Command-line entry points for data processing, training, comparison, and orchestration
  - `configuration/` – Config loading, validation, and Pydantic models
  - `solver/` – CG algorithms and preconditioner implementations
  - `workflows/` – Prefect-based orchestration workflows
  - `generation/` – Data synthesis and collection utilities
  - `diagnostics/` – Logging, metrics, and analysis tools
- `tests/` – End-to-end and unit coverage across CLI, configuration, generation, solver, and workflows.

## Experiment Configuration System

Experiments are organized as config bundles that combine model, data, and solver settings. The system uses **typed Pydantic models** for validation at load time:

```python
from neuralls.configuration.loader import load_experiment
from neuralls.configuration.solver import get_solver_params

# Load experiment with model, data, and solver configs
experiment = load_experiment(
    model_config_path="configs/experiments/default/model.toml",
    data_config_path="configs/datasets/collect-504-solutions.toml",
    solver_config_path="configs/experiments/default/solver.toml",  # optional
)

# Access validated settings and paths
params = get_solver_params(experiment.settings)
workspace = experiment.workspace
checkpoint_dir = workspace.checkpoint_dir
```

**Configuration structure:**
- **Model configs:** Neural architecture and training hyperparameters (validated by `ModelConfigFile`)
- **Data configs:** Dataset sources and generation strategy (validated by `DataConfigFile`)
- **Solver configs:** CG tolerances, iteration limits, and preconditioner specifications (validated by `SolverConfigFile`)

All configs are validated using **Pydantic** at load time, catching configuration errors early with clear, actionable error messages.

The master `configs/experiments.toml` file orchestrates multiple experiment bundles for batch workflows.

## Workflows and CLI

All CLI scripts are located in `src/neuralls/cli/` and use the `neuralls` package.

- **Process data** (collection or generation):
  ```bash
  uv run python src/neuralls/cli/process_data.py configs/datasets/collect-504-solutions.toml --solve
  ```

- **Train a model**:
  ```bash
  # Using model config and data config
  uv run python src/neuralls/cli/train_model.py \
    configs/experiments/default/linear.toml \
    --data-config configs/datasets/collect-504-solutions.toml
  ```

- **Make predictions**:
  ```bash
  uv run python src/neuralls/cli/predict.py \
    configs/experiments/default/linear.toml \
    --data-config configs/datasets/collect-504-solutions.toml
  ```

- **Compare preconditioners** across experiments:
  ```bash
uv run python src/neuralls/cli/compare_preconditioners.py --experiments configs/experiments.toml
  ```

- **Run full experiment matrix** (data + train + compare):
  ```bash
  uv run python src/neuralls/cli/run_experiments.py --config configs/experiments.toml
  ```

- **Optional MLflow logging** (adds remote tracking; local files remain):
  ```bash
  # Add --enable-mlflow to any CLI command
  uv run python src/neuralls/cli/train_model.py configs/experiments/default/linear.toml \
    --data-config configs/datasets/collect-504-solutions.toml --enable-mlflow
  ```

Prefect orchestration is implemented in `src/neuralls/workflows/` and exposed via the CLI scripts above.

## Tests and Tooling

**Run tests:**
```bash
# All tests
uv run pytest tests -v

# Targeted suites
uv run pytest tests/generation -v        # Data generation tests
uv run pytest tests/solver -v            # CG solver tests
uv run pytest tests/cli tests/workflows -v  # CLI and workflow tests
uv run pytest tests/configuration -v     # Config loading tests
```

**Type checking:**
```bash
uv run pyright src/neuralls
```

**Output paths:**
- Default output root: `/data/projects/graph-cg/data/output`
- Override via environment: `export GRAPH_CG_OUTPUT_DIR=/custom/path`
- Or configure in `configs/experiments.toml` under the `output_dir` key

All experiment outputs are organized by experiment name and include checkpoints, predictions, comparisons, and visualizations.
