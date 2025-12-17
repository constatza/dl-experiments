# Graph-CG: Neural Preconditioners for Conjugate Gradient Solvers

Graph-CG explores neural networks as preconditioners and warm-starts for Conjugate Gradient (CG) on graph-structured systems. The repository is organized around a three-config system and a unified data-generation pipeline.

## Project Layout

- `configs/` – Model templates (FFNN, GNN, linear) and the experiment matrix `experiments.toml`.
- `data-configs/` – Data templates (e.g., `collect-504-solutions.toml`, `collect-2040-solutions.toml`, `test-solutions.toml`, eigenvector-based tests).
- `solver-configs/` – CG settings decoupled from models (`default.toml`, `cg.toml`, `pcg.toml`).
- `scripts/` – Automation entry points for data processing, training, comparison, and experiment orchestration. (Note: `test_prediction_quality.py` was removed).
- `src/` – Library code (configuration, generation, solver, diagnostics, modularized workflows package).
- `tests/` – End-to-end and unit coverage across CLI, configuration, generation, solver, and workflows.

## Three-Config System

Configurations are loaded together to keep models, data, and solver settings independent. The system now uses **typed Pydantic models** for validation at load time:

```python
from src.configuration.loader import load_experiment
from src.configuration.solver import get_solver_params

# Load experiment with validated, typed configs
experiment = load_experiment(
    "configs/ffnn.toml",
    data_config_path="data-configs/collect-504-solutions.toml",
    solver_config_path="solver-configs/default.toml",
)

# Access validated settings and paths
params = get_solver_params(experiment.settings)
workspace = experiment.workspace
checkpoint_dir = workspace.checkpoint_dir
```

- **Model configs (`configs/`):** architecture and training hyperparameters (validated by `ModelConfigFile`)
- **Data configs (`data-configs/`):** data sources and generation strategy (validated by `DataConfigFile`)
- **Solver configs (`solver-configs/`):** CG tolerances, iteration limits, and preconditioner set (validated by `SolverConfigFile`)

All configs are validated using **Pydantic** at load time, catching configuration errors early with clear, actionable error messages.

`configs/experiments.toml` pairs these configs for batch workflows.

## Workflows and CLI

- Process data (collection or generation):
  ```bash
  uv run python scripts/process_data.py data-configs/collect-504-solutions.toml --solve
  ```
- Train a model:
  ```bash
  uv run python scripts/train_model.py \
    --config configs/ffnn.toml \
    --data-config data-configs/collect-504-solutions.toml \
    --solver-config solver-configs/default.toml
  ```
- Compare preconditioners:
  ```bash
  uv run python scripts/compare_methods.py --experiments configs/experiments.toml
  ```
- Run the full experiment matrix (data + train + compare):
  ```bash
  uv run python scripts/run_experiments.py --config configs/experiments.toml
  ```
- Optional MLflow logging (adds uploads; local files remain):
  ```bash
  uv run python scripts/train_model.py --config configs/ffnn.toml --data-config data-configs/collect-504-solutions.toml --enable-mlflow
  uv run python scripts/predict.py --config configs/ffnn.toml --data-config data-configs/collect-504-solutions.toml --enable-mlflow
  uv run python scripts/compare_methods.py --experiments configs/experiments.toml --enable-mlflow
  ```

Prefect orchestration lives in `src/workflows/flow.py` (also exposed via the scripts above).

## Tests and Tooling

- Run all tests: `uv run pytest tests -v`
- Targeted suites:
  - Generation: `uv run pytest tests/generation -v`
  - Solver: `uv run pytest tests/solver -v`
  - CLI/workflows: `uv run pytest tests/cli tests/workflows -v`
- Type checking: `uv run pyright src`

Outputs and large artifacts default to `/data/projects/graph-cg/data/output`. For more control over the output root, set the `GRAPH_CG_OUTPUT_DIR` environment variable, or specify `output_root` in your `experiments.toml` paths section.
