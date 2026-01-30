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

# Load experiment with model and data configs
experiment = load_experiment(
    model_config_path="configs/models/ffnn.toml",
    data_config_path="configs/datasets/collect-504-solutions.toml",
)

# Access validated settings and workspace paths
params = get_solver_params(experiment.settings)
workspace = experiment.workspace
checkpoint_dir = workspace.checkpoint_dir  # Derived from MLflow artifact_uri

# Access paths from workspace
data_dir = workspace.data_dir         # Where datasets live
output_root = workspace.root_dir      # Master output directory (from MLflow)
```

**Configuration structure:**
- **Model configs:** Neural architecture and training hyperparameters (validated by `ModelConfigFile`)
- **Data configs:** Dataset sources and generation strategy (validated by `DataConfigFile`)
- **Solver configs:** CG tolerances, iteration limits, and preconditioner specifications (validated by `SolverConfigFile`)

All configs are validated using **Pydantic** at load time, catching configuration errors early with clear, actionable error messages.

The master `configs/experiments.toml` file orchestrates multiple experiment bundles for batch workflows.

## Recent API Changes

### Configuration Loader (Breaking Changes)

**Simplified `load_experiment()` signature:**
- **Removed:** `solver_config_path` parameter - solver configs are now only used at comparison time, not during experiment loading
- **Added:** `output_root` optional parameter for overriding the master output directory
- **Before:**
  ```python
  load_experiment(model_config, data_config, solver_config)  # solver config removed
  ```
- **After:**
  ```python
  load_experiment(model_config, data_config, output_root=None)  # clean separation
  ```

**Removed legacy functions:**
- `build_flow_context()` - Use `build_path_context()` and `WorkspaceFactory` instead
- `load_data_context()` - Use `load_experiment()` with dedicated data processing utilities
- Import paths: `from neuralls.configuration.loader import load_experiment, load_batch` (not from top-level `__init__.py`)

**New path resolution system:**
- Moved from `neuralls.paths.core` → `neuralls.configuration.paths`
- Simplified from `FlowContext`/`ProjectRoots` → `PathContext`/`build_path_context()`
- Three core functions: `resolve_project_root()`, `resolve_output_root()`, `resolve_processed_root()`

### Solver Module (Major Refactoring)

**Modular architecture replacing monolithic files:**
- **`core/`** - Abstract base classes (`ISolver`, `IterativeSolverBase`, `KrylovSolverBase`)
- **`solvers/`** - Concrete implementations (`FlexibleCGSolver`, `PreconditionedCGSolver`)
- **`models/`** - Immutable state hierarchy (`SolverState`, `KrylovState`, `CGState`, `SolverResult`)
- **`monitoring/`** - Diagnostics (`TraceRecorder`, `HistoryTracker`)
- **`strategies/`** - Strategy patterns (`OrthogonalizationStrategy`, `IConvergenceCriterion`)

**Removed monolithic files:**
- `fcg_solver.py`, `pcg_solver.py` → `solvers/` directory with focused implementations
- `state.py`, `info.py` → `models/` directory with type-safe state containers
- `trace_recorder.py`, `convergence.py` → `monitoring/` directory
- `helpers.py` → `utils/` and `strategies/orthogonalization.py`

**Factory function updates:**
- `flexible_cg()` - Now uses `FlexibleCGSolver` class internally
- `preconditioned_cg()` - Now uses `PreconditionedCGSolver` class internally
- Import from: `from neuralls.solver.factories import flexible_cg, preconditioned_cg`

### Preconditioner Module (New)

**New dedicated package: `neuralls.preconditioner/`**
- **`builders.py`** - Factory functions for preconditioner construction
- **`registry.py`** - Type-safe preconditioner registry and lookup
- **`predictor.py`** - Neural preconditioner prediction interface

**Removed from solver package:**
- `preconditioner_factory.py` → Consolidated into `preconditioner/builders.py`

### IO and Comparison

**Moved functions:**
- `load_solver_config()`: `io.comparison` → `io.toml_loader` (canonical location with other loaders)

**Import updates:**
```python
# Before
from neuralls.io.comparison import load_solver_config

# After
from neuralls.io.toml_loader import load_solver_config
```

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

The `output_root` is the single source of truth for all experiment artifacts. MLflow tracking database and artifact storage are automatically derived from it:
- MLflow tracking: `{output_root}/mlruns/mlflow.db`
- MLflow artifacts: `{output_root}/mlartifacts/{experiment_id}/{run_id}/`

Processed datasets live separately in the `processed_root` directory (configurable per dataset config).

All path resolution is handled by three simple functions in `src/neuralls/configuration/paths.py`:
- `resolve_project_root()` - Project base directory
- `resolve_output_root()` - Master output directory (drives MLflow paths)
- `resolve_processed_root()` - Processed data directory
