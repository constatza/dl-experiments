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
  - `solver/` – CG algorithms and preconditioner implementations ([README](src/neuralls/solver/README.md))
  - `workflows/` – High-level orchestration for experiments and comparisons ([README](src/neuralls/workflows/README.md))
  - `generation/` – Data synthesis and collection utilities ([README](src/neuralls/generation/README.md))
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
- **`monitoring/`** - Diagnostics (`IterationHistory`, `EventLog`, `ResidualHistoryTracker`)
- **`strategies/`** - Strategy patterns (`OrthogonalizationStrategy`, `IConvergenceCriterion`)

**Removed monolithic files:**
- `fcg_solver.py`, `pcg_solver.py` → `solvers/` directory with focused implementations
- `state.py`, `info.py` → `models/` directory with type-safe state containers
- Monitoring refactored to `IterationHistory` (continuous) + `EventLog` (discrete events)
- `helpers.py` → `utils/` and `strategies/orthogonalization.py`

**Factory function updates:**
- `flexible_cg()` - Now uses `FlexibleCGSolver` class internally
- `preconditioned_cg()` - Now uses `PreconditionedCGSolver` class internally
- Import from: `from neuralls.solver.factories import flexible_cg, preconditioned_cg`

### Preconditioner Module

**Location:** `neuralls.solver.preconditioners/`

**Rationale:** Preconditioners are tightly coupled to solvers and used exclusively by solver algorithms.

**Package structure:**
- **`base.py`** - Abstract base classes (Preconditioner, LinearPreconditioner, ContextualPreconditioner)
- **`implementations.py`** - Concrete preconditioners (Identity, Jacobi, ILU, ICholesky, Neural, etc.)
- **`builders.py`** - Factory function for TOML-based preconditioner creation
- **`adapters.py`** - Framework integration (DLKit/PyTorch adapter for neural preconditioners)
- **`ports.py`** - Framework-agnostic predictor interface
- **`tensor_utils.py`** - Pure tensor conversion utilities

**Usage:**
```python
from neuralls.solver.preconditioners import create_preconditioner
```

**Key features:**
- LSP-compliant: Optional context parameter enables substitutability
- Type-safe: Modern Python 3.12+ generics for LinearPreconditioner[T]
- Unified interface: All preconditioners callable via `__call__` or `.apply()`
- Clean separation: Configuration models in `configuration/preconditioner.py`

### IO and Comparison

**Moved functions:**
- `load_solver_config()`: `io.comparison` → `io.toml_loader` (canonical location with other loaders)
- `run_cg_comparison()`, `format_results_summary()`, `summarize_best_combinations()`: `solver/` → `workflows/` (architectural cleanup - comparison orchestration is a workflow concern, not core solver logic)

**Import updates:**
```python
# Config loading
from neuralls.io.toml_loader import load_solver_config  # ✓ Canonical location

# Comparison orchestration
from neuralls.workflows import run_cg_comparison  # ✓ Moved from solver module

# Old import (deprecated)
from neuralls.solver import run_cg_comparison  # ✗ No longer available
```

**API simplification:**
- Removed parameters: `apply_every` (preconditioners applied continuously), `first_n` (redundant with `limit_iters`)
- Scheduling now handled by `ScheduledPreconditioner` wrapper (see `workflows/README.md`)

## Workflows and CLI

All CLI scripts are registered as commands and can be run via `uv run <command>`.

- **Process data** (collection or generation):
  ```bash
  uv run process-data configs/datasets/collect-504-solutions.toml
  ```
  - `config`: Path to data configuration TOML.

- **Train a model**:
  ```bash
  uv run train-model configs/models/linear.toml --data-config configs/datasets/collect-504-solutions.toml
  ```
  - `config`: Path to model architecture configuration.
  - `--data-config`: Path to the dataset metadata config used for training.
  - `--max-epochs`: (Optional) Override the maximum number of training epochs.

- **Make predictions**:
  ```bash
  uv run predict --config configs/models/linear.toml --data-config configs/datasets/collect-504-solutions.toml
  ```
  - `--config`: Path to the model configuration.
  - `--data-config`: Path to the data configuration.
  - `--synthetic`: Run a synthetic benchmark (using $x=ones$, $b=Ax$) instead of loading data.

- **Compare preconditioners** (standalone mode):
  ```bash
  uv run compare-preconditioners --solver-config configs/solvers/default.toml
  ```
  - `--solver-config`: Path to solver parameters and preconditioner specs.
  - *Note: In standalone mode, neural solvers must specify an explicit `checkpoint_path`.*

- **Compare preconditioners** (pipeline mode):
  ```bash
  uv run compare-preconditioners --solver-config configs/solvers/default.toml --comparison-run output/training/comparison_run.json
  ```
  - `--comparison-run`: Path to the `comparison_run.json` produced by `train-multiple`. Enables resolution of experiment IDs to their latest checkpoints.

- **Run full experiment matrix** (data + train):
  ```bash
  uv run run-experiments --config configs/experiments.toml
  ```
  - `--config`: Path to the master experiments registry.
  - `--force`: Force re-training even if checkpoints already exist.

- **Batch training with aggregate metrics**:
  ```bash
  uv run train-multiple configs/experiments.toml --metric eval/rel_error
  ```
  - `config`: Path to the master experiments registry.
  - `--metric`: The MLflow metric to compare and plot across experiments.

### Comparison Workflows

The repository uses **`compare-preconditioners`** as the unified tool for evaluating solver performance. It supports two modes:

1.  **Standalone**: Benchmarks specific checkpoints against classical baselines (Jacobi, ILU).
2.  **Pipeline**: Automates the comparison of all experiments defined in a training batch, using the metadata generated during the `train-multiple` phase.

- **Optional MLflow logging**:
  ```bash
  # Add --enable-mlflow to supported CLI commands (e.g., predict)
  uv run predict --config configs/models/linear.toml --enable-mlflow
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

The `output_root` is the single source of truth for all experiment artifacts:
- MLflow tracking: `{output_root}/mlruns/mlflow.db`
- MLflow artifacts: `{output_root}/mlartifacts/{experiment_id}/{run_id}/`
- Checkpoints: `{output_root}/checkpoints/{dataset_id}/{model_name}.ckpt`
- Comparison tracking: `/data/projects/graph-cg/data/comparisons/mlflow.db`
- Comparison artifacts: `/data/projects/graph-cg/data/comparisons/mlartifacts/{exp_id}/{run_id}/`

Processed datasets live separately in the `processed_root` directory (configurable per dataset config).

All path resolution is handled by three simple functions in `src/neuralls/configuration/paths.py`:
- `resolve_project_root()` - Project base directory
- `resolve_output_root()` - Master output directory (drives MLflow paths)
- `resolve_processed_root()` - Processed data directory
