# Workflows Module

This module provides high-level orchestration for experiments, training, and solver comparison workflows.

## Architecture

The workflows module sits above the core solver and training modules, coordinating multi-step processes:

```
workflows/
├── cg_runner.py         # CG comparison execution (moved from solver/)
├── compare.py           # Single preconditioner comparison orchestration
├── comparison.py        # Batch comparison across experiments
├── runner.py            # Experiment matrix orchestration
├── training.py          # Model training workflow
├── prediction.py        # Inference workflow
├── data.py              # Data loading utilities
├── checkpoints.py       # Checkpoint management
├── results.py           # Result data structures
└── specs.py             # Experiment specifications
```

## Key Workflows

### 1. CG Comparison Runner (`cg_runner.py`)

**Purpose**: Execute CG solver with multiple preconditioners and compare results.

**Moved from**: `solver/comparison.py` (architectural cleanup - comparison orchestration is a workflow concern)

**Key Functions**:
- `run_cg_comparison()` - Run CG with multiple preconditioners
- `format_results_summary()` - Format comparison results for display
- `summarize_best_combinations()` - Identify best-performing preconditioners

**Usage**:
```python
from neuralls.workflows import run_cg_comparison
from neuralls.solver.preconditioners import JacobiPreconditioner, ILUPreconditioner

preconditioners = {
    "jacobi": JacobiPreconditioner(A),
    "ilu": ILUPreconditioner(A),
}

results = run_cg_comparison(A, b, preconditioners=preconditioners)
```

**Design**:
- Type-based routing: contextual and non-linear preconditioners → `flexible_cg`, static linear preconditioners → `pcg`
- No scheduling logic here - handled by `ScheduledPreconditioner` wrapper
- Returns `CGComparisonResult` with convergence metrics

### 2. Preconditioner Comparison (`compare.py`)

**Purpose**: Orchestrate single comparison (matrix + RHS + comparison config → results + plots)

**Key Function**: `compare_preconditioners()`

**Workflow** (10 steps):
1. Validate inputs
2. Resolve paths (matrix, rhs, output, figures)
3. Load and validate linear system
4. Create preconditioners via factory
5. Compute condition numbers for diagnostics
6. Wrap preconditioners with scheduling (iteration limits, fallbacks)
7. Run CG comparison
8. Generate recommendations
9. Generate plots (if enabled)
10. Return result

**Usage**:
```python
from neuralls.workflows.compare import compare_preconditioners
from neuralls.configuration.comparison import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.configuration.preconditioner import StandardPreconditionerConfig

general = ComparisonGeneral(
    params=SolverParams(
        rtol=1.0e-6,
        atol=1.0e-14,
        max_iterations=100,
        stopping_criterion="residual_norm",
        m_max=10,
        breakdown_tol=None,
    ),
    data=ComparisonData(
        matrix_path="data/matrix.txt",
        rhs_path="data/rhs.txt",
    ),
    tracking=None,
    model_store=None,
)

configs = [
    StandardPreconditionerConfig(name="jacobi", type="jacobi"),
    StandardPreconditionerConfig(name="ilu", type="ilu"),
]

result = compare_preconditioners(
    general_params=general,
    preconditioner_configs=configs,
    save_plots=True,
)
```

**Scheduling Example**:
```python
from neuralls.solver.preconditioners import (
    PreconditionerScheduleConfig,
    PreconditionerType,
)

schedule = PreconditionerScheduleConfig(
    limit_iters=10,
    fallback=PreconditionerType.IDENTITY,
)
```

### 3. Batch Comparison (`comparison.py`)

**Purpose**: Run all configured comparisons from the master registry

**Key Function**: `run_comparison_batch()`

**Usage**:
```bash
uv run compare-all <registry.toml>
```
- The selected registry provides MLflow topology, experiment bindings, and the ordered `[[comparisons]]` batch.
- Each comparison profile file contains only solver/data/preconditioner settings.

### 4. Experiment Runner (`runner.py`)

**Purpose**: Orchestrate experiment matrix (data generation + training)

**Key Function**: `run_experiment_matrix()`

**Workflow**:
1. Load experiment bindings from the master registry
2. Compute source code hash for cache invalidation
3. For each experiment:
   - Generate/cache dataset
   - Check for existing checkpoint
   - Train model (if needed)
4. Return results summary

**Usage**:
```bash
uv run run-experiments --config <registry.toml>
```

## Design Principles

### Separation of Concerns

**Solver Layer** (`src/neuralls/solver/`):
- Core algorithms (PCG, Flexible CG)
- Preconditioner implementations
- Mathematical primitives

**Workflow Layer** (`src/neuralls/workflows/`):
- Orchestration and coordination
- Result formatting and analysis
- File I/O and visualization
- Batch processing

**Why This Matters**:
- Solver modules should be pure, testable, composable
- Workflows handle side effects (I/O, plotting, logging)
- Clear dependency direction: workflows depend on solvers, not vice versa

### Scheduling Architecture

Scheduling is handled at the preconditioner layer, not in the comparison orchestration:

```python
from neuralls.solver.preconditioners import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)

# Create scheduled preconditioner that switches after 10 iterations
schedule = PreconditionerScheduleConfig(limit_iters=10)
scheduled = create_scheduled_preconditioner(
    primary=neural_precond,
    schedule=schedule,
)

# Use in comparison (scheduling handled transparently)
results = run_cg_comparison(A, b, preconditioners={"neural": scheduled})
```

**Design Benefits**:
- **Single Responsibility**: Comparison orchestrates CG runs; preconditioner handles its own scheduling
- **Composability**: Scheduled preconditioners work anywhere `Preconditioner` is accepted
- **Testability**: Scheduling logic tested independently from comparison workflow

## Configuration Flow

### Registry + Config System

**Master Registry**:
```toml
[[datasets]]
id = "eig-solutions-smallest"
path = "datasets/eig-solutions-smallest.toml"

[[models]]
id = "symmetric"
path = "models/symmetric.toml"

[[experiments]]
id = "eig-solutions-smallest-symmetric"
dataset = "eig-solutions-smallest"
model = "symmetric"
```

Registry filenames are examples only. Use whichever registry file matches the run you want to execute.

**During Training**:
```toml
# configs/models/linear.toml - Neural architecture
# configs/datasets/collect-504.toml - Dataset generation
```

Model configs rely on default MLflow settings. Tracking topology comes from
the selected registry or runtime environment variables.

**During Comparison**:
```toml
# configs/comparison/linear.toml - Comparison parameters + preconditioner specs
[general]

[general.params]
rtol = 1.0e-6
atol = 1.0e-14
max_iterations = 200

[general.data]
matrix_path = "/path/to/matrix-or-dataset-dir"
rhs_path = "/path/to/rhs-or-dataset-dir"

[[preconditioners]]
name = "neural"
type = "neural"
experiment = "linear_test_solutions"
model_ref = { source = "registered", alias = "@dataset" }
limit_iters = 10
fallback = "identity"
```

**Why Separate**:
- The master registry is the only orchestration input for datasets, models, comparisons, and experiment bindings.
- Comparison profiles stay reusable without becoming a second source of truth for MLflow topology.
- Neural preconditioners stay explicit peers of classical baselines.
- Training and comparison MLflow infrastructure both come from the selected experiments registry.
- Training run identity is passed directly to `dlkit.interfaces.api.execute(...)`:
  experiment name from `[names].training`, run name from the experiment display name plus a readable timestamp.
- Dataset identity comes from dataset config `id`, not filename stems.

### 5. Inference (`prediction.py`)

**Purpose**: Load a checkpoint, run prediction, and persist plots/CSV outputs.

**Interface**:
- Uses `dlkit.interfaces.api.load_model()` as the supported inference entrypoint.
- Does not route inference through `execute()`.
- Loads inference settings via `load_experiment(..., mode="inference")`.
- Returns an `InferenceResult` dataclass rather than a dict.

## Common Patterns

### Creating Scheduled Preconditioners

```python
from neuralls.solver.preconditioners import (
    create_preconditioner,
    create_scheduled_preconditioner,
    PreconditionerScheduleConfig,
    PreconditionerType,
)

# 1. Create base preconditioners from comparison config
primary = create_preconditioner(matrix, primary_config)

# 2. Wrap with scheduling if needed
schedule = PreconditionerScheduleConfig(
    limit_iters=10,
    fallback=PreconditionerType.IDENTITY,
)
scheduled = create_scheduled_preconditioner(
    primary=primary,
    schedule=schedule,
)

# 3. Use anywhere Preconditioner is accepted
z = scheduled.apply(residual, context)
```

### Type-Based CG Routing

The comparison runner automatically routes to the correct CG variant:

```python
# Contextual or non-linear preconditioner → flexible_cg
results = run_cg_comparison(A, b, {"neural": neural_precond})  # -> uses flexible_cg

# Standard preconditioner → pcg
jacobi = JacobiPreconditioner(A)
results = run_cg_comparison(A, b, {"jacobi": jacobi})  # -> uses pcg
```

## Migration Notes

### Breaking Changes

**Functions Moved** (import path changed):
```python
# Before
from neuralls.solver import run_cg_comparison

# After
from neuralls.workflows import run_cg_comparison
```

**Removed Parameters**:
- `apply_every` - Preconditioners are applied continuously (no periodic skipping)
- `first_n` - Redundant with `limit_iters`
- `solver_options` - Replaced by scheduled preconditioner wrappers
- `solver_types` - Replaced by type-based routing

**Current API**:
```python
# Scheduling handled by preconditioner wrapper
from neuralls.solver.preconditioners import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)

scheduled = create_scheduled_preconditioner(
    primary=neural_precond,
    schedule=PreconditionerScheduleConfig(limit_iters=10)
)

results = run_cg_comparison(A, b, preconditioners={"neural": scheduled})
```

## Testing

Run workflow tests:
```bash
pytest tests/workflows/
```

Key test files:
- `test_compare_workflow.py` - Preconditioner comparison orchestration
- `test_runner.py` - Experiment matrix execution
- `tests/solver/test_cg_algorithms.py` - Uses `run_cg_comparison()` (integration test)

## See Also

- `src/neuralls/solver/README.md` - Core solver algorithms
- `src/neuralls/generation/README.md` - Data generation workflows
- `CLAUDE.md` - Project architecture and conventions
