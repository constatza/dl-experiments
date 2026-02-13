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
- Type-based routing: `ContextualPreconditioner` → flexible_cg, others → pcg
- No scheduling logic here - handled by `ScheduledPreconditioner` wrapper
- Returns `CGComparisonResult` with convergence metrics

### 2. Preconditioner Comparison (`compare.py`)

**Purpose**: Orchestrate single comparison (matrix + RHS + solver configs → results + plots)

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
from neuralls.configuration.comparison import GeneralSolverConfig
from neuralls.configuration.preconditioner import StandardPreconditionerConfig

general = GeneralSolverConfig(
    matrix_path="data/matrix.txt",
    rhs_path="data/rhs.txt",
    output_root="output/comparison",
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
# Limit neural preconditioner to first 10 iterations
neural_config = NeuralPreconditionerConfig(
    name="neural",
    type="neural",
    checkpoint_path="model.pt",
    limit_iters=10,        # Switch to fallback after 10 iterations
    fallback="jacobi",     # Use Jacobi as fallback
)
```

### 3. Batch Comparison (`comparison.py`)

**Purpose**: Run comparisons across multiple experiments from `experiments.toml`

**Key Function**: `run_batch_comparison()`

**Usage**:
```bash
uv run python src/neuralls/cli/compare_preconditioners.py \
  --experiments configs/experiments.toml \
  --solver-config configs/solvers/default.toml
```

### 4. Experiment Runner (`runner.py`)

**Purpose**: Orchestrate experiment matrix (data generation + training)

**Key Function**: `run_experiment_matrix()`

**Workflow**:
1. Load experiment definitions from `experiments.toml`
2. Compute source code hash for cache invalidation
3. For each experiment:
   - Generate/cache dataset
   - Check for existing checkpoint
   - Train model (if needed)
4. Return results summary

**Usage**:
```bash
uv run python src/neuralls/cli/run_experiments.py --config configs/experiments.toml
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
from neuralls.solver.preconditioners import create_scheduled_preconditioner

# Create scheduled preconditioner that switches after 10 iterations
scheduled = create_scheduled_preconditioner(
    primary=neural_precond,
    fallback=jacobi_precond,
    limit_iters=10,
)

# Use in comparison (scheduling handled transparently)
results = run_cg_comparison(A, b, preconditioners={"neural": scheduled})
```

**Design Benefits**:
- **Single Responsibility**: Comparison orchestrates CG runs; preconditioner handles its own scheduling
- **Composability**: Scheduled preconditioners work anywhere `Preconditioner` is accepted
- **Testability**: Scheduling logic tested independently from comparison workflow

## Configuration Flow

### Two-Config System

**During Training**:
```toml
# configs/models/linear.toml - Neural architecture
# configs/datasets/collect-504.toml - Dataset generation
```

**During Comparison**:
```toml
# configs/solvers/default.toml - Solver parameters + preconditioner specs
[[solvers]]
name = "neural"
type = "neural"
checkpoint_path = "path/to/model.pt"
limit_iters = 10
fallback = "jacobi"
```

**Why Separate**:
- Experiments don't need solver configs (only generate data + train)
- Solver configs specified at comparison time (flexible testing of checkpoints)
- Single checkpoint can be tested with multiple solver configurations

## Common Patterns

### Creating Scheduled Preconditioners

```python
from neuralls.solver.preconditioners import (
    create_preconditioner,
    create_scheduled_preconditioner,
)

# 1. Create base preconditioners from config
primary = create_preconditioner(matrix, primary_config)
fallback = create_preconditioner(matrix, fallback_config)

# 2. Wrap with scheduling if needed
scheduled = create_scheduled_preconditioner(
    primary=primary,
    fallback=fallback,
    limit_iters=10,  # None = no limit (always use primary)
)

# 3. Use anywhere Preconditioner is accepted
z = scheduled.apply(residual, context)
```

### Type-Based CG Routing

The comparison runner automatically routes to the correct CG variant:

```python
# ContextualPreconditioner (needs iteration context) → flexible_cg
scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=10)
results = run_cg_comparison(A, b, {"scheduled": scheduled})  # → uses flexible_cg

# Standard preconditioner → pcg
jacobi = JacobiPreconditioner(A)
results = run_cg_comparison(A, b, {"jacobi": jacobi})  # → uses pcg
```

## Migration Notes

### Breaking Changes (v2.0)

**Functions Moved** (import path changed):
```python
# Before (v1.x)
from neuralls.solver import run_cg_comparison

# After (v2.0)
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
from neuralls.solver.preconditioners import create_scheduled_preconditioner

scheduled = create_scheduled_preconditioner(
    primary=neural_precond,
    fallback=jacobi_precond,
    limit_iters=10
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
