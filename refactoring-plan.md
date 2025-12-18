# Plan: Neutral Test RHS + Config Reorganization

## Overview
Redesign the testing and configuration architecture to:
1. Create standalone neutral test datasets (x=ones) independent of training data generation
2. Save matrix_norm in all normalized.npz files for denormalization
3. Make checkpoint_path optional in solver.toml with auto-resolution from workspace
4. Update solver.toml to reference neutral test datasets instead of training data

## Core Problems Identified

### Problem 1: Matrix Norm Not Persisted
- **Location**: `src/neuralls/generation/orchestration.py:392-397`
- **Issue**: `build_dataset()` saves normalized.npz WITHOUT matrix_norm metadata
- **Impact**: Cannot denormalize or understand the normalization scale later

### Problem 2: No Neutral Test RHS Workflow
- **Current State**: Synthetic test (x=ones, b=A@x) generated ad-hoc in prediction workflow
- **Location**: `src/neuralls/diagnostics/synthetic.py:generate_synthetic_test_case()`
- **Issue**: Ugly coupling between prediction and test generation; no reusable test dataset
- **Impact**: Each solver comparison lacks objective neutral baseline

### Problem 3: Hardcoded Solver Config Paths
- **Location**: `configs/experiments/default/solver.toml:6-7, 31`
- **Issue**:
  - `matrix_path` and `rhs_path` point to specific training dataset
  - `checkpoint_path` hardcoded to specific experiment's checkpoint
- **Impact**: Must copy/edit solver.toml for each experiment; error-prone

## Solution Architecture

### 1. Neutral Test Dataset Generation

**New Data Config Type**: Create `data-configs/neutral-test-{matrix-name}.toml`

```toml
[DATASET]
name = "neutral-test-laplacian504"
strategy = "neutral_test"  # New strategy type

[DATASET.matrix]
path = "/data/projects/graph-cg/data/raw/laplacian_504.txt"

[DATASET.output]
dir = "/data/projects/graph-cg/data/processed/neutral-tests/laplacian504"

[DATASET.normalization]
type = "matrix"  # Match training normalization

[DATASET.test_config]
# Generate x=ones, compute b=A@x in normalized space
num_samples = 1
```

**Implementation**:
- Create new strategy `neutral_test` in generation strategies
- Add to strategy registry
- Generates: normalized.npz with {matrix, rhs, solutions, matrix_norm}
- One dataset per matrix, reusable across all experiments

**Critical Files**:
- NEW: `src/neuralls/generation/strategies/neutral_test.py`
- MODIFY: `src/neuralls/generation/orchestration.py` (add matrix_norm saving)
- NEW: `data-configs/neutral-test-*.toml` (one per matrix)

### 2. Persist Matrix Norm in All Datasets

**Changes to `build_dataset()`** (`src/neuralls/generation/orchestration.py:390-400`):

```python
# Step 8: Persist dataset with matrix_norm
normalized_file = dataset_dir_path / "normalized.npz"

# Compute matrix_norm for saving
if normalize == "spectral":
    from ..math_utils import calculate_spectral_norm, compute_dim_scale
    spectral_norm = calculate_spectral_norm(matrix)
    dimension_scale = compute_dim_scale(matrix.shape[0])
    matrix_norm_value = spectral_norm * dimension_scale
elif normalize == "matrix":
    from ..math_utils import compute_matrix_scale
    matrix_norm_value, _ = compute_matrix_scale(matrix)
elif normalize == "diagonal":
    matrix_norm_value = None  # Diagonal doesn't have single scalar
elif normalize in ("none", "rhs"):
    matrix_norm_value = None
else:
    matrix_norm_value = None

# Save with matrix_norm
save_dict = {
    "matrix": raw_samples.matrix,
    "rhs": raw_samples.rhs,
    "solutions": raw_samples.solutions,
}
if matrix_norm_value is not None:
    save_dict["matrix_norm"] = matrix_norm_value

np.savez(normalized_file, **save_dict)
```

**Critical Files**:
- MODIFY: `src/neuralls/generation/orchestration.py:390-400`

### 3. Auto-Resolve Checkpoint Path (Per-Experiment Context)

**Key Principle**: Checkpoint resolution happens at **runtime per-experiment**, not in solver.toml.
- Solver.toml remains experiment-agnostic and reusable
- Each experiment execution resolves checkpoints using its own workspace context

**Resolution Mechanism**:

```
Runtime Context Flow:
1. Experiment starts (e.g., ffnn_collect_504_solutions)
2. Has workspace: /data/output/collect-504/NormScaledFFNN/
3. Loads solver.toml (shared, no checkpoint_path specified)
4. For each neural solver spec:
   a. Check if checkpoint_path specified in solver.toml
   b. If not, resolve from THIS experiment's workspace.checkpoint_dir
   c. Apply selection strategy for multiple checkpoints
```

**Changes to Solver Config Schema**:

```python
# src/neuralls/configuration/solver_models.py
class SolverSpecConfig(BaseModel):
    name: str
    type: str
    # Neural preconditioner fields (all optional for non-neural solvers)
    limit_iters: int | None = None
    fallback: str | None = None
    checkpoint_path: Path | None = None  # OPTIONAL - resolved at runtime if omitted
    checkpoint_selection: str | None = "latest"  # "latest", "best", or explicit filename
```

**Resolution Strategy** in `src/neuralls/workflows/comparison.py`:

```python
def _resolve_checkpoint_for_solver(
    solver_spec: SolverSpecConfig,
    experiment_workspace: ExperimentWorkspace,
) -> Path | None:
    """Resolve checkpoint using experiment context.

    Resolution priority:
    1. Explicit checkpoint_path in solver.toml (absolute or relative)
    2. Auto-discover from experiment's workspace.checkpoint_dir
    3. None (solver doesn't need checkpoint)

    Args:
        solver_spec: Solver specification from solver.toml
        experiment_workspace: Runtime experiment workspace (provides checkpoint_dir)

    Returns:
        Resolved checkpoint path or None
    """
    if solver_spec.type != "neural":
        # Non-neural solvers don't need checkpoints
        return None

    # Priority 1: Explicit path in solver.toml
    if solver_spec.checkpoint_path:
        path = Path(solver_spec.checkpoint_path)
        if path.is_absolute():
            return path if path.exists() else None
        # Relative path: resolve from solver.toml location or project root
        # (depends on your preference for relative path base)
        return path if path.exists() else None

    # Priority 2: Auto-discover from experiment workspace
    ckpt_dir = experiment_workspace.checkpoint_dir
    if not ckpt_dir.exists():
        logger.warning(f"Checkpoint directory not found: {ckpt_dir}")
        return None

    # Selection strategy
    strategy = solver_spec.checkpoint_selection or "latest"

    if strategy == "latest":
        # Most recently modified .ckpt file
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: p.stat().st_mtime)

    elif strategy == "best":
        # Look for checkpoint with "best" in name (common pattern)
        ckpts = list(ckpt_dir.glob("*best*.ckpt"))
        if ckpts:
            return ckpts[0]
        # Fallback to latest
        all_ckpts = list(ckpt_dir.glob("*.ckpt"))
        return max(all_ckpts, key=lambda p: p.stat().st_mtime) if all_ckpts else None

    elif strategy:
        # Explicit filename
        ckpt = ckpt_dir / strategy
        return ckpt if ckpt.exists() else None

    return None


def run_comparisons(
    specs: Iterable[ComparisonSpec],
    params: ComparisonParams,
) -> list[ComparisonOutcome]:
    """Run comparisons with per-experiment checkpoint resolution."""
    outcomes: list[ComparisonOutcome] = []
    for spec in specs:
        enable_mlflow = _get_mlflow_enabled(spec)
        mlflow_state = _start_comparison_run(spec, enable_mlflow)

        try:
            solver_cfg = load_solver_config(spec.solver_config)

            # RESOLVE CHECKPOINTS USING THIS EXPERIMENT'S WORKSPACE
            resolved_specs = []
            for solver_spec in solver_cfg.solvers:
                if solver_spec.type == "neural":
                    # Resolve checkpoint for this specific experiment
                    checkpoint = _resolve_checkpoint_for_solver(
                        solver_spec,
                        spec.workspace  # Current experiment's workspace!
                    )
                    if checkpoint is None:
                        logger.warning(
                            f"No checkpoint found for neural solver '{solver_spec.name}' "
                            f"in experiment '{spec.name}'"
                        )
                    # Create resolved copy with checkpoint path
                    resolved_spec = solver_spec.model_copy(
                        update={"checkpoint_path": checkpoint}
                    )
                    resolved_specs.append(resolved_spec)
                else:
                    resolved_specs.append(solver_spec)

            # Build preconditioner configs with resolved checkpoints
            precond_configs = build_preconditioner_configs_from_specs(resolved_specs)

            result = compare_preconditioners(
                general_params=solver_cfg.general,
                preconditioner_configs=precond_configs,
                output_root=spec.output_dir or spec.workspace.root_dir,
                figures_root=spec.figures_dir or spec.workspace.figures_dir,
            )
        except Exception as exc:
            # ... error handling
        # ... rest of function
```

**Example: Multiple Checkpoints Per Experiment**

```
Experiment: ffnn_collect_504_solutions
Workspace: /data/output/collect-504/NormScaledFFNN/
Checkpoints:
  - epoch=50-step=1000.ckpt  (latest)
  - best-val-loss.ckpt       (best)
  - final-model.ckpt

Solver.toml options:
1. checkpoint_path omitted + checkpoint_selection="latest"
   → Uses epoch=50-step=1000.ckpt

2. checkpoint_path omitted + checkpoint_selection="best"
   → Uses best-val-loss.ckpt

3. checkpoint_path omitted + checkpoint_selection="final-model.ckpt"
   → Uses final-model.ckpt

4. checkpoint_path="/other/experiment/checkpoint.ckpt"
   → Uses explicit path (cross-experiment reuse)
```

**Shared Solver.toml Across Experiments**:

```toml
# configs/experiments/default/solver.toml
# This config is SHARED by all experiments using the same matrix

[general]
matrix_path = "/data/neutral-tests/laplacian504/normalized.npz"
rhs_path = "/data/neutral-tests/laplacian504/normalized.npz"
rtol = 1.0e-9
# ... other params

[[solvers]]
name = "neural"
type = "neural"
limit_iters = 1
fallback = "identity"
# checkpoint_path OMITTED - resolved per-experiment at runtime
# checkpoint_selection = "latest"  # Optional: override default
```

**When Experiments Run**:
- Experiment A: Uses its own workspace → finds checkpoints in A's checkpoint_dir
- Experiment B: Uses its own workspace → finds checkpoints in B's checkpoint_dir
- Both use same solver.toml, but resolve different checkpoints!

**Critical Files**:
- MODIFY: `src/neuralls/configuration/solver_models.py` (add optional checkpoint_path + checkpoint_selection)
- MODIFY: `src/neuralls/workflows/comparison.py` (add _resolve_checkpoint_for_solver + call it in run_comparisons)
- MODIFY: `src/neuralls/preconditioner_factory.py` (expect resolved checkpoint paths from workflow layer)

### 4. Reorganize Config Structure (Experiment-Centric with Embedded Solver Config)

**New Architecture**: Embed solver config in experiments.toml, organize experiments by dataset

**Key Principle**: "Experiment = Dataset + Neural Model"
- One experiment directory per dataset
- Multiple model configs inside each experiment
- experiments.toml defines all experiments and solver comparisons
- Solver config embedded in experiments.toml (or separate solver.toml at same level)

**New Repository Structure**:

```
configs/
  experiments.toml          # Master config: experiments + solver settings
  experiments/
    collect_504/
      data.toml              # Dataset definition (strategy, matrix, etc.)
      models/
        ffnn.toml            # FFNN architecture + training
        gnn.toml             # GNN architecture + training
        linear.toml          # Linear model
    test_solutions/
      data.toml
      models/
        ffnn.toml
        gnn.toml
    collect_2040/
      data.toml
      models/
        ffnn.toml
```

**New experiments.toml Format**:

```toml
# experiments.toml - Single source of truth

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

[[experiments]]
id = "collect_504_ffnn"
data_config = "experiments/collect_504/data.toml"
model_config = "experiments/collect_504/models/ffnn.toml"

[[experiments]]
id = "collect_504_gnn"
data_config = "experiments/collect_504/data.toml"     # SAME dataset
model_config = "experiments/collect_504/models/gnn.toml"  # Different model

[[experiments]]
id = "collect_504_linear"
data_config = "experiments/collect_504/data.toml"
model_config = "experiments/collect_504/models/linear.toml"

[[experiments]]
id = "test_solutions_ffnn"
data_config = "experiments/test_solutions/data.toml"
model_config = "experiments/test_solutions/models/ffnn.toml"

# ============================================================================
# SOLVER COMPARISON CONFIGURATION
# ============================================================================

[solver]
# Neutral test dataset for objective comparison (matrix-specific)
matrix_path = "/data/projects/graph-cg/data/processed/neutral-tests/laplacian504/normalized.npz"
rhs_path = "/data/projects/graph-cg/data/processed/neutral-tests/laplacian504/normalized.npz"

# CG solver parameters
rtol = 1.0e-9
atol = 1.0e-14
max_iterations = 200
stopping_criterion = "residual_norm"
reorthogonalize = "full"

# ============================================================================
# PRECONDITIONER SPECIFICATIONS
# Test multiple neural preconditioners + baselines in one comparison
# ============================================================================

[[solver.preconditioners]]
name = "none"
type = "none"

[[solver.preconditioners]]
name = "jacobi"
type = "jacobi"

[[solver.preconditioners]]
name = "ilu"
type = "ilu"

[[solver.preconditioners]]
name = "neural_ffnn"
type = "neural"
experiment = "collect_504_ffnn"    # References experiment for checkpoint
checkpoint = "epoch=100-val_loss=0.001.ckpt"  # EXPLICIT filename (optional, recommended)
# If checkpoint omitted: strict validation searches experiment checkpoint dir
limit_iters = 1
fallback = "identity"

[[solver.preconditioners]]
name = "neural_gnn"
type = "neural"
experiment = "collect_504_gnn"     # Different experiment → different checkpoint
# checkpoint OMITTED: will search checkpoint_dir with strict validation
limit_iters = 1
fallback = "identity"

[[solver.preconditioners]]
name = "neural_linear"
type = "neural"
experiment = "collect_504_linear"
checkpoint = "final.ckpt"  # Explicit filename
limit_iters = 1
fallback = "identity"

# Can also reference experiments from different datasets:
[[solver.preconditioners]]
name = "neural_test_solutions"
type = "neural"
experiment = "test_solutions_ffnn"
# Or use absolute path for cross-experiment reuse:
# checkpoint_path = "/absolute/path/to/specific/checkpoint.ckpt"
limit_iters = 1
fallback = "identity"
```

**Checkpoint Resolution with Strict Validation**:

```python
def _resolve_checkpoint_for_preconditioner(
    precond_spec: dict,  # Preconditioner from experiments.toml
    experiments_map: dict[str, ExperimentConfig],  # All experiments indexed by ID
) -> Path:
    """Resolve checkpoint using experiment reference with strict validation.

    Resolution priority:
    1. Explicit checkpoint_path (absolute path)
    2. Explicit checkpoint filename (relative to experiment checkpoint_dir)
    3. Auto-discover from experiment checkpoint_dir (STRICT: exactly 1 checkpoint)

    Raises:
        ValueError: If checkpoint resolution fails or is ambiguous
    """
    if precond_spec["type"] != "neural":
        raise ValueError(
            f"Checkpoint resolution called for non-neural preconditioner "
            f"'{precond_spec['name']}'"
        )

    # Priority 1: Explicit absolute path
    if "checkpoint_path" in precond_spec:
        ckpt_path = Path(precond_spec["checkpoint_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for preconditioner '{precond_spec['name']}': "
                f"{ckpt_path}"
            )
        return ckpt_path

    # Priority 2 & 3: Resolve from experiment reference
    exp_id = precond_spec.get("experiment")
    if not exp_id:
        raise ValueError(
            f"Neural preconditioner '{precond_spec['name']}' must specify "
            "'experiment' or 'checkpoint_path'"
        )

    experiment = experiments_map.get(exp_id)
    if not experiment:
        raise ValueError(f"Unknown experiment '{exp_id}' for preconditioner '{precond_spec['name']}'")

    ckpt_dir = experiment.workspace.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found for experiment '{exp_id}': {ckpt_dir}"
        )

    # Priority 2: Explicit checkpoint filename (relative to checkpoint_dir)
    if "checkpoint" in precond_spec:
        ckpt_filename = precond_spec["checkpoint"]
        ckpt_path = ckpt_dir / ckpt_filename
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for preconditioner '{precond_spec['name']}': "
                f"{ckpt_path} (experiment: {exp_id})"
            )
        return ckpt_path

    # Priority 3: Auto-discover with STRICT validation
    ckpts = list(ckpt_dir.glob("*.ckpt"))

    if len(ckpts) == 0:
        raise FileNotFoundError(
            f"No checkpoints found for preconditioner '{precond_spec['name']}' "
            f"in {ckpt_dir} (experiment: {exp_id}). "
            f"Either train the model first or specify 'checkpoint' explicitly."
        )

    if len(ckpts) > 1:
        ckpt_list = "\n  ".join(str(p.name) for p in sorted(ckpts))
        raise ValueError(
            f"Ambiguous checkpoint for preconditioner '{precond_spec['name']}' "
            f"(experiment: {exp_id}). Found {len(ckpts)} checkpoints in {ckpt_dir}:\n  {ckpt_list}\n"
            f"Please specify 'checkpoint' explicitly in experiments.toml:\n"
            f"  checkpoint = \"{ckpts[0].name}\"  # Or choose another"
        )

    # Exactly 1 checkpoint found
    logger.info(
        f"Auto-discovered checkpoint for '{precond_spec['name']}': {ckpts[0].name}"
    )
    return ckpts[0]
```

**Benefits of Strict Validation**:

1. **Deterministic**: Explicit filenames guarantee reproducibility
2. **Fails Loudly**: Ambiguous cases raise clear errors with actionable messages
3. **Safe Default**: Auto-discovery works only when unambiguous (single checkpoint)
4. **Actionable Errors**: Error messages show exact checkpoints and suggest fix
5. **No Silent Failures**: Never uses wrong checkpoint due to pattern matching

**Benefits**:

1. **Single Source of Truth**: experiments.toml defines everything
2. **No Redundancy**: One experiment dir per dataset, multiple models inside
3. **Flexible Comparisons**: Test all neural preconditioners together
4. **Explicit Relationships**: Preconditioner explicitly references experiment
5. **DRY Principle**: Shared data.toml, distinct model.toml files
6. **Clear Semantics**: "experiment = dataset + model" is explicit in structure

**Example Workflows**:

```bash
# Train all models on collect_504 dataset
for experiment in collect_504_{ffnn,gnn,linear}; do
  uv run python src/neuralls/cli/train_model.py $experiment
done

# Compare ALL neural preconditioners + baselines
uv run python src/neuralls/cli/compare_methods.py
# Automatically tests: none, jacobi, ilu, neural_ffnn, neural_gnn, neural_linear, neural_test_solutions
# All on the same neutral test dataset
```

**Migration Impact**:

```
OLD Structure:
configs/experiments/
  ffnn_collect_504/model.toml
  ffnn_collect_504/data.toml
  gnn_collect_504/model.toml
  gnn_collect_504/data.toml     # DUPLICATE!
  linear_collect_504/model.toml
  linear_collect_504/data.toml  # DUPLICATE!

NEW Structure:
configs/experiments/
  collect_504/data.toml           # ONCE!
  collect_504/models/ffnn.toml
  collect_504/models/gnn.toml
  collect_504/models/linear.toml
```

**Critical Files**:
- MODIFY: `configs/experiments.toml` (add solver section + reorganize experiments)
- REORGANIZE: `configs/experiments/*/` (group by dataset, models inside)
- MODIFY: `src/neuralls/configuration/loader.py` (parse new experiments.toml format)
- MODIFY: `src/neuralls/workflows/comparison.py` (use experiment references for checkpoints)
- REMOVE: Separate `solver.toml` files (merged into experiments.toml)

## Implementation Steps

### Step 1: Add Matrix Norm Persistence (Immediate Fix)
1. Modify `src/neuralls/generation/orchestration.py:build_dataset()`
2. Add logic to compute and save matrix_norm based on normalization type
3. Save matrix_norm in normalized.npz for all new datasets

**Estimated Impact**: Low risk, high value
**Files**: `src/neuralls/generation/orchestration.py:390-400`

### Step 2: Create Neutral Test Strategy & Data Configs
1. Create `src/neuralls/generation/strategies/neutral_test.py`
2. Implement strategy that generates x=ones, b=A@x in normalized space
3. Register strategy in strategy registry
4. Create data configs: `data-configs/neutral-test-laplacian504.toml`, etc.
5. Generate neutral test datasets for each matrix

**Estimated Impact**: Medium - new code, clean separation
**Files**:
- NEW: `src/neuralls/generation/strategies/neutral_test.py`
- NEW: `data-configs/neutral-test-*.toml`

### Step 3: Reorganize Config Structure
1. Restructure `configs/experiments/` directories:
   - Group by dataset: `collect_504/`, `test_solutions/`, etc.
   - Move model configs inside: `collect_504/models/{ffnn,gnn,linear}.toml`
   - One `data.toml` per dataset directory
2. Update `configs/experiments.toml`:
   - Redefine experiments with new paths
   - Add `[solver]` section with neutral test paths
   - Add `[[solver.preconditioners]]` for all preconditioners
   - Use `experiment` references for neural preconditioners
3. Remove redundant data config copies

**Estimated Impact**: High - major reorganization but cleaner architecture
**Files**:
- REORGANIZE: `configs/experiments/*/`
- MODIFY: `configs/experiments.toml`
- REMOVE: Duplicate data configs

### Step 4: Update Config Loader
1. Modify `src/neuralls/configuration/loader.py` to parse new experiments.toml format:
   - Parse `[solver]` section
   - Parse `[[solver.preconditioners]]` array
   - Maintain experiment loading (paths changed but structure same)
2. Update Pydantic models if needed:
   - Add solver config models
   - Add preconditioner config models

**Estimated Impact**: Medium - touches core config loading
**Files**:
- MODIFY: `src/neuralls/configuration/loader.py`
- MODIFY: `src/neuralls/configuration/solver_models.py`

### Step 5: Update Comparison Workflow
1. Modify `src/neuralls/workflows/comparison.py`:
   - Read solver config from experiments.toml (not separate file)
   - Implement `_resolve_checkpoint_for_preconditioner()` using experiment references
   - Create experiments_map for lookup
   - Resolve all neural preconditioner checkpoints before comparison
2. Update comparison CLI to use new config format

**Estimated Impact**: Medium - changes comparison execution
**Files**:
- MODIFY: `src/neuralls/workflows/comparison.py`
- MODIFY: `src/neuralls/cli/compare_methods.py`

### Step 6: Clean Up Prediction Workflow (Optional)
1. Remove synthetic test generation from prediction workflow
2. Update to use neutral test dataset if needed
3. Simplify prediction.py

**Estimated Impact**: Low - cleanup, no new functionality
**Files**: `src/neuralls/workflows/prediction.py`

### Step 7: Migration & Testing
1. Migrate existing configs to new structure
2. Regenerate neutral test datasets
3. Verify all experiments load correctly
4. Run comparison workflow to verify checkpoint resolution
5. Update documentation

**Estimated Impact**: Critical - ensures smooth transition

## Testing Strategy

1. **Unit Tests**:
   - Test matrix_norm saving for each normalization type
   - Test neutral_test strategy generates correct data
   - Test checkpoint auto-resolution logic

2. **Integration Tests**:
   - Generate neutral test dataset
   - Run solver comparison using neutral test
   - Verify denormalization works with saved matrix_norm

3. **Regression Tests**:
   - Existing experiments should still work
   - Backward compatibility: explicit checkpoint_path still works

## Critical Files Summary

### To Create:
- `src/neuralls/generation/strategies/neutral_test.py`
- `data-configs/neutral-test-laplacian504.toml` (and others per matrix)

### To Modify:
- `src/neuralls/generation/orchestration.py` (matrix_norm persistence)
- `src/neuralls/configuration/solver_models.py` (optional checkpoint_path)
- `src/neuralls/workflows/comparison.py` (checkpoint auto-resolution)
- `src/neuralls/preconditioner_factory.py` (use resolver)
- `configs/experiments/default/solver.toml` (remove hardcoded paths)
- `src/neuralls/workflows/prediction.py` (optional cleanup)

### To Review:
- All solver.toml files in experiments
- Data generation tests
- Comparison workflow tests

## Adherence to SOLID Principles

1. **Single Responsibility**:
   - Neutral test generation separated from training data generation
   - Checkpoint resolution separated into dedicated function

2. **Open/Closed**:
   - New strategy added without modifying existing strategies
   - Optional checkpoint_path extends functionality without breaking existing code

3. **Interface Segregation**:
   - Neutral test strategy implements same interface as other strategies
   - Clean separation of concerns

4. **Dependency Inversion**:
   - Checkpoint resolver abstracts path resolution
   - Config references tests by path, not coupled to specific datasets

## Benefits

1. **Objectivity**: Neutral test RHS (x=ones) provides consistent baseline across all experiments
2. **Reusability**: One neutral test per matrix, shared across experiments
3. **Maintainability**: No more copying solver.toml per experiment
4. **Denormalization**: matrix_norm saved for proper result interpretation
5. **Clarity**: Clean separation between training data and evaluation data
6. **Flexibility**: Checkpoint auto-resolution reduces config boilerplate

## Migration Path

For existing experiments:
1. Generate neutral test datasets for each matrix
2. Update solver.toml to reference neutral tests
3. Remove checkpoint_path lines (or keep for explicit control)
4. Regenerate or patch existing datasets to include matrix_norm

Backward compatible: Explicit paths still work if specified.
