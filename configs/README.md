# Configuration System

This directory contains all configuration files for the dl-experiments project using a **dataset-centric organization**.

## Directory Structure

```
configs/
  experiments.toml          # Master experiment registry (NEW FORMAT)
  datasets/                 # Dataset generation configs (shared)
    test-solutions.toml
    collect-504-solutions.toml
    neutral-test-*.toml
  models/                   # Model architecture configs (shared)
    ffnn.toml
    gnn.toml
    linear.toml
  solvers/                  # Solver comparison configs (shared)
    default.toml
    compare-all-on-neutral-test.toml
```

## New Configuration Format

### experiments.toml (Master Registry)

**Format**: Array of `[[experiment]]` entries, each defining:
- `id`: Unique experiment identifier
- `dataset`: Reference to dataset config (in `datasets/`)
- `model`: Reference to model config (in `models/`)
- `solver`: Reference to solver config (in `solvers/`)
- `checkpoint_path`: **Explicit** path to trained checkpoint

**Example**:
```toml
project_root = ".."
output_dir = "/data/projects/graph-cg/data/output"

[[experiment]]
id = "ffnn_test_solutions"
dataset = "test-solutions"
model = "ffnn"
solver = "default"
checkpoint_path = "/data/projects/graph-cg/data/output/test-solutions/NormScaledConstantWidthFFNN/checkpoints/ffnn.ckpt"

[[experiment]]
id = "gnn_test_solutions"
dataset = "test-solutions"  # SAME dataset
model = "gnn"                # Different model
solver = "default"           # SAME solver config
checkpoint_path = "/data/projects/graph-cg/data/output/test-solutions/GNN/checkpoints/gnn.ckpt"
```

**Benefits**:
- ✅ Zero config duplication (one config per unique dataset/model/solver)
- ✅ Explicit checkpoint paths (no auto-resolution magic)
- ✅ Easy to add new experiments (just add one [[experiment]] entry)
- ✅ Clear relationships: experiment = dataset + model + solver + checkpoint

### Dataset Configs (`datasets/`)

**Shared across experiments** - define data generation settings:

```toml
# datasets/neutral-test-45x15-displacements.toml
[flow]

[source]
matrix_path = "/data/projects/graph-cg/data/raw/SpectralData/45x15-displacements/stiffness/subdomain_1_Kaa.txt"

[generation]
normalize = "matrix"
shuffle = false
seed = 42

[[generation.strategy]]
name = "neutral_ones"  # x=ones, b=A@x (unbiased baseline)
samples = 1

[output]
processed_dir = "/data/projects/graph-cg/data/processed/neutral-tests"
```

### Model Configs (`models/`)

**Shared across experiments** - define neural network architecture:

```toml
# models/ffnn.toml
[SESSION]
seed = 42
precision = "float64"

[MODEL]
name = "NormScaledConstantWidthFFNN"
module_path = "dlkit.nn.ffnn"
hidden_size = 500
num_layers = 1

[TRAINING]
max_epochs = 500

[TRAINING.optimizer]
lr = 1e-2
name = "AdamW"

[DATASET]
name = "FlexibleDataset"

[DATAMODULE]
name = "InMemoryModule"

[DATAMODULE.dataloader]
batch_size = 32
```

### Solver Configs (`solvers/`)

**Shared across experiments** - define solver comparison settings:

```toml
# solvers/compare-all-on-neutral-test.toml
[general]
# Test on shared neutral dataset
matrix_path = "/data/projects/graph-cg/data/processed/neutral-tests/neutral-test-45x15-displacements/normalized.npz"
rhs_path = "/data/projects/graph-cg/data/processed/neutral-tests/neutral-test-45x15-displacements/normalized.npz"
rtol = 1.0e-9
atol = 1.0e-14
max_iterations = 200

# Baseline preconditioners
[[solvers]]
name = "none"
type = "none"

[[solvers]]
name = "jacobi"
type = "jacobi"

# Neural preconditioners (reference experiments by ID)
[[solvers]]
name = "neural_ffnn"
type = "neural"
experiment = "ffnn_test_solutions"  # Checkpoint auto-resolved from experiments.toml
limit_iters = 1
fallback = "identity"
```

## Key Features

### 1. Zero Duplication
- One dataset config per unique dataset
- One model config per unique architecture
- One solver config per comparison scenario
- Experiments just reference configs by name

### 2. Explicit Checkpoints
- Every experiment specifies exact checkpoint path
- No auto-resolution guessing
- Full reproducibility

### 3. Experiment References in Solver Configs
- Neural solvers can reference experiments by ID
- Checkpoints automatically resolved from `experiments.toml`
- Easy to compare multiple neural preconditioners

### 4. Neutral Test Datasets
- Shared `neutral-test-*.toml` configs generate x=ones baselines
- Consistent test data across all experiments
- No training data bias in comparisons

### 5. Scale Metadata Persistence
- All datasets save normalization metadata
- Enables denormalization of predictions
- Full auditability of normalization parameters

## Usage

### Training a Model
```bash
# Load experiment from experiments.toml
uv run python src/neuralls/cli/train_model.py ffnn_test_solutions

# Or specify configs directly
uv run python src/neuralls/cli/train_model.py \
  --model-config configs/models/ffnn.toml \
  --data-config configs/datasets/test-solutions.toml
```

### Generating a Dataset
```bash
# Generate neutral test dataset
uv run python src/neuralls/cli/process_data.py \
  configs/datasets/neutral-test-45x15-displacements.toml

# Generate training dataset
uv run python src/neuralls/cli/process_data.py \
  configs/datasets/collect-504-solutions.toml
```

### Comparing Solvers
```bash
# Compare all neural preconditioners + baselines on neutral test
uv run python src/neuralls/cli/compare_preconditioners.py \
  --solver-config configs/solvers/compare-all-on-neutral-test.toml
```

### Adding a New Experiment

**Step 1**: Create/reuse configs (if needed):
```bash
# Create new model config (if needed)
cp configs/models/ffnn.toml configs/models/my_new_model.toml
# Edit my_new_model.toml...
```

**Step 2**: Add experiment entry to `experiments.toml`:
```toml
[[experiment]]
id = "my_new_experiment"
dataset = "test-solutions"  # Reuse existing dataset
model = "my_new_model"      # Reference new model
solver = "default"          # Reuse existing solver
checkpoint_path = "/data/.../checkpoints/my_model.ckpt"  # After training
```

**Step 3**: Train:
```bash
uv run python src/neuralls/cli/train_model.py my_new_experiment
```

## Migration from Old Format

### Before (OLD):
```
configs/experiments/
  ffnn_test_solutions/
    model.toml
    data.toml              -> pointer to datasets/test-solutions.toml
    solver.toml            -> hardcoded checkpoint_path
  gnn_test_solutions/
    model.toml
    data.toml              -> pointer to datasets/test-solutions.toml (DUPLICATE!)
    solver.toml            -> hardcoded checkpoint_path (DUPLICATE!)
```

### After (NEW):
```
configs/
  datasets/
    test-solutions.toml    (ONE config, shared)
  models/
    ffnn.toml             (ONE config per model)
    gnn.toml
  solvers/
    default.toml          (ONE config, shared)
  experiments.toml        (All experiments in one file)
```

## What is Norm Scaling? ⭐

The **NormScaledFFNN wrapper** provides input/output normalization:

1. **Normalize input**: `b_scaled = b / ||b||`
2. **Predict**: `x_scaled = model(b_scaled)`
3. **Rescale output**: `x = x_scaled * ||b||`

This enforces homogeneous scaling for the `Ax = b` problem and improves training stability.

**Available wrappers:**
- `NormScaledLinearFFNN` - Single linear layer
- `NormScaledConstantWidthFFNN` - Multi-layer constant-width FFNN

## Checkpoint Configuration

DLKit uses **two separate checkpoint fields**:

1. **`MODEL.checkpoint`** - For inference only (model weights)
2. **`TRAINING.resume_from_checkpoint`** - For resuming training (full state)

**Example**:
```toml
[MODEL]
name = "NormScaledLinearFFNN"
checkpoint = "/path/to/model.ckpt"  # For inference

[TRAINING]
# Uncomment to resume training
# resume_from_checkpoint = "/path/to/model.ckpt"
```

## Advanced: Neutral Test Datasets

Neutral test datasets provide unbiased baselines using `x=ones, b=A@x`:

**Generate**:
```bash
uv run python src/neuralls/cli/process_data.py \
  configs/datasets/neutral-test-45x15-displacements.toml
```

**Use in comparison**:
```toml
# configs/solvers/compare-all-on-neutral-test.toml
[general]
matrix_path = "/data/.../neutral-tests/.../normalized.npz"
rhs_path = "/data/.../neutral-tests/.../normalized.npz"

[[solvers]]
name = "neural_ffnn"
type = "neural"
experiment = "ffnn_test_solutions"  # Auto-resolves checkpoint
```

**Benefits**:
- Consistent across all experiments
- No training data bias
- Simple, known solution (x=ones)
