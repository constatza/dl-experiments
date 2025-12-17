# Training Configurations

This directory contains model-specific training configurations for the graph-cg project.

## Available Configs

### FFNN Models (504-dimensional data)
- **`linear.toml`** - Single linear layer (baseline, no scaling)
- **`ffnn-normscaled.toml`** - NormScaled linear FFNN (default) ⭐
- **`ffnn-constant.toml`** - NormScaled constant-width FFNN (3 layers, 288 hidden) ⭐

### Graph Neural Networks
- **`gnn.toml`** - GATv2 graph neural network

## Flow Metadata & Unified Paths

Model configs are now data agnostic — they only describe the model and trainer.
Dataset selection lives entirely inside the paired data configs under
`graph-cg/data-configs/`:

```toml
# data-configs/collect-504.toml
[flow]
id = "spectral-baseline"
dataset = "collect-504-norm"
```

All orchestrators (`scripts/train_model.py`, `scripts/predict.py`, `scripts/compare_methods.py`, PCA
helpers) accept a `--data-config` argument and call
`load_experiment(model_config, data_config, solver_config)` to resolve unified paths with **Pydantic-validated configs**:

- Dataset artefacts (`data/processed/<flow>/<dataset>/...`)
- Training runs (`output/<flow>/train/<dataset>/<run_id>`)
- Prediction dumps (`output/<flow>/predict/<dataset>`)
- Solver comparison reports (`output/<flow>/compare/<dataset>`)
- Figure exports (`output/<flow>/figures/<dataset>`)

To wire up a new dataset, add a data config with the appropriate `[flow]` and
`[generation]` metadata. Any model config can then target it by passing
`--data-config path/to/data-config.toml` on the command line — no per-model
path edits required.

### What is Norm Scaling? ⭐

The **NormScaled wrapper** (`NormScaledFFNN`) is a reusable component that provides input/output normalization:

1. **Normalize input**: `b_scaled = b / ||b||`
2. **Predict**: `x_scaled = model(b_scaled)`
3. **Rescale output**: `x = x_scaled * ||b||`

This enforces homogeneous scaling consistency for the `Ax = b` problem and helps training stability.

**Available wrappers:**
- `NormScaledLinearFFNN` - Wraps single Linear layer
- `NormScaledConstantWidthFFNN` - Wraps multi-layer constant-width FFNN

Both use the same `NormScaledFFNN` base wrapper class in dlkit.

### Checkpoint Configuration

DLKit uses **two separate checkpoint fields** for different purposes:

1. **`MODEL.checkpoint`** - For **inference only** (model weights)
   - Used when running predictions/inference
   - Contains only model architecture weights
   - Specified in the `[MODEL]` section

2. **`TRAINING.resume_from_checkpoint`** - For **resuming training** (full state)
   - Used when continuing interrupted training
   - Contains model weights + optimizer state + scheduler state + epoch counter + global step
   - Specified in the `[TRAINING]` section
   - Commented out by default (uncomment when needed)

**Example:**
```toml
[MODEL]
name = "NormScaledLinearFFNN"
checkpoint = "/path/to/model.ckpt"  # For inference

[TRAINING]
# Uncomment to resume training from a checkpoint
# resume_from_checkpoint = "/path/to/model.ckpt"
```

## Usage

### Basic Training
```bash
# Train with default config (ffnn-normscaled)
uv run python graph-cg/scripts/train_model.py --data-config graph-cg/data-configs/collect-504.toml

# Train with specific config
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/linear.toml --data-config graph-cg/data-configs/collect-504.toml
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/ffnn.toml --data-config graph-cg/data-configs/generate-90-krylov50.toml
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/gnn.toml --data-config graph-cg/data-configs/collect-504.toml
```

### Prediction/Inference
```bash
# Predict with default config
uv run python graph-cg/scripts/predict.py --data-config graph-cg/data-configs/collect-504.toml

# Predict with specific config and checkpoint
uv run python graph-cg/scripts/predict.py --config graph-cg/configs/linear.toml --data-config graph-cg/data-configs/collect-504.toml --checkpoint /path/to/checkpoint.ckpt
```

## Better Approaches to Config Management

### Current Approach
- ✅ Simple and explicit
- ✅ Version controlled
- ✅ Easy to understand
- ❌ Duplicates common settings
- ❌ Requires multiple files for variations

### Recommended Improvements

#### 1. **Command-Line Overrides** (Simplest)
Keep one config per model type, override hyperparameters via CLI:

```bash
# Train linear model with different learning rates
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/linear.toml --data-config graph-cg/data-configs/collect-504.toml --lr 1e-3
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/linear.toml --data-config graph-cg/data-configs/collect-504.toml --lr 1e-4

# Train constant-width with different sizes
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/ffnn.toml --data-config graph-cg/data-configs/generate-90-krylov50.toml --hidden-size 128 --num-layers 5
```

**Implementation**: Add CLI arguments to `scripts/train_model.py` that override config values.

#### 2. **Config Inheritance** (Most Flexible)
Use a base config + specific overrides:

```toml
# configs/base.toml
[SESSION]
seed = 42
precision = "32"
root_dir = "/data/projects/graph-cg"

[TRAINING]
max_epochs = 1000
...

# configs/linear.toml
inherit = "base.toml"
[MODEL]
name = "LinearFFNN"
```

**Implementation**: Requires custom config loading logic to merge configs.

#### 3. **Environment Variables** (For Paths)
Use env vars for paths that change between environments:

```bash
export GRAPH_CG_DATA_DIR=/data/projects/graph-cg
export GRAPH_CG_OUTPUT_DIR=/data/projects/graph-cg/output
uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/linear.toml
```

**Implementation**: Update config loading to substitute env vars.

#### 4. **Session-Based Differentiation** (Current Best Practice)
Keep one config per model architecture, use session names to differentiate runs:

```toml
# Same config file, different session names for experiments
[SESSION]
name = "FFNN-NormScaled-504-lr1e3-exp1"  # Descriptive session name
```

- MLflow/tracking uses session name for run identification
- Checkpoint filenames can include session name
- No config duplication needed

## Recommended Approach for This Project

**Use current approach + CLI overrides + descriptive session names**

1. Keep one config per model architecture (current setup)
2. Add CLI overrides for common hyperparameters:
   ```python
   # In scripts/train_model.py
   from src.configuration.loader import load_experiment

   def main(
       config: Path,
       data_config: Path,
       lr: float | None = None,
       hidden_size: int | None = None,
       num_layers: int | None = None,
       batch_size: int | None = None,
       max_epochs: int | None = None,
   ):
       # Load with Pydantic validation
       experiment = load_experiment(config, data_config)

       # Override settings from CLI
       if lr is not None:
           experiment.settings.TRAINING.optimizer.lr = lr
       # ... apply other overrides
   ```

3. Use descriptive session names to track variations:
   ```bash
   # Experiment with different learning rates
   uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/linear.toml --lr 1e-3
   # Session name in config: "Linear-504-lr1e3"

   uv run python graph-cg/scripts/train_model.py --config graph-cg/configs/linear.toml --lr 1e-4
   # Session name in config: "Linear-504-lr1e4"
   ```

This gives you:
- ✅ Clean, minimal config files
- ✅ Easy hyperparameter sweeps via CLI
- ✅ Full tracking via MLflow session names
- ✅ No code duplication
- ✅ Simple to understand and maintain
