# Dataset Config Guide

Dataset configs are the first layer most users should touch. They answer one
question: how do we turn a matrix plus optional archives into processed training
data?

## First Successful Run

```bash
uv run process-data configs/datasets/residuals-100.toml
```

Start with one dataset config before moving on to training or registry-wide
workflows.

## Minimal Shape

```toml
id = "residuals-100"

[source]
matrix_path = "/path/to/matrix.txt"

[generation]
normalize = "matrix"
seed = 42

[[generation.strategy]]
name = "residuals"
samples = 20000
cg_iters = 100

[output]
data_dir = "/path/to/processed"
```

## Strategy Ladder

Choose the simplest strategy that matches the learning target.

### Basic forward-pair strategies

Use these when you want standard `(b, x)` training pairs:

- `normal`
- `krylov`
- `eigenvector_forward`
- `eigenvector_inverse`
- `rhs_archive`
- `solution_archive`

### Trace strategies

Use these when the model should learn from CG internals rather than only final
solution pairs:

- `residual_traces`: records `(r_k, x_k)`
- `residuals`: records `(r_k, x_true - x_k)`
- `gaussian_residuals`: same target as `residuals`, but samples `x_true` from a
  Gaussian instead of reading archived solutions
- `search_directions`: records search-direction traces

For trace strategies, `samples` is a row budget, not a base-system count.

## Residual Strategies

`residual_traces` and `residuals` are similar but target different models:

- `residual_traces` is useful when the model should predict the current iterate
  from the current residual
- `residuals` is useful when the model should predict the correction
  `x_true - x_k`
- `gaussian_residuals` removes archive dependence when Gaussian true solutions
  are acceptable

## Important Fields

### Top-level `id`

This is the canonical dataset identity used by:

- processed dataset lookup
- workspace layout
- registry-driven training
- experiment-bound comparison resolution

Do not rely on the filename stem as the dataset identity.

### `[source]`

Usually includes:

- `matrix_path`
- optional archive paths used by strategy-specific logic

### `[generation]`

Controls shared behavior:

- `normalize`
- `seed`
- `shuffle`

### `[[generation.strategy]]`

Each block adds one source of samples to the final dataset. Mixed datasets are
built by combining multiple strategy blocks in order.

## Output And Test Metadata

`[output].data_dir` controls where processed datasets land.

Optional `[test]` fields can attach evaluation assets such as:

- `solutions_path`
- `rhs`
- `matrix`

These are used later by comparison and inference workflows.

## Next Steps

After the dataset builds:

1. train one model with `train-model`
2. move to an experiments registry
3. run `compare-all` once the trained checkpoint exists
