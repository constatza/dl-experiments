# Data Generation and Collection Configs

This directory contains configuration files for data generation and collection, separate from training configs.

## Directory Naming Convention

Generated and archived RHS data follows a consistent naming pattern:

### Collected Data (from SpectralData)
- Format: `collect-{dim}-{norm|nonorm}`
- Examples:
  - `collect-504-norm` - 504-dim system, normalized
  - `collect-2040-nonorm` - 2040-dim system, not normalized
- **No method tag** for RHS archive ingestion (no krylov generation)

### Generated Data (synthetic)
- Format: `generate-{dim}-{norm|nonorm}` or `generate-{dim}-krylov{percent}-{norm|nonorm}`
- Examples:
  - `generate-90-norm` - Pure normal generation (no krylov tag when 0%)
  - `generate-90-krylov50-norm` - 50% krylov, 50% normal
- `generate-280-krylov100-norm` - 100% krylov

**Key Rule**: Krylov tag is **only added when krylov percentage > 0**

### Strategy Declaration

Declare the data mix by repeating array-of-table entries `[[generation.strategy]]`.
Each entry sets a `name`, `percentage` (fractions must sum to 1.0), and any
strategy-specific options. Strategy options override generation-wide defaults
when present (for example, a `krylov` entry can include `krylov_iters = 20`).

```toml
[generation]
num_samples = 6000
normalize = "spectral"

[[generation.strategy]]
name = "normal"
percentage = 0.4

[[generation.strategy]]
name = "cg_residual"
percentage = 0.4
residual_iters = 12  # example strategy-specific option

[[generation.strategy]]
name = "rhs_archive"
percentage = 0.2
rhs_glob = "/data/projects/graph-cg/data/raw/rhs-*.txt"
```

Supported overrides (all optional):

- `normal`: no additional options
- `krylov`: `krylov_iters`, `seed`
- `cg_residual` / `residual`: `residual_iters`, `seed`
- `rhs_archive`: `rhs_glob`, `solve_systems`, `cg_tolerance`, `cg_max_iters`
- `solution_archive`: `solutions_glob`, `shuffle`, `seed`

Supported strategy identifiers:

- `normal`: dense Gaussian solutions (default synthetic generator)
- `krylov`: enriched Krylov subspace samples (adds `krylovXX` suffix when used)
- `cg_residual` / `residual`: capture CG residual traces; produces `cg-residuals.npy`, `cg-solutions.npy`, and `cg-trace-meta.npz`
- `rhs_archive`: reuse existing RHS files; requires `rhs_glob`
- `solution_archive`: ingest pre-computed solution vectors; requires `solutions_glob`

### Solution Archive Data (pre-computed solutions)
- Format: `solutions-{dim}-{norm|nonorm}` or similar
- These configs load a matrix and a directory of solution vectors (`x_i`) and
  transform them into `(A, b, x)` triples by computing `b_i = A x_i`.
- Declare the pipeline with `[[generation.strategy]]` (`name = "solution_archive"`) and provide
  `percentage = 1.0` plus a `solutions_glob` pointing at the solution files.

## Flow Coordination

Each config declares a `[flow]` block for flow namespace coordination.
The `dataset` field has been **removed** - dataset names are now automatically
derived from the config filename.

```toml
[flow]
# Optional flow ID (defaults to dataset name if not provided)
id = "spectral-baseline"
```

**Dataset Naming**: The dataset name is automatically derived from the config filename stem.
For example, `collect-504-solutions.toml` creates dataset `collect-504-solutions`.
Data is written to: `data/processed/collect-504-solutions/`

**Benefits**:
- Single source of truth (filename)
- No mismatches between config content and output directory
- Simpler config structure

## Test Configuration for Preconditioner Comparison

Data configs can optionally specify test data for preconditioner comparison via the `[test]` section:

```toml
[test]
# Method 1: Provide test solutions - RHS will be computed as b_test = A @ x_test
solutions_path = "/path/to/test-solutions/x_*.txt"

# Method 2: Provide explicit test RHS (and optionally matrix)
# rhs = "/path/to/test-rhs.npy"
# matrix = "/path/to/test-matrix.txt"  # defaults to source.matrix_path
```

### Test Solutions (Recommended)

When `test.solutions_path` is provided, the system:
1. Loads test solution vectors (`x_test`) from the glob pattern
2. Computes test RHS: `b_test = A @ x_test` using the first solution
3. Uses `b_test` for preconditioner evaluation with **known ground truth**

This allows precise measurement of how well each preconditioner recovers the known solution.

**Example**: Using displacement vectors as test solutions:
```toml
[test]
solutions_path = "/data/SpectralData/45x15-displacements/UaVectorsFromSpectral_0_5_b/sample_*.txt"
```

### Explicit Test RHS

Alternatively, you can directly specify a test RHS vector and optionally a test matrix.

**Path Resolution Priority**:
1. Explicit CLI parameters (highest priority)
2. `test.solutions_path`: computes `b_test = A @ x_test`
3. `test.rhs` / `test.matrix` from `[test]` section
4. Default paths from training data

**Use Cases**:
- Test preconditioners on different problem instances
- Use canonical test cases for reproducible benchmarks
- Separate training data from evaluation data
- Measure recovery accuracy against known ground truth

## Usage

### Collecting Data from SpectralData

```bash
# Collect 504-dimensional case
uv run python collect_data.py data-configs/collect-504.toml
# Output: /data/projects/graph-cg/data/processed/collect-504-norm/

# Collect 2040-dimensional case
uv run python collect_data.py data-configs/collect-2040.toml
# Output: /data/projects/graph-cg/data/processed/collect-2040-norm/
```

### Generating Synthetic Data

```bash
# Pure normal generation (no krylov)
uv run python generate_data.py data-configs/generate-90.toml
# Output: /data/projects/graph-cg/data/processed/generate-90-norm/

# Mixed generation (50% krylov)
  uv run python generate_data.py data-configs/generate-90-krylov50.toml
# Output: /data/projects/graph-cg/data/processed/generate-90-krylov50-norm/

### Ingesting a Solution Bank

```
uv run python generate_data.py data-configs/solution-bank-example.toml
# Output: /data/projects/graph-cg/data/processed/solution-bank-example/
```
```

## Output Structure

Each data directory contains:

```
collect-504-norm/
├── matrix.npy              # System matrix (original, unnormalized)
├── rhs-samples.npy         # Right-hand side samples (shape: N × 504)
├── sol-samples.npy         # Solutions (solved via CG)
├── rhs-mother.npy          # Reference/first RHS sample
├── cg-residuals.npy        # Optional residual traces (present when cg_residual strategy is used)
├── cg-solutions.npy        # Optional per-iteration solution targets
├── cg-trace-meta.npz       # Metadata (sample_idx, iteration_idx) for residual traces
├── normalization.json      # Normalization metadata
└── metadata.json           # General metadata
```

## Config File Structure

### Collection Config (collect-*.toml)

```toml
[flow]
# Dataset name automatically derived from filename (e.g., "collect-504" from "collect-504.toml")

[source]
matrix_path = "/path/to/SpectralData/45x15/stiffness/subdomain_1_Kaa.txt"
rhs_path = "/path/to/SpectralData/45x15/faVectorsFromSpectral/fa_*.txt"

[generation]
# Normalization method: "matrix" (spectral radius), "rhs" (per-sample), or "none"
normalize = "matrix"

[[generation.strategy]]
name = "rhs_archive"
percentage = 1.0
rhs_glob = "/path/to/SpectralData/45x15/faVectorsFromSpectral/fa_*.txt"

[output]
processed_dir = "/data/projects/graph-cg/data/processed"

[test]
# Optional: specify test data for preconditioner comparison
# Method 1 (recommended): Provide test solutions (computes b_test = A @ x_test)
# solutions_path = "/path/to/test-solutions/x_*.txt"
# Method 2: Provide explicit test RHS
# rhs = "/path/to/test-rhs.npy"
# matrix = "/path/to/test-matrix.txt"
```

### Generation Config (generate-*.toml)

```toml
[flow]
# Dataset name automatically derived from filename (e.g., "generate-90-krylov50" from "generate-90-krylov50.toml")

[source]
matrix_path = "/path/to/matrix.txt"
rhs_path = "/path/to/rhs.txt"

[generation]
num_samples = 5000
# Normalization method: "matrix" (spectral radius), "rhs" (per-sample), or "none"
normalize = "matrix"
krylov_iters = 15
residual_iters = 15
seed = 42
shuffle = true

[[generation.strategy]]
name = "normal"
percentage = 0.6

[[generation.strategy]]
name = "cg_residual"
percentage = 0.2
residual_iters = 15  # overrides global if needed

[[generation.strategy]]
name = "rhs_archive"
percentage = 0.2
rhs_glob = "/data/projects/graph-cg/data/raw/rhs-*.txt"

[output]
processed_dir = "/data/projects/graph-cg/data/processed"

[test]
# Optional: specify test data for preconditioner comparison
# Method 1 (recommended): Provide test solutions (computes b_test = A @ x_test)
# solutions_path = "/path/to/test-solutions/x_*.txt"
# Method 2: Provide explicit test RHS
# rhs = "/path/to/test-rhs.npy"
# matrix = "/path/to/test-matrix.txt"
```

## Using Generated Data for Training

After generating or collecting data, reuse the same `[flow]` block in your
training config. The centralized loader will resolve feature/target paths,
training directories, prediction exports, and comparison outputs without
manual path edits.

## Available Templates

- `collect-504.toml` - Collect from SpectralData 45×15 (504-dim)
- `collect-2040.toml` - Collect from SpectralData 93×31 (2040-dim)
- `generate-90.toml` - Generate 90-dim, pure normal (no krylov)
- `generate-90-krylov50.toml` - Generate 90-dim, 50% krylov
- `generate-280-krylov50.toml` - Generate 280-dim, 50% krylov
- `solution-bank-example.toml` - Ingest pre-computed solutions for 90-dim matrix
- `spectral-504-solutions.toml` - Displacement solutions paired with 45×15 matrix (504-dim)
- `spectral-2040-solutions.toml` - Displacement solutions paired with 93×31 matrix (2040-dim)

## Workflow

1. **Generate/collect data once:**
   ```bash
   uv run python collect_data.py data-configs/collect-504.toml
   ```

2. **Train multiple models on same data:**
   ```bash
   uv run python train.py config-ffnn-504.toml
   uv run python train.py config-gat-504.toml
   ```

3. **Switch datasets:** Point both data and training configs to a new
   `[flow]` block (no manual path editing required)
