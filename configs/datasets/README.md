# Data Generation and Collection Configs

This directory contains configuration files for data generation and collection. These configurations are handled by the local project's data generation module (`src/generation`), **not** by `dlkit`.

## Directory Naming Convention

Generated and archived RHS data follows a consistent naming pattern:

### Collected Data (from SpectralData)
- Format: `collect-{dim}-{norm|nonorm}`
- Examples:
  - `collect-504-norm` - 504-dim system, normalized
  - `collect-2040-nonorm` - 2040-dim system, not normalized

### Generated Data (synthetic)
- Format: `generate-{dim}-{norm|nonorm}` or `generate-{dim}-krylov{percent}-{norm|nonorm}`
- Examples:
  - `generate-90-norm` - Pure normal generation
  - `generate-90-krylov50-norm` - 50% krylov, 50% normal

## Configuration Structure

Data processing is configured via TOML files with the following sections:

### [flow]
Optional flow identification.
- **id**: `str`. Flow identifier. Defaults to the dataset name if not provided.

### [source]
Defines input data paths.
- **matrix_path**: `str`. Path to the system matrix file. **Required**.
- **rhs_path**: `str`. Path to RHS file (optional, used by some strategies).

### [generation]
Global generation settings.
- **normalize**: `str`. Normalization method. Options: `"matrix"` (spectral radius), `"spectral"` (spectral norm), `"rhs"` (per-sample), `"diagonal"` (Jacobi), `"none"`. Default: `"matrix"`.
- **seed**: `int`. Global random seed. Default: `42`.
- **shuffle**: `bool`. Whether to shuffle the final dataset. Default: `True`.

### [[generation.strategy]]
Defines a specific data generation strategy. You can have multiple blocks to mix different data sources.

**Common Fields:**
- **name**: `str`. Strategy name (e.g., `"normal"`, `"krylov"`, `"rhs_archive"`). **Required**.
- **samples**: `int`. Number of samples to generate. **Required**.
    - `> 0`: Exact number of samples.
    - `-1`: Use all available samples (for archives).
    - `0`: Skip this strategy.
- **seed**: `int`. Strategy-specific random seed. Default: `42`.
- **shuffle**: `bool`. Strategy-specific shuffle. Default: `True`.

**Strategies and Options:**

#### `normal` (Random Normal)
Generates dense Gaussian solution vectors and computes RHS as $b = Ax$.
- **target_rhs_scale**: `float`. Target scale for generated RHS vectors (Euclidean norm). Default: `1.0`.

#### `krylov` (Krylov Subspace)
Generates samples enriched with Krylov subspace components.
- **krylov_iters**: `int`. Number of Krylov iterations. Default: `5`.

#### `rhs_archive` (Existing RHS)
Uses existing RHS files.
- **rhs_glob**: `str`. Glob pattern for RHS files. **Required**.
- **solve_systems**: `bool`. Whether to solve $Ax=b$ to get solutions. Default: `True`.
- **cg_tolerance**: `float`. CG convergence tolerance. Default: `1e-10`.
- **cg_max_iters**: `int`. Maximum CG iterations. Default: `1000`.

#### `solution_archive` (Existing Solutions)
Uses existing solution files and computes RHS as $b = Ax$.
- **solutions_glob**: `str`. Glob pattern for solution files. **Required**.

#### `eigenvector_forward`
Uses eigenvectors as solutions ($x = v_i$), computes RHS as $b = \lambda_i v_i$.
- **which**: `str`. Eigenvalue selection. Options: `"smallest"`, `"largest"`, `"random"`. Default: `"smallest"`.
- **include_eigenvectors**: `bool`. Whether to include eigenvectors in solutions. Default: `True`.
- **num_eigenvectors**: `int`. Number of eigenvectors to include. Default: `1`.

#### `eigenvector_inverse`
Uses eigenvectors as RHS ($b = v_i$), solves for $x = A^{-1}v_i$.
- **which**: `str`. Eigenvalue selection. Options: `"smallest"`, `"largest"`, `"random"`. Default: `"smallest"`.
- **include_eigenvectors**: `bool`. Whether to include eigenvectors in solutions. Default: `True`.
- **num_eigenvectors**: `int`. Number of eigenvectors to include. Default: `1`.

#### `cg_residual_error`
Generates data based on CG residual errors.
- **residual_iters**: `int`. Number of residual iterations. Default: `15`.
- **archive_solutions**: `bool`. Archive intermediate solutions. Default: `False`.
- **archive_rhs**: `bool`. Archive intermediate RHS. Default: `False`.

### [output]
Output configuration.
- **processed_dir**: `str`. Directory where processed datasets will be saved. Optional.
- **output_root**: `str`. Root directory for all outputs (guides where results are saved). **Required**.

### [test]
Optional configuration for generating a comparison set.
- **solutions_path**: `str`. Path/glob to test solution vectors.
- **rhs**: `str`. Path to specific test RHS file.
- **matrix**: `str`. Path to specific test matrix file.

## Usage

Run the `process_data.py` script with your config:

```bash
uv run python graph-cg/scripts/process_data.py configs/datasets/collect-504-solutions.toml
```