# Data Generation Strategies

This module implements a plugin-based strategy pattern for generating training data
for neural CG solvers and preconditioners. Each strategy encapsulates a specific
algorithm for producing training pairs from a system matrix **A**.

## Quick Reference

| Strategy | Requires b | Output | Description |
|----------|:----------:|--------|-------------|
| `random` | No | (b, x) | Random normal x, forward multiply |
| `krylov` | No | (b, x) | Lanczos basis sampling |
| `eigenvector_forward` | No | (b, x) | Eigenvector combinations, forward multiply |
| `eigenvector_inverse` | No | (b, x) | Eigenvector RHS, direct solve |
| `cg_residual` | Yes | (rₖ, xₖ) | CG iteration traces |
| `cg_residual_error` | Yes | (rₖ, eₖ) | CG error correction traces |
| `search_directions` | Yes | (Apₖ, pₖ) | CG search direction pairs |
| `rhs_archive` | No | (b, x) | Load b from disk, CG solve |
| `solution_archive` | No | (b, x) | Load x from disk, forward multiply |
| `neutral_ones` | No | (b, x) | x = **1**, deterministic baseline |

**Notation:**
- (b, x): RHS-solution pairs where b = Ax
- (rₖ, xₖ): Residual-solution pairs at CG iteration k
- (rₖ, eₖ): Residual-error pairs where eₖ = x* − xₖ
- (Apₖ, pₖ): Search direction product-direction pairs

---

## Conjugate Gradient Algorithm Reference

Strategies `cg_residual`, `cg_residual_error`, and `search_directions` collect
intermediate vectors from the standard CG algorithm. Given Ax = b with x₀ = 0:

**Initialization:**
```
r₀ = b − Ax₀ = b
p₀ = r₀
```

**Iteration k = 0, 1, 2, ..., K−1:**
```
αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)           # step length
xₖ₊₁ = xₖ + αₖpₖ                   # solution update
rₖ₊₁ = rₖ − αₖApₖ                  # residual update
βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)        # conjugacy coefficient
pₖ₊₁ = rₖ₊₁ + βₖpₖ                 # search direction update
```

**Key relationship:** rₖ = b − Axₖ = A(x* − xₖ) = Aeₖ

Implementation: `solver/solvers/scipy_cg_solver.py`

---

## Default Configuration Values

From `constants.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `shuffle` | True | Shuffle generated samples |
| `krylov_iters` | 15 | Lanczos iterations (Krylov subspace dimension) |
| `cg_iters` | 8 | CG iterations for trace strategies |
| `cg_tolerance` | 1e-15 | CG convergence tolerance (archive strategies) |
| `cg_max_iters` | 1,000,000 | Maximum CG iterations (archive strategies) |

---

## Strategy Details

### `random` / `normal`

**Implementation:** `strategies/random_normal.py` → `RandomNormalStrategy`

**Config:** `strategy_configs.py` → `RandomNormalConfig`

**Mathematical formulation:**

Given system matrix A ∈ ℝⁿˣⁿ and scale parameter σ:

1. Sample N solutions from normal distribution:
   ```
   xᵢ ~ N(0, σ²Iₙ),  i = 1, ..., N
   ```

2. Compute RHS via forward multiplication:
   ```
   bᵢ = Axᵢ
   ```

**Output:** {(bᵢ, xᵢ)}ᵢ₌₁ᴺ

**Parameters:**
- `samples`: N (number of pairs)
- `target_rhs_scale`: σ (standard deviation)

---

### `krylov`

**Implementation:** `strategies/krylov.py` → `KrylovStrategy`

**Config:** `strategy_configs.py` → `KrylovConfig`

**Mathematical formulation:**

Generates solutions aligned with A's spectral structure via Lanczos decomposition.

1. **Build Krylov basis** using Lanczos iteration with random start v₀:
   ```
   v₀ ~ N(0, I), v₀ ← v₀/‖v₀‖

   For j = 0, ..., m−1:
       w = Avⱼ − βⱼvⱼ₋₁
       αⱼ = vⱼᵀw
       w ← w − αⱼvⱼ
       βⱼ₊₁ = ‖w‖
       vⱼ₊₁ = w / βⱼ₊₁
   ```

   Result: orthonormal basis V = [v₀, v₁, ..., vₘ₋₁] ∈ ℝⁿˣᵐ

2. **Form tridiagonal matrix:**
   ```
   T = VᵀAV = tridiag(β₁, α₀, β₁, α₁, ..., βₘ₋₁, αₘ₋₁, βₘ)
   ```

3. **Eigendecompose:** T = QΛQᵀ where Λ = diag(λ₁, ..., λₘ)

4. **Generate samples** for i = 1, ..., N:
   ```
   εᵢ ~ N(0, Iₘ)                    # random in Krylov space
   xᵢ = V(QΛ⁻¹εᵢ)                   # project to full space
   bᵢ = Axᵢ                         # forward multiply
   ```

**Output:** {(bᵢ, xᵢ)}ᵢ₌₁ᴺ

**Parameters:**
- `samples`: N
- `krylov_iters`: m (Krylov subspace dimension)

**Note:** Early termination if βⱼ₊₁ < 10⁻¹⁴ (invariant subspace found).

---

### `eigenvector_forward`

**Implementation:** `strategies/eigenvector.py` → `EigenvectorForwardStrategy`

**Config:** `strategy_configs.py` → `EigenvectorForwardConfig`

**Mathematical formulation:**

1. **Compute eigendecomposition:** A = VΛVᵀ where V = [v₁, ..., vₙ]

2. **Select k eigenvectors** based on `which` parameter:
   - `"smallest"`: eigenvectors corresponding to k smallest eigenvalues
   - `"largest"`: eigenvectors corresponding to k largest eigenvalues
   - `"random"`: k randomly selected eigenvectors

3. **Generate solutions** as linear combinations:
   ```
   xᵢ = Σⱼ₌₁ᵏ cᵢⱼvⱼ,  where cᵢⱼ ~ N(0, 1)
   ```

   If `include_eigenvectors=True`, first k samples are pure eigenvectors:
   ```
   x₁ = v₁, x₂ = v₂, ..., xₖ = vₖ
   ```

4. **Forward multiply:** bᵢ = Axᵢ

**Output:** {(bᵢ, xᵢ)}ᵢ₌₁ᴺ

**Parameters:**
- `samples`: N
- `which`: eigenvalue selection mode
- `num_eigenvectors`: k
- `include_eigenvectors`: whether to include pure eigenvectors

---

### `eigenvector_inverse`

**Implementation:** `strategies/eigenvector.py` → `EigenvectorInverseStrategy`

**Config:** `strategy_configs.py` → `EigenvectorInverseConfig`

**Mathematical formulation:**

Same eigenvector selection as `eigenvector_forward`, but:

1. Use eigenvector combinations as **RHS**:
   ```
   bᵢ = Σⱼ₌₁ᵏ cᵢⱼvⱼ
   ```

2. **Solve** for solutions using Cholesky factorization:
   ```
   xᵢ = A⁻¹bᵢ   (via scipy.linalg.solve with assume_a="pos")
   ```

**Output:** {(bᵢ, xᵢ)}ᵢ₌₁ᴺ

**Parameters:** Same as `eigenvector_forward`

---

### `cg_residual` / `residual`

**Implementation:** `strategies/residual_traces.py` → `ResidualTraceStrategy`

**Config:** `strategy_configs.py` → `ResidualTraceConfig`

**Mathematical formulation:**

Collects (residual, solution) pairs from CG iterations.

1. **Generate N base systems:**
   - If archive provided: load xᵢ* from archive, compute bᵢ = Axᵢ*
   - Otherwise: sample xᵢ* ~ N(0, I), compute bᵢ = Axᵢ*

2. **For each system i**, run CG for K iterations starting from x₀ = 0:
   ```
   Collect: (r₀, x₀), (r₁, x₁), ..., (rₖ₋₁, xₖ₋₁)
   ```

   where rₖ = bᵢ − Axₖ (see CG algorithm above)

**Output:** `ResidualTraceSamples` containing:
- `residuals`: all rₖ vectors, shape (N·K, n)
- `solutions`: all xₖ vectors, shape (N·K, n)
- `sample_indices`: which base system each trace belongs to
- `iteration_indices`: which CG iteration

**Parameters:**
- `samples`: N (number of base systems)
- `cg_iters`: K (iterations per system)

**Training mapping:** NN(rₖ) → xₖ

---

### `cg_residual_error` / `residual_error`

**Implementation:** `strategies/residual_error.py` → `ResidualErrorStrategy`

**Config:** `strategy_configs.py` → `ResidualErrorConfig`

**Mathematical formulation:**

Collects (residual, error) pairs from CG iterations for error correction training.

1. **Generate N base systems** with known true solutions xᵢ*:
   - Preferably from solution archive (warning issued if random)
   - Compute bᵢ = Axᵢ*

2. **For each system i**, run CG for K iterations:
   ```
   Collect: (r₀, e₀), (r₁, e₁), ..., (rₖ₋₁, eₖ₋₁)

   where:
       rₖ = bᵢ − Axₖ           (residual)
       eₖ = xᵢ* − xₖ           (error to true solution)
   ```

**Key mathematical relationship:**
```
rₖ = A(xᵢ* − xₖ) = Aeₖ
```

This means (rₖ, eₖ) pairs have the same structure as (Ax, x) pairs,
making them valid training data for learning A⁻¹.

**Output:** `ErrorTraceSamples` containing:
- `residuals`: all rₖ vectors (network inputs)
- `errors`: all eₖ = x* − xₖ vectors (network targets)
- `solutions_current`: all xₖ vectors (for reference)
- `true_solutions`: x* per base system

**Parameters:**
- `samples`: N (number of base systems)
- `cg_iters`: K (iterations per system)

**Training mapping:** NN(rₖ) → eₖ, learning NN ≈ A⁻¹

---

### `search_directions`

**Implementation:** `strategies/search_directions.py` → `SearchDirectionsStrategy`

**Config:** `strategy_configs.py` → `SearchDirectionsConfig`

**Mathematical formulation:**

Collects (Apₖ, pₖ) pairs from CG **without requiring true solutions**.

1. **Generate N base systems** (random or from archive)

2. **For each system**, run CG for K iterations collecting search directions:
   ```
   Collect: (Ap₀, p₀), (Ap₁, p₁), ..., (Apₖ₋₁, pₖ₋₁)
   ```

   where pₖ is the search direction at iteration k (see CG algorithm)

**Rationale:** The mapping (Apₖ) → pₖ directly learns A⁻¹ since:
```
NN(Apₖ) ≈ pₖ  ⟹  NN ≈ A⁻¹
```

Unlike residual/error strategies, this doesn't require known solutions.

**Output:** `SearchDirectionsSamples` containing:
- `search_direction_products`: Apₖ vectors (network inputs)
- `search_directions`: pₖ vectors (network targets)
- `sample_indices`, `iteration_indices`: trace indexing

**Parameters:**
- `samples`: N (number of base systems)
- `cg_iters`: K (iterations per system)

**Training mapping:** NN(Apₖ) → pₖ, learning NN ≈ A⁻¹

---

### `rhs_archive`

**Implementation:** `strategies/rhs_archive.py` → `RhsArchiveStrategy`

**Config:** `strategy_configs.py` → `RhsArchiveConfig`

**Mathematical formulation:**

1. **Load RHS vectors** from files matching glob pattern:
   ```
   b₁, b₂, ..., bₙ ← files matching rhs_glob
   ```

2. **Solve each system** using scipy CG:
   ```
   xᵢ = CG(A, bᵢ, tol=ε, maxiter=M)
   ```

**Output:** {(bᵢ, xᵢ)}ᵢ₌₁ᴺ

**Parameters:**
- `samples`: N files to load (−1 for all)
- `rhs_glob`: file pattern (e.g., `/data/rhs_*.txt`)
- `cg_tolerance`: ε
- `cg_max_iters`: M

---

### `solution_archive`

**Implementation:** `strategies/solution_archive.py` → `SolutionArchiveStrategy`

**Config:** `strategy_configs.py` → `SolutionArchiveConfig`

**Mathematical formulation:**

1. **Load solution vectors** from files:
   ```
   x₁, x₂, ..., xₙ ← files matching solutions_glob
   ```

2. **Forward multiply** to obtain RHS:
   ```
   bᵢ = Axᵢ
   ```

**Output:** {(bᵢ, xᵢ)}ᵢ₌₁ᴺ

**Parameters:**
- `samples`: N files to load (−1 for all)
- `solutions_glob`: file pattern

---

### `neutral_ones`

**Implementation:** `strategies/neutral_ones.py` → `NeutralOnesStrategy`

**Config:** `strategies/neutral_ones.py` → `NeutralOnesConfig`

**Mathematical formulation:**

Deterministic baseline with solution vector of all ones:

```
x = [1, 1, ..., 1]ᵀ = 𝟙 ∈ ℝⁿ
b = Ax = A𝟙
```

**Output:** {(b, 𝟙)}

**Parameters:**
- `samples`: N identical copies

**Use:** Unbiased baseline for solver comparison across experiments.

---

## Output Data Structures

Defined in `interfaces.py` and `normalization.py`.

### GeneratedSamples

Main container returned by all strategies (`interfaces.py`):

| Field | Type | Description |
|-------|------|-------------|
| `matrix` | (n, n) | System matrix A |
| `rhs` | (N, n) | RHS vectors bᵢ |
| `solutions` | (N, n) | Solution vectors xᵢ |
| `residual_traces` | ResidualTraceSamples | From `cg_residual` |
| `error_traces` | ErrorTraceSamples | From `cg_residual_error` |
| `search_directions_traces` | SearchDirectionsSamples | From `search_directions` |

### Trace Sample Types

From `normalization.py`:

| Type | Fields | Shape | Description |
|------|--------|-------|-------------|
| `ResidualTraceSamples` | residuals, solutions | (M, n) | (rₖ, xₖ) pairs |
| `ErrorTraceSamples` | residuals, errors | (M, n) | (rₖ, eₖ) pairs |
| `SearchDirectionsSamples` | search_direction_products, search_directions | (M, n) | (Apₖ, pₖ) pairs |

All trace types include `sample_indices` and `iteration_indices` for indexing.

---

## Mixing Strategies

The `generate_mixture()` function in `orchestration.py` combines multiple strategies.

### Explicit Counts

```python
counts={"random": 500, "krylov": 300}
```

### Proportional Mixing

```python
mix={"random": 1.0, "krylov": 1.0}, total=1000  # 500 each
```

### Trace Strategy Counts

For CG-based strategies, `counts_represent_final_pairs=True` specifies output pairs
rather than base systems. The orchestrator computes:

```
num_base_systems = ⌈desired_pairs / cg_iters⌉
```

### Strategy Overrides

Per-strategy configuration via `strategy_overrides` dict:

```python
strategy_overrides={"krylov": {"krylov_iters": 25}}
```

See `orchestration.py:generate_mixture()` for full API.

---

## Architecture

```
Config (TOML)
     │
     ▼
orchestration.py:generate_mixture()
     │
     ├─── runner.py:_registry (strategy lookup)
     │
     ▼
For each strategy:
     │
     ├─── Build config from strategy_configs.py
     ├─── Call strategy.generate()
     └─── Collect GeneratedSamples
     │
     ▼
Merge + shuffle
     │
     ▼
Final Dataset (.npz)
```

## Adding New Strategies

1. Create `strategies/my_strategy.py` with:
   - Config class extending `BaseStrategyConfig`
   - Strategy class with `@register_strategy` decorator
   - Implement `requires_rhs()` and `generate()` methods

2. Add import to `strategies/__init__.py`

See existing strategies for implementation patterns.
