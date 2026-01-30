# Solver Implementations

This directory contains iterative solver implementations for sparse linear systems.

## Solver Hierarchy

```
IterativeSolverBase (base.py)
    └── KrylovSolverBase (krylov_base.py)
            └── ConjugateGradientBase (cg_base.py)
                    ├── PreconditionedCGSolver (pcg_solver.py)
                    └── FlexibleCGSolver (fcg_solver.py)

SciPyCGSolver (scipy_cg_solver.py) - Wrapper, not part of hierarchy
```

## Design Principles

- **Template Method**: Base classes define iteration skeleton, subclasses customize steps
- **Strategy Pattern**: Orthogonalization and convergence strategies are injected
- **Single Responsibility**: Each solver implements one algorithm variant
- **Composition over Inheritance**: Strategies injected, not inherited

---

## Solver Algorithms

### 1. PreconditionedCGSolver (Standard PCG)

Standard Preconditioned Conjugate Gradient with two-term recurrence.

**Algorithm:**

```
Initialize:
    u₀ = initial guess (or 0)
    r₀ = b - A u₀
    w₀ = M⁻¹ r₀
    d₀ = w₀

For i = 0, 1, 2, ... until convergence:
    1. Ad_i = A d_i                        # Matrix-vector product
    2. α_i = (r_i, w_i) / (d_i, Ad_i)      # Step length
    3. u_{i+1} = u_i + α_i d_i             # Update solution
    4. r_{i+1} = r_i - α_i Ad_i            # Update residual
    5. w_{i+1} = M⁻¹ r_{i+1}               # Apply preconditioner
    6. β_i = (r_{i+1}, w_{i+1}) / (r_i, w_i)  # Fletcher-Reeves formula
    7. d_{i+1} = w_{i+1} + β_i d_i         # Two-term recurrence
```

**Code Variable Mapping:**
- `u_i` ↔ `x` (solution iterate)
- `r_i` ↔ `r` (residual)
- `w_i` ↔ `z` (preconditioned residual)
- `d_i` ↔ `p` (search direction)
- `Ad_i` ↔ `q` (matrix-vector product)

**Properties:**
- Memory: O(n) - only stores current vectors
- Cost per iteration: 1 matvec + 1 preconditioner apply
- Requirements: A must be SPD, M must be SPD

**Beta Formulas:**
- Fletcher-Reeves: `β_i = (r_{i+1}, w_{i+1}) / (r_i, w_i)`
- Polak-Ribière: `β_i = (r_{i+1}, w_{i+1} - w_i) / (r_i, w_i)`

**Reference:** Hestenes & Stiefel (1952). Methods of Conjugate Gradients.

---

### 2. FlexibleCGSolver (FCG with Orthogonalization)

Flexible Conjugate Gradient for variable or non-SPD preconditioners. This implementation
supports FCG(m) with periodic restart as described in Notay (2000).

**Algorithm (FCG from Notay 2000, Algorithm 2.1):**

```
Initialize:
    u_0 = initial guess (or 0)
    r_0 = b - A u_0
    m_0 = 0

For i = 0, 1, 2, ... until convergence:
    1. w_i = B(r_i)                                        # Apply (variable) preconditioner
    2. Compute m_i = max(1, i mod (m_max + 1))             # Periodic restart formula
    3. d_i = w_i - Σ_{k=i-m_i}^{i-1} γ_{ik} d_k           # Orthogonalize
       where γ_{ik} = (w_i, A d_k) / (d_k, A d_k)
    4. α_i = (d_i, r_i) / (d_i, A d_i)                     # Step length
    5. u_{i+1} = u_i + α_i d_i                             # Update solution
    6. r_{i+1} = r_i - α_i A d_i                           # Update residual
```

**Orthogonalization Formula (A-conjugacy):**

```
d_i = w_i - Σ_{k=i-m_i}^{i-1} γ_{ik} d_k

where:
    γ_{ik} = (w_i, A d_k) / (d_k, A d_k)   # Orthogonalization coefficient
    m_i = max(1, i mod (m_max + 1))        # Periodic restart formula
```

This enforces approximate A-conjugacy: `d_i^T A d_k ≈ 0` for `k < i`.

**Code Variable Mapping:**
- `w_i` ↔ `z` (preconditioned residual)
- `d_i` ↔ `p` (search direction)
- `A d_k` ↔ `q` (matrix-vector product, stored for reuse)

**FCG(m) Notation:**

The notation FCG(m) or FCG(m_max) denotes Flexible CG with orthogonalization
window size m_max:
- **FCG(10)**: Orthogonalize against at most 10 previous directions
- **FCG(∞)**: Full reorthogonalization (orthogonalize against all history)
- **FCG(1)**: Minimal orthogonalization (only against most recent direction)

The periodic restart formula `m_i = max(1, i mod (m_max + 1))` creates a cyclic
pattern where the window resets every (m_max + 1) iterations, preventing error
accumulation while bounding memory.

**Properties:**
- Memory: O(m_max × n) - stores window of directions
- Cost per iteration: 1 matvec + 1 precond + O(m_max) dot products
- Requirements: A must be SPD, M can be variable/non-SPD

**Reference:**
- Notay, Y. (2000). Flexible Conjugate Gradients. *SIAM Journal on Scientific Computing*, 22(4), 1444-1460.
  doi:10.1137/S1064827599362314

---

### 3. SciPyCGSolver (scipy.sparse.linalg.cg Wrapper)

Wrapper around scipy's CG implementation with callback-based monitoring.

**Use Cases:**
- Comparison baseline against custom implementations
- Validation against reference scipy implementation
- When scipy's implementation is preferred

**Features:**
- Callback integration via `SciPyCallbackAdapter`
- History tracking via `HistoryTracker`
- Compatible `SolverResult` output format

---

## Orthogonalization Strategies

Three orthogonalization strategies are available for FlexibleCGSolver, implementing
different variants from the literature:

### 1. PeriodicRestartOrthogonalization (FCG with Periodic Restart)

**Location:** `orthogonalization.py`

**Algorithm:** FCG(mₘₐₓ) from Notay (2000) - Flexible Conjugate Gradient with periodic restart

Implements the paper's recommended cyclic restart formula:

```
m_i = max(1, i mod (m_max + 1))
```

**Example with m_max = 10:**

| Iteration i | m_i = max(1, i mod 11) | Behavior |
|-------------|------------------------|----------|
| 0           | max(1, 0) = 1          | 1 direction |
| 1-10        | 1-10                   | accumulate history |
| 11          | max(1, 0) = **1**      | **RESTART** - discard old directions |
| 12-21       | 2-11 → clamp to 10     | rebuild history |

**Properties:**
- Periodic reset prevents accumulation of rounding errors
- Generally more cost-efficient per Notay (2000) Table 1 benchmarks
- Memory: O(m_max × n)
- **This is the paper's original FCG algorithm**

**Reference:** Notay (2000) Section 2, Algorithm 2.1

### 2. FullOrthogonalization (FCG(∞))

**Location:** `orthogonalization.py`

**Algorithm:** FCG(∞) - Full reorthogonalization against all history

Implements FCG with infinite orthogonalization window. Mathematically equivalent
to setting m_max → ∞ in the periodic restart formula, which gives m_i = i.

**Properties:**
- Orthogonalizes against all previous directions
- Exact A-conjugacy (within numerical precision)
- Memory: O(k × n) where k = iteration count
- Used for ill-conditioned problems (e.g., Notay 2000 Table 1, Case 3: κ=1100)

**Reference:** Notay (2000) Section 5.1, Table 1 Case 3

### 3. TruncatedGramSchmidt (Tr-FCG with Sliding Window)

**Location:** `orthogonalization.py`

**Algorithm:** Tr-FCG - Truncated FCG with pure sliding window (no periodic restart)

Implements truncated Gram-Schmidt with fixed window size:

```
m_i = min(i, m_max)
```

| Iteration i | m_i = min(i, 10) | Behavior |
|-------------|------------------|----------|
| 0           | 0                | none     |
| 1-10        | 1-10             | accumulate history |
| 11+         | 10               | sliding window, keeps last m_max |

**Properties:**
- Simpler implementation (no restart logic)
- Always orthogonalizes against last m_max directions
- Memory: O(m_max × n)
- **This is NOT the original FCG from the paper** (no periodic restart)

**Note:** This is a variant sometimes called "Tr-FCG" in the literature.

### FlexibleCGSolver Default

`FlexibleCGSolver` defaults to `PeriodicRestartOrthogonalization(m_max=10)`,
which implements the original FCG algorithm from Notay (2000) with periodic restart.

### Core Algorithm (Both Strategies)

Both strategies use the same orthogonalization formula and update equations:

| Component | Formula (Notay 2000) |
|-----------|----------------------|
| Orthogonalization | `γ_{ik} = (w_i, A d_k) / (d_k, A d_k)` |
| Step length | `α_i = (d_i, r_i) / (d_i, A d_i)` |
| Solution update | `u_{i+1} = u_i + α_i d_i` |
| Residual update | `r_{i+1} = r_i - α_i A d_i` |

---

## Parameters

### FlexibleCGSolver

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preconditioner` | `Preconditioner` | Identity | M⁻¹ operator |
| `orthogonalization` | `OrthogonalizationStrategy` | PeriodicRestartOrthogonalization(10) | Orthogonalization method |
| `convergence_criterion` | `IConvergenceCriterion` | Combined(rtol, atol) | Stopping criterion |

### PeriodicRestartOrthogonalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m_max` | `int` | 10 | Maximum window size before periodic restart |
| `epsilon` | `float` | 1e-14 | Small denominator threshold |

### TruncatedGramSchmidt (Tr-FCG)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | 10 | mₘₐₓ - maximum sliding window size |
| `epsilon` | `float` | 1e-14 | Small denominator threshold |

### Solve Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rtol` | `float` | 1e-6 | Relative tolerance: ‖r‖/‖b‖ < rtol |
| `atol` | `float` | 1e-14 | Absolute tolerance: ‖r‖ < atol |
| `maxiter` | `int` | 100 | Maximum iterations |

---

## References

### Primary References

1. **Notay, Y. (2000)**. Flexible Conjugate Gradients. *SIAM Journal on Scientific Computing*, 22(4), 1444-1460.
   - **DOI:** [10.1137/S1064827599362314](https://doi.org/10.1137/S1064827599362314)
   - **Original FCG paper** - Defines Algorithm 2.1 (Section 2) with periodic restart formula
   - Analyzes truncation strategies and provides benchmark comparisons (Table 1)
   - Recommends periodic restart (m_i = max(1, i mod (m_max + 1))) as "generally more cost efficient"
   - Section 5.1 discusses FCG(∞) for ill-conditioned problems

2. **Hestenes, M. R., & Stiefel, E. (1952)**. Methods of Conjugate Gradients for Solving Linear Systems. *Journal of Research of the National Bureau of Standards*, 49(6), 409-436.
   - **DOI:** [10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)
   - Original conjugate gradient paper

### Comprehensive References

3. **Saad, Y. (2003)**. *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
   - **ISBN:** 978-0-89871-534-7
   - Chapter 9 covers CG methods and preconditioning
   - Section 9.5 discusses flexible variants

4. **Greenbaum, A. (1997)**. *Iterative Methods for Solving Linear Systems*. SIAM.
   - **ISBN:** 978-0-89871-396-1
   - Theoretical foundations of iterative methods
   - Analysis of A-conjugacy and orthogonalization

### Implementation Notes

- Our `FlexibleCGSolver` implements Notay (2000) Algorithm 2.1 with periodic restart
- Our `PeriodicRestartOrthogonalization` implements the formula from Section 2
- Our `FullOrthogonalization` implements FCG(∞) from Section 5.1
- Test cases in `tests/solver/conftest.py` reproduce Notay (2000) Table 1 conditions
