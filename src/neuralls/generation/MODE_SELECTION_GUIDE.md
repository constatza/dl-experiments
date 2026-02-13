# Data Generation Mode Selection Guide

## Overview

Data generation strategies operate in two fundamental modes:

1. **Forward Mode** (Cheap): Generate solutions x → Compute RHS b = A @ x
2. **Inverse Mode** (Expensive): Generate/load RHS b → Solve for solutions x = A^-1 @ b

This guide helps you choose the right mode and strategy for your use case.

---

## Quick Decision Tree

```
Do you have pre-computed solutions (x)?
├─ YES → Use FORWARD mode (compute b = A @ x)
│         Examples: solution_archive, eigenvector_forward, random_normal
│
└─ NO → Do you have RHS vectors (b)?
         ├─ YES → Use INVERSE mode (solve x = A^-1 @ b)
         │         Examples: rhs_archive, eigenvector_inverse
         │
         └─ NO → Can you generate solutions easily?
                  ├─ YES → Use FORWARD mode (generate x, compute b)
                  │         Recommended: random_normal, eigenvector_forward
                  │
                  └─ NO → Generate RHS and solve (EXPENSIVE!)
                           Use: eigenvector_inverse with caution
```

---

## Mode Comparison

| Aspect | Forward Mode | Inverse Mode |
|--------|--------------|--------------|
| **Operation** | b = A @ x | x = A^-1 @ b |
| **Complexity** | O(n²) per sample | O(n³) or O(n² × iters) |
| **Speed** | Fast (matrix-vector multiply) | Slow (solve linear system) |
| **Accuracy** | Exact (modulo fp64) | Approximate (CG) or exact (direct) |
| **Use when** | Have solutions, need RHS | Have RHS, need solutions |
| **Cost** | ~0.1s for n=1000 | ~1-10s for n=1000 |

---

## Strategy-by-Strategy Guide

### Forward Mode Strategies

#### 1. **random_normal** (Recommended for most cases)
- **What it does**: Generates random solutions x ~ N(0, σ²), computes b = A @ x
- **When to use**:
  - Training neural preconditioners from scratch
  - Need diverse solution/RHS pairs
  - No specific structure required
- **Performance**: Very fast (O(n²) per sample)
- **Example**:
  ```toml
  [generation.strategies.random_normal]
  samples = 1000
  target_rhs_scale = 1.0
  seed = 42
  ```

#### 2. **eigenvector_forward**
- **What it does**: Generates solutions from eigenvector combinations, computes b = A @ x
- **When to use**:
  - Want solutions spanning specific eigenspaces
  - Training preconditioners for particular spectral regions
  - Testing solver behavior on eigenvector-based inputs
- **Performance**: Fast (O(n²) per sample, after O(n³) eigendecomposition)
- **Example**:
  ```toml
  [generation.strategies.eigenvector_forward]
  samples = 500
  which = "smallest"  # or "largest", "random"
  num_eigenvectors = 10
  include_eigenvectors = true
  ```

#### 3. **solution_archive**
- **What it does**: Loads solutions x from disk, computes b = A @ x
- **When to use**:
  - Have pre-computed solutions from previous runs
  - Want to reuse solutions with different matrices
  - Testing with specific solution patterns
- **Performance**: Fast (O(n²) per sample + I/O)
- **Example**:
  ```toml
  [generation.strategies.solution_archive]
  samples = 100
  solutions_glob = "/data/solutions_*.txt"
  shuffle = true
  seed = 42
  ```

---

### Inverse Mode Strategies

#### 4. **rhs_archive**
- **What it does**: Loads RHS b from disk, solves x = A^-1 @ b via CG
- **When to use**:
  - Have pre-computed RHS vectors
  - Need exact solutions for specific RHS
  - Validating solver accuracy
- **Performance**: Slow (O(n² × iters) per sample + I/O)
- **Configuration**:
  - `cg_tolerance`: Default 1e-12 (relative)
  - `cg_max_iters`: Default 500
- **Example**:
  ```toml
  [generation.strategies.rhs_archive]
  samples = 50
  rhs_glob = "/data/rhs_*.txt"
  cg_tolerance = 1e-12
  cg_max_iters = 500
  ```

#### 5. **eigenvector_inverse**
- **What it does**: Generates RHS from eigenvector combinations, solves x = A^-1 @ b
- **When to use**:
  - Need solutions for eigenvector-based RHS
  - Testing solver on specific spectral components
  - **Rarely recommended** (forward mode is usually better)
- **Performance**: Very slow (O(n³) per sample with direct solve)
- **Configuration**:
  ```toml
  [generation.strategies.eigenvector_inverse]
  samples = 100
  which = "smallest"

  [generation.strategies.eigenvector_inverse.solve_config]
  method = "direct"  # or "cg"
  rtol = 1e-12
  atol = 0.0
  max_iters = 500
  assume_pos_def = true
  ```

---

### Archive Strategies

#### 6. **validated_archive** (New!)
- **What it does**: Loads both x and b from disk, verifies A @ x = b
- **When to use**:
  - Have pre-computed (x, b) pairs
  - Need to validate archived data quality
  - Want to skip redundant computation
- **Performance**: Fast (O(n²) per sample for verification + I/O)
- **Benefits**:
  - Avoids wasted computation (no solving/computing needed)
  - Quality checks archived data
  - Catches inconsistencies early
- **Example**:
  ```toml
  [generation.strategies.validated_archive]
  samples = 1000
  solutions_glob = "/data/solutions_*.txt"
  rhs_glob = "/data/rhs_*.txt"
  verification_tolerance = 1e-10
  fail_on_invalid = true  # Raise error on invalid pairs
  ```

---

## Performance Guidelines

### Cost Estimates (n = 1000)

| Strategy | Time per Sample | 1000 Samples |
|----------|----------------|--------------|
| **random_normal** | ~0.1 ms | ~0.1 seconds |
| **eigenvector_forward** | ~0.1 ms* | ~0.1 seconds* |
| **solution_archive** | ~0.1 ms + I/O | ~0.1s + I/O |
| **validated_archive** | ~0.1 ms + I/O | ~0.1s + I/O |
| **rhs_archive (CG)** | ~50 ms | ~50 seconds |
| **eigenvector_inverse (direct)** | ~500 ms | ~500 seconds |

*After initial O(n³) eigendecomposition

### Scaling with Problem Size

- **Forward mode**: Scales as O(n²) → feasible for n up to ~10,000
- **Inverse mode (CG)**: Scales as O(n² × iters) → feasible for n up to ~5,000
- **Inverse mode (direct)**: Scales as O(n³) → only feasible for n < 2,000

---

## Common Patterns

### Pattern 1: Training Neural Preconditioner
**Recommendation**: Use `random_normal` (forward mode)
```toml
[generation.strategies.random_normal]
samples = 10000
target_rhs_scale = 1.0
seed = 42
```

### Pattern 2: Validating Solver Accuracy
**Recommendation**: Use `rhs_archive` (inverse mode) or `validated_archive`
```toml
# Option A: Generate solutions by solving (expensive)
[generation.strategies.rhs_archive]
samples = 100
rhs_glob = "/data/test_rhs_*.txt"
cg_tolerance = 1e-15  # Very tight for validation

# Option B: Load pre-computed pairs and verify (cheap)
[generation.strategies.validated_archive]
samples = 100
solutions_glob = "/data/test_solutions_*.txt"
rhs_glob = "/data/test_rhs_*.txt"
verification_tolerance = 1e-10
```

### Pattern 3: Testing Spectral Properties
**Recommendation**: Use `eigenvector_forward` (forward mode)
```toml
[generation.strategies.eigenvector_forward]
samples = 500
which = "smallest"  # Test low-frequency modes
num_eigenvectors = 20
include_eigenvectors = true
```

### Pattern 4: Mixed Generation (Diverse Data)
**Recommendation**: Mix `random_normal` + `eigenvector_forward`
```toml
[generation]
total = 1000

[generation.strategies.random_normal]
samples = 700  # 70% random

[generation.strategies.eigenvector_forward]
samples = 300  # 30% eigenvector-based
which = "random"
```

---

## Configuration Trade-offs

### Solve Method Selection (Inverse Mode)

#### Direct Solve (`method = "direct"`)
- **Pros**: Exact solution (modulo fp64 rounding)
- **Cons**: O(n³) cost, only feasible for n < 2,000
- **Use when**: Need exact solutions, small problems

#### CG Solve (`method = "cg"`)
- **Pros**: O(n² × iters) cost, scales to larger problems
- **Cons**: Approximate solution, depends on tolerance/iters
- **Use when**: Large problems (n > 2,000), can tolerate approximation

### Tolerance Settings

| Use Case | rtol | atol | Rationale |
|----------|------|------|-----------|
| **Training data** | 1e-8 | 0.0 | Moderate accuracy, fast |
| **Validation** | 1e-12 | 0.0 | High accuracy |
| **Testing** | 1e-15 | 0.0 | Very high accuracy |

---

## Anti-Patterns (What NOT to Do)

### ❌ Anti-Pattern 1: Using Inverse Mode When Forward Works
**Bad**:
```toml
# Generate random RHS, then solve (EXPENSIVE!)
[generation.strategies.eigenvector_inverse]
samples = 10000  # Will take hours!
```

**Good**:
```toml
# Generate random solutions, compute RHS (FAST!)
[generation.strategies.random_normal]
samples = 10000  # Will take seconds!
```

### ❌ Anti-Pattern 2: Recomputing When Archives Exist
**Bad**:
```toml
# Load solutions, recompute RHS (wasted work!)
[generation.strategies.solution_archive]
samples = 1000
solutions_glob = "/data/solutions_*.txt"
# RHS already exists at /data/rhs_*.txt but not used!
```

**Good**:
```toml
# Load both, verify consistency (efficient!)
[generation.strategies.validated_archive]
samples = 1000
solutions_glob = "/data/solutions_*.txt"
rhs_glob = "/data/rhs_*.txt"
```

### ❌ Anti-Pattern 3: Mixed Inverse and Forward Without Reason
**Bad**:
```toml
# Mixing expensive (inverse) and cheap (forward) strategies
[generation.strategies.rhs_archive]
samples = 500  # Slow!

[generation.strategies.random_normal]
samples = 500  # Fast!
# Total time dominated by rhs_archive
```

**Good**:
```toml
# Use all forward strategies for consistent speed
[generation.strategies.random_normal]
samples = 700

[generation.strategies.eigenvector_forward]
samples = 300
```

---

## FAQ

### Q: Should I always use forward mode?
**A**: Almost always YES for training data. Use inverse mode only when:
- You have specific RHS vectors you need to solve
- You're validating solver accuracy against known solutions
- You have a specific reason to start with RHS

### Q: Why does `eigenvector_inverse` exist if forward is better?
**A**: Historical reasons + specific testing scenarios. Forward mode is preferred in 99% of cases.

### Q: Can I mix forward and inverse strategies?
**A**: Yes, but be aware of performance asymmetry. One slow inverse strategy can dominate total time.

### Q: How do I choose `num_eigenvectors` for eigenvector strategies?
**A**:
- Small (1-10): Focus on specific modes (e.g., smallest eigenvalues)
- Medium (10-50): Balanced spectral coverage
- Large (50+): Wide spectral diversity
- Use `which="random"` for uniform coverage

### Q: What's the difference between `cg_tolerance` (rhs_archive) and `verification_tolerance` (validated_archive)?
**A**:
- `cg_tolerance`: Controls CG solver convergence (how accurately to solve)
- `verification_tolerance`: Checks consistency of archived pairs (how much error to tolerate)

---

## Summary

**Default recommendation**: Use `random_normal` (forward mode) for training data.

**Special cases**:
- Have solutions archived → `solution_archive` (forward)
- Have both x and b archived → `validated_archive` (verify)
- Need specific RHS solved → `rhs_archive` (inverse, CG)
- Testing spectral properties → `eigenvector_forward` (forward)

**Avoid**:
- `eigenvector_inverse` unless you have a specific reason
- Mixing inverse and forward without performance consideration
- Solving when you can compute (always prefer forward mode!)

---

## Further Reading

- `src/neuralls/generation/strategies/` - Strategy implementations
- `src/neuralls/generation/strategy_configs.py` - Configuration models
- `tests/generation/` - Strategy usage examples
