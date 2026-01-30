# Notay 2000 - Flexible Conjugate Gradients Benchmark

Benchmark tests based on:

> **Notay, Y. (2000). Flexible Conjugate Gradients.**
> SIAM Journal on Scientific Computing, 22(4), 1444-1460.

Implements **Section 5.1: Some artificial experiments** (Variant a only).

---

## Paper Parameters

| Parameter               | Value                          |
|-------------------------|--------------------------------|
| $n$                     | $10^4$                         |
| $\delta$ (tolerance)    | $10^{-6}$ (relative threshold) |
| $f$ random perturbation | Pseudorandom uniform $[-1, 1]$ |
| Initial guess           | Zero vector                    |
| Preconditioner $B$      | $I$ (identity)                 |

---

## Convergence Criterion

Probably regular relative residual:

$$\|r_i\| \leq \delta \|b\|$$

where:
- $r_i = b - Au_i$ the residual
- $u_i$ is the solution iterate at iteration $i$
- $\delta = 10^{-6}$ is the tolerance


---

## Section 5.1 Cases

| Case | Eigenvalue Formula | $\kappa$ | FCG variant |
|------|-------------------|---|-------------|
| 1 | $\lambda_i = 1 + \frac{5(i-1)}{n-1}$ | 6 | FCG(1) |
| 2 | $\lambda_i = 1 + \frac{50(i-1)}{n-1}$ | 51 | FCG(1) |
| 3 | $\lambda_1=0.01$, $\lambda_i=1+\frac{10(i-2)}{n-2}$ for $i>1$ | 1100 | FCG($\infty$) |

---

## Variant (a): Random Perturbation

Perturbed preconditioner application (Notay 2000, Section 5.1):

$$w_i = r_i + \varepsilon \frac{\|r_i\|}{\|f\|} f$$

where:
- $w_i$ is the preconditioned residual (in code: z)
- $r_i$ is the residual
- $f$ is a perturbation vector with components uniformly distributed in $[-1, 1]$
- $\varepsilon$ is the perturbation parameter

**Two implementations:**
- **Fixed (Paper):** $f$ generated once at initialization and reused
- **Dynamic (Alternative):** Fresh $f$ generated at each iteration

---

## Paper Table 1 Results (Variant a)

| $\varepsilon$ | 0 | $10^{-2}$ | $10^{-1}$ | $1/7$ | $1/4$ | $1/3$ | $1/2$ |
|---|---|-------|-------|-----|-----|-----|-----|
| Case 1 ($\kappa=6$) | 15 | 15 | 16 | 17 | 19 | 22 | 28 |
| Case 2 ($\kappa=51$) | 49 | 49 | 55 | 59 | 69 | 81 | 116 |
| Case 3 ($\kappa=1100$) | 31 | 31 | 32 | 33 | 37 | 40 | 49 |

---

## Tolerance Scaling

Since different RNG implementations produce different perturbation vectors $f$,
exact iteration counts cannot be matched. Tolerance scales with $\varepsilon$:

$$\text{atol} = \text{BASE_ATOL} + \text{EPSILON_SCALE} \cdot \varepsilon \cdot \text{expected_iterations}$$

At $\varepsilon=0$ (no perturbation), results should match closely.

---

## Running Tests

```bash
uv run pytest tests/benchmarks/notay2000/ -v
```

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Case 1 + Variant (a) | ✓ Done |
| Case 2 + Variant (a) | ✓ Done |
| Case 3 + Variant (a) | ✓ Done |
| Variant (b) - Inner PCG | Not implemented |