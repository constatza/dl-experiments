# Perturbation Analysis for Preconditioned Conjugate Gradient Methods

## Overview

This document provides a comprehensive mathematical description of the perturbation methods used in robustness analysis of preconditioned conjugate gradient solvers. Each method is designed to test different aspects of solver robustness under specific error conditions that arise in real-world applications.

## Mathematical Foundation

Consider the linear system:
$$\mathbf{A}\mathbf{x} = \mathbf{b}$$

Where:
- $\mathbf{A} \in \mathbb{R}^{n \times n}$ is a symmetric positive definite matrix
- $\mathbf{b} \in \mathbb{R}^n$ is the right-hand side vector
- $\mathbf{x} \in \mathbb{R}^n$ is the unknown solution vector

The condition number $\kappa(\mathbf{A}) = \frac{\lambda_{\max}(\mathbf{A})}{\lambda_{\min}(\mathbf{A})}$ characterizes the sensitivity of the system to perturbations.

### Fundamental Perturbation Bound

For a perturbed system $\mathbf{A}\tilde{\mathbf{x}} = \tilde{\mathbf{b}}$, the classical perturbation theory gives:

$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \frac{\|\tilde{\mathbf{b}} - \mathbf{b}\|}{\|\mathbf{b}\|}$$

This bound motivates our algorithm-specific scaling strategies.

---

## Perturbation Methods

### 1. Global Gaussian Noise

**Mathematical Formulation:**
$$\tilde{\mathbf{b}} = \mathbf{b} + \bf{\epsilon}$$

Where $\bf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ with $\sigma = \rho \cdot \|\mathbf{b}\|_2$.

**Simple Explanation:** Adds random noise to every component of the RHS vector. Like adding measurement error to every sensor reading.

**Parameter Space:**
- **Variable**: $\rho$ (Signal-to-Noise Ratio)
- **Units**: Dimensionless
- **Range**: $[0, 0.5]$
- **Physical Meaning**: $\rho = \frac{\sigma}{\|\mathbf{b}\|_2}$ represents the ratio of noise standard deviation to RHS magnitude

**What it does:** Simulates when every measurement has some random error
**Applications:**
- Models uniform measurement errors in data acquisition
- Represents discretization errors in finite element methods
- Simulates floating-point round-off errors

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot \rho$$

---

### 2. Single Dimension Exact Percentage Perturbation

**Mathematical Formulation:**
$$\tilde{b}_i = b_i + p \cdot |b_i| \cdot \text{sign}(b_i) = b_i \cdot (1 + p), \quad \tilde{b}_j = b_j \text{ for } j \neq i$$

Where $p$ is the exact percentage perturbation and $i$ is randomly selected or specified.

**Simple Explanation:** Changes exactly one component by a specific percentage. Like one sensor being miscalibrated.

**Parameter Space:**
- **Variable**: $p$ (Local Percentage Perturbation)
- **Units**: Dimensionless
- **Range**: $[0, 1.0]$ (0% to 100% perturbation)
- **Physical Meaning**: Exact relative error in measurement or scaling of component $i$

**Deterministic Properties:**
- **Sign preservation**: $\text{sign}(\tilde{b}_i) = \text{sign}(b_i)$ for $p > -1$
- **Magnitude scaling**: $|\tilde{b}_i| = |b_i| \cdot (1 + p)$
- **Predictable**: No stochastic component, only dimension selection is random
- **Reversible**: Original value recoverable as $b_i = \tilde{b}_i / (1 + p)$

**What it does:** Tests what happens when one specific measurement is systematically wrong
**Applications:**
- Models calibration errors in sensors (multiplicative bias)
- Represents scaling uncertainties in measurement devices
- Simulates component-specific systematic errors
- Tests robustness to localized parameter variations

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot p \cdot \frac{|b_i|}{\|\mathbf{b}\|_2}$$

**Worst-Case Analysis:**
For the component with largest magnitude $|b_{\max}| = \max_j |b_j|$:
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot p \cdot \frac{|b_{\max}|}{\|\mathbf{b}\|_2}$$

---

### 3. Block-wise Correlated Noise

**Mathematical Formulation:**
$$\tilde{\mathbf{b}} = \mathbf{b} + \sum_{k=1}^{m} \epsilon_k \bf{\chi}_k$$

Where:
- $\epsilon_k \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = \rho \cdot \|\mathbf{b}\|_2$
- $\bf{\chi}_k$ is the indicator vector for block $B_k$
- $\{B_1, B_2, \ldots, B_m\}$ forms a partition of $\{1, 2, \ldots, n\}$

**Simple Explanation:** Groups nearby components and adds the same random error to each group. Like having correlated measurement errors in neighboring sensors.

**Parameter Space:**
- **Variable**: $\rho$ (Block Correlation Strength)
- **Units**: Dimensionless
- **Range**: $[0, 0.5]$
- **Physical Meaning**: Magnitude of spatially correlated perturbations

**What it does:** Tests robustness when nearby measurements have similar errors (spatial correlation)
**Applications:**
- Models correlated errors in discretized PDEs
- Represents material property uncertainties in finite elements
- Simulates spatially correlated measurement errors

**Block Size Selection:**
Automatic block size: $\text{block\_size} = \max(1, \lfloor n/6 \rfloor)$ to create approximately 4-8 blocks.

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot \rho \cdot \sqrt{m}$$

Where $m$ is the number of blocks.

---

### 4. Worst-Case Direction Perturbations

**Mathematical Formulation:**
$$\tilde{\mathbf{b}} = \mathbf{b} + \epsilon \cdot \|\mathbf{b}\|_2 \cdot \mathbf{v}_{\min}$$

Where:
- $\mathbf{v}_{\min}$ is the eigenvector corresponding to $\lambda_{\min}(\mathbf{A}^T\mathbf{A})$
- $\epsilon = \pm \frac{\alpha}{\kappa(\mathbf{A})}$ with random sign

**Simple Explanation:** Adds noise in the "worst possible direction" - the direction that causes maximum error amplification. Tests the algorithm's weakness.

**Parameter Space:**
- **Variable**: $\alpha$ (Condition-Scaled Perturbation Parameter)
- **Units**: Condition-number-scaled
- **Range**: $[0, 3.0]$
- **Physical Meaning**: $\alpha = \epsilon \times \kappa(\mathbf{A})$ represents the theoretical error amplification factor

**Why Minimum Eigenvalue Direction:**
Perturbations along $\mathbf{v}_{\min}$ (the eigenvector of the smallest eigenvalue) are amplified most by the condition number, representing the worst-case scenario for numerical stability.

**What it does:** Deliberately attacks the solver's biggest weakness to test worst-case performance
**Applications:**
- Tests sensitivity to condition-number-dependent errors
- Models perturbations that maximize solution error
- Evaluates preconditioner effectiveness against ill-conditioning

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \approx \alpha$$

This bound is achieved by construction since perturbations are scaled by the condition number.

---

### 5. Load Redistribution

**Mathematical Formulation:**
$$\tilde{\mathbf{b}} = \mathbf{b} + \mathbf{R}(\mathbf{b}, \rho)$$

Where the redistribution operator $\mathbf{R}$ satisfies:

1. **Mass Conservation**: $\sum_{i=1}^n \tilde{b}_i = \sum_{i=1}^n b_i$
2. **Magnitude Bound**: $\|\tilde{\mathbf{b}} - \mathbf{b}\|_\infty \leq \rho \cdot \|\mathbf{b}\|_2$
3. **Random Transfer**: Components exchange load pairwise

**Implementation:**
For $k = 1, \ldots, \lfloor n/4 \rfloor$:
1. Select indices $i, j$ randomly without replacement
2. Generate transfer amount $\tau \sim \mathcal{U}(-\rho \cdot \|\mathbf{b}\|_2, \rho \cdot \|\mathbf{b}\|_2)$
3. Update: $\tilde{b}_i \leftarrow \tilde{b}_i - \tau$, $\tilde{b}_j \leftarrow \tilde{b}_j + \tau$

**Simple Explanation:** Moves "load" from one component to another while keeping the total unchanged. Like redistributing weight in a structure.

**Parameter Space:**
- **Variable**: $\rho$ (Relative Redistribution Magnitude)
- **Units**: Relative L2-norm
- **Range**: $[0, 0.3]$
- **Physical Meaning**: $\rho = \frac{\|\Delta\mathbf{b}\|_2}{\|\mathbf{b}\|_2}$ represents the relative magnitude of load transfer

**What it does:** Tests what happens when load moves around but total amount stays the same
**Applications:**
- Models load transfer in structural analysis due to member failure
- Represents flow redistribution in fluid networks
- Simulates load balancing effects in parallel computing

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot \rho$$

---

### 6. Missing Data Corruption

**Mathematical Formulation:**
$$\tilde{\mathbf{b}} = (\mathbf{I} - \mathbf{M})\mathbf{b}$$

Where $\mathbf{M} = \text{diag}(m_1, m_2, \ldots, m_n)$ with $m_i \sim \text{Bernoulli}(\rho)$.

**Simple Explanation:** Randomly sets some components to zero. Like having sensors fail or missing measurements.

**Parameter Space:**
- **Variable**: $\rho$ (Corruption Rate)
- **Units**: Fraction
- **Range**: $[0, 0.4]$
- **Physical Meaning**: Fraction of components set to zero (missing data)

**What it does:** Tests robustness when some measurements are completely missing
**Applications:**
- Models sensor failures in measurement systems
- Represents missing boundary conditions
- Simulates data loss in communication systems

**Expected Sparsity:**
$$\mathbb{E}[\text{number of zeros}] = \rho \cdot n$$

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot \rho \cdot \frac{\|\mathbf{b}\|_\infty}{\|\mathbf{b}\|_2}$$

---

### 7. Corrupted Data

**Mathematical Formulation:**
$$\tilde{\mathbf{b}} = (\mathbf{I} - \mathbf{M})\mathbf{b} + \mathbf{M}\bf{\eta}$$

Where:
- $\mathbf{M} = \text{diag}(m_1, m_2, \ldots, m_n)$ with $m_i \sim \text{Bernoulli}(\rho)$
- $\bf{\eta} \sim \mathcal{N}(\mathbf{0}, (2 \cdot \|\mathbf{b}\|_2)^2 \mathbf{I})$

**Simple Explanation:** Randomly replaces some components with completely wrong random values. Like having malfunctioning sensors giving garbage readings.

**Parameter Space:**
- **Variable**: $\rho$ (Corruption Rate)
- **Units**: Fraction
- **Range**: $[0, 0.4]$
- **Physical Meaning**: Fraction of components replaced with random values

**What it does:** Tests what happens when some measurements are completely wrong (not just noisy)
**Applications:**
- Models data corruption in digital systems
- Represents outliers in measurement data
- Simulates adversarial attacks on sensor networks

**Corruption Amplitude:**
The corruption values have standard deviation $2 \times \|\mathbf{b}\|_2$, making them significantly different from the original values.

**Theoretical Bound:**
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \cdot \rho \cdot 4$$

---

## Algorithmic Implementation

### Parameter Conversion

Each perturbation method uses its natural parameter space but internally converts to a standardized `noise_std_pct` for the underlying noise generation functions:

```python
def convert_to_noise_pct(strategy: str, parameter_value: float, A: np.ndarray) -> float:
    if strategy in ["global", "single_dim", "blockwise"]:
        return parameter_value * 100.0  # ρ → percentage
    elif strategy in ["worst_case_max", "worst_case_min"]:
        kappa = np.linalg.cond(A)
        epsilon = parameter_value / kappa
        return epsilon * 100.0  # α/κ(A) → percentage
    elif strategy == "load_redistribution":
        return parameter_value * 100.0  # ρ → percentage
    elif strategy in ["missing_data", "corrupted_data"]:
        return parameter_value * 100.0  # fraction → percentage
```

### Theoretical Bounds Computation

```python
def compute_theoretical_bounds(strategy: str, parameter_range: list, A: np.ndarray, b: np.ndarray) -> list:
    kappa = np.linalg.cond(A)
    bounds = []

    for param_value in parameter_range:
        if strategy in ["global", "single_dim", "blockwise"]:
            bound = kappa * param_value
        elif strategy in ["worst_case_max", "worst_case_min"]:
            bound = param_value  # By construction
        elif strategy == "load_redistribution":
            bound = kappa * param_value
        elif strategy in ["missing_data", "corrupted_data"]:
            bound = kappa * param_value * 2.0  # Safety factor

        bounds.append(bound)

    return bounds
```

---

## Visualization and Analysis

### Algorithm-Specific X-Axis Labels

Each perturbation method uses its mathematically appropriate parameter space:

| Method | X-Axis Label | Units | Physical Meaning |
|--------|--------------|-------|------------------|
| Global | Signal-to-Noise Ratio (σ/‖b‖₂) | Dimensionless | Noise relative to signal |
| Single Dim | Local Percentage Perturbation (p) | Dimensionless | Exact relative component error |
| Block-wise | Block Correlation Strength (σ/‖b‖₂) | Dimensionless | Spatially correlated noise |
| Extreme Magnitude | Magnitude Factor (‖b̃‖₂/‖b‖₂) | Dimensionless | RHS magnitude scaling |
| Load Redis | Relative Redistribution (‖Δb‖₂/‖b‖₂) | Relative L2-norm | Mass-conserving transfer |
| Missing | Corruption Rate (fraction) | Fraction | Missing data percentage |
| Corrupted | Corruption Rate (fraction) | Fraction | Corrupted data percentage |

### Comparative Analysis

The enhanced framework enables fair comparison through efficiency measurement:

1. **Convergence efficiency**: Y-axis shows iterations needed to reach the same tolerance
2. **Method-specific plots**: Each perturbation type uses its natural parameter space
3. **Lower is better**: Fewer iterations indicate superior preconditioner performance
4. **Condition number display**: Show κ(A) prominently in all plots for theoretical context

**Key Insight**: Instead of comparing "residual after N iterations", we compare "iterations needed for tolerance ε". This gives a true measure of preconditioner effectiveness - the best preconditioner reaches the target accuracy fastest.

---

## Usage Guidelines

### Choosing Perturbation Methods

- **Gaussian methods** (global, single_dim, blockwise): Test robustness to measurement/discretization errors
- **Worst-case directions**: Evaluate sensitivity to ill-conditioning
- **Load redistribution**: Test equilibrium-preserving perturbations
- **Corruption methods**: Assess robustness to data quality issues

### Parameter Range Selection

Recommended parameter ranges balance physical realism with numerical stability:

- **Gaussian methods**: ρ ∈ [0, 0.5] (up to 50% noise-to-signal ratio)
- **Worst-case**: α ∈ [0, 3.0] (up to 3× condition number amplification)
- **Redistribution**: ρ ∈ [0, 0.3] (up to 30% load transfer)
- **Corruption**: ρ ∈ [0, 0.4] (up to 40% corrupted components)

### Interpretation of Results

1. **Slope analysis**: Steeper slopes indicate higher sensitivity
2. **Threshold identification**: Find parameter values where methods fail
3. **Relative performance**: Compare preconditioners within each perturbation type
4. **Cross-method robustness**: Identify universally robust vs. specialized methods

---

## References

1. **Higham, N. J.** (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
2. **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations*. Johns Hopkins University Press.
3. **Saad, Y.** (2003). *Iterative Methods for Sparse Linear Systems*. SIAM.
4. **Stewart, G. W. & Sun, J. G.** (1990). *Matrix Perturbation Theory*. Academic Press.