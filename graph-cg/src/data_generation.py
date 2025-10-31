"""Data generation strategies for training data creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np
from scipy.linalg import norm

from .constants import ConfigKeys
from .normalization import ResidualTraceSamples


@dataclass
class StrategyOutput:
    """Container for samples produced by a generation strategy."""

    rhs: np.ndarray
    solutions: np.ndarray
    residual_traces: ResidualTraceSamples | None = None


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Create random number generator from seed.

    Args:
        seed: Random seed (None for random)

    Returns:
        Random number generator
    """
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def rounded_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    """Convert proportions to integer counts that sum exactly to total.

    Args:
        total: Total number of samples
        proportions: Dictionary of strategy -> proportion

    Returns:
        Dictionary of strategy -> count
    """
    # Compute raw counts
    raw_counts = {k: v * total for k, v in proportions.items()}
    int_counts = {k: int(v) for k, v in raw_counts.items()}
    remainder = {k: raw_counts[k] - int_counts[k] for k in raw_counts}

    # Distribute remainder using largest-remainder method
    total_assigned = sum(int_counts.values())
    remaining = total - total_assigned

    # Sort by remainder descending and assign extra samples
    sorted_keys = sorted(remainder.keys(), key=lambda k: remainder[k], reverse=True)
    for i in range(remaining):
        key = sorted_keys[i % len(sorted_keys)]
        int_counts[key] += 1

    return int_counts


def normal_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
) -> StrategyOutput:
    """Generate samples using normal random solutions.

    Args:
        A: System matrix
        b: RHS vector (mother RHS, used for norm scaling if match_rhs_norm=True)
        count: Number of samples to generate
        rng: Random number generator
        match_rhs_norm: If True, scale generated RHS to consistent target scale
        target_rhs_scale: Target RHS norm scale (default: 1.0)

    Returns:
        Tuple of (features, targets) where features are RHS and targets are solutions
    """
    n = A.shape[0]

    # Generate random solutions (ensure float64)
    # Scale to get RHS norms around target_rhs_scale * sqrt(n)
    X = rng.normal(size=(count, n), scale=target_rhs_scale).astype(np.float64, copy=False)

    # Compute RHS = A @ x
    R = np.array([A @ x for x in X], dtype=np.float64)

    return StrategyOutput(rhs=R, solutions=X)


def krylov_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    krylov_iters: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
) -> StrategyOutput:
    """Generate samples using Lanczos-based Krylov subspace method.

    Implements the Lanczos algorithm to build a Krylov subspace basis, then
    samples solutions from this enriched subspace and computes corresponding RHS.
    This ensures A @ x = b holds exactly for all samples.

    Args:
        A: System matrix (SPD)
        b: RHS vector (mother RHS, not used in Krylov generation)
        count: Number of samples to generate
        krylov_iters: Dimension of Krylov subspace (m in the algorithm)
        rng: Random number generator
        match_rhs_norm: Unused (kept for compatibility)
        target_rhs_scale: Unused (kept for compatibility)

    Returns:
        Tuple of (features, targets) where features are RHS and targets are solutions,
        with A @ x = b guaranteed to hold exactly for each sample
    """
    n = A.shape[0]
    m = krylov_iters

    # Step 1: Build Lanczos basis V_m (n × m)
    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m + 1, dtype=np.float64)

    # Initialize: v1 = u / ||u||, u ~ N(0, I_n)
    v = rng.normal(size=n).astype(np.float64, copy=False)
    v = v / norm(v)
    V[:, 0] = v

    v_prev = np.zeros(n, dtype=np.float64)
    beta[0] = 0.0

    # Lanczos iteration
    m_eff = m  # Effective dimension (may be less if early termination)
    for j in range(m):
        # w = A @ v_j - β_j * v_{j-1}
        w = A @ V[:, j] - beta[j] * v_prev

        # α_j = v_j^T @ w
        alpha[j] = np.dot(V[:, j], w)

        # w = w - α_j * v_j
        w = w - alpha[j] * V[:, j]

        # β_{j+1} = ||w||
        beta[j + 1] = norm(w)

        # Early termination if w ≈ 0
        if beta[j + 1] <= 1e-14:
            m_eff = j + 1
            V = V[:, :m_eff]
            alpha = alpha[:m_eff]
            beta = beta[:m_eff + 1]
            break

        # v_{j+1} = w / β_{j+1}
        v_prev = V[:, j].copy()
        if j + 1 < m:
            V[:, j + 1] = w / beta[j + 1]

    # Step 2: Build tridiagonal matrix T_m from Lanczos coefficients
    T = np.diag(alpha[:m_eff]) + \
        np.diag(beta[1:m_eff], k=-1) + \
        np.diag(beta[1:m_eff], k=1)

    # Step 3: Eigendecompose T_m = Q Λ Q^T
    Lambda, Q = np.linalg.eigh(T)  # eigh for symmetric matrices
    Lambda_inv = 1.0 / Lambda  # Λ^{-1}

    # Step 4: Generate samples
    R = []
    X = []

    for _ in range(count):
        # Draw ε ~ N(0, I_m) in LOW dimension
        eps = rng.normal(size=m_eff).astype(np.float64, copy=False)

        # x = V_m @ Q @ Λ^{-1} @ ε
        x = V @ (Q @ (Lambda_inv * eps))

        # b = A @ x (ensures A @ x = b exactly!)
        b_sample = A @ x

        R.append(b_sample)
        X.append(x)

    return StrategyOutput(
        rhs=np.array(R, dtype=np.float64),
        solutions=np.array(X, dtype=np.float64),
    )


def _cg_trace(
    A: np.ndarray,
    b_vec: np.ndarray,
    max_iters: int,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a CG solve and capture residual/solution pairs per iteration."""

    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    r = b_vec.astype(np.float64, copy=False) - A @ x
    p = r.copy()
    rs_old = float(np.dot(r, r))

    residuals: list[np.ndarray] = []
    solutions: list[np.ndarray] = []

    if rs_old <= tol * tol:
        residuals.append(r.copy())
        solutions.append(x.copy())
        return np.vstack(residuals), np.vstack(solutions)

    for iteration in range(max_iters):
        residuals.append(r.copy())

        Ap = A @ p
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < tol:
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        solutions.append(x.copy())

        r_new = r - alpha * Ap
        rs_new = float(np.dot(r_new, r_new))
        if rs_new <= tol * tol:
            break

        beta = rs_new / rs_old
        p = r_new + beta * p
        r = r_new
        rs_old = rs_new

    if not residuals:
        residuals.append(r.copy())
    if len(solutions) < len(residuals):
        solutions.append(x.copy())

    return np.vstack(residuals), np.vstack(solutions)


def residual_trace_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    cg_iters: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
) -> StrategyOutput:
    """Generate samples augmented with CG residual traces for each system."""

    n = A.shape[0]

    solutions = rng.normal(size=(count, n), scale=target_rhs_scale).astype(np.float64, copy=False)
    rhs_samples = np.array([A @ x for x in solutions], dtype=np.float64)

    residual_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    iteration_indices: list[np.ndarray] = []

    for sample_idx, rhs_vec in enumerate(rhs_samples):
        residual_seq, solution_seq = _cg_trace(A, rhs_vec, max_iters=cg_iters)
        num_pairs = residual_seq.shape[0]

        residual_blocks.append(residual_seq)
        solution_blocks.append(solution_seq)
        sample_indices.append(
            np.full(num_pairs, sample_idx, dtype=np.int64)
        )
        iteration_indices.append(
            np.arange(num_pairs, dtype=np.int64)
        )

    residual_traces = ResidualTraceSamples(
        residuals=np.vstack(residual_blocks),
        solutions=np.vstack(solution_blocks),
        sample_indices=np.concatenate(sample_indices),
        iteration_indices=np.concatenate(iteration_indices),
    )

    return StrategyOutput(
        rhs=rhs_samples,
        solutions=solutions,
        residual_traces=residual_traces,
    )


# Registry of available strategies
STRATEGY_REGISTRY: dict[str, Callable] = {
    "normal": normal_strategy,
    "krylov": krylov_strategy,
    "cg_residual": residual_trace_strategy,
    "residual": residual_trace_strategy,
}


def _offset_residual_traces(
    traces: ResidualTraceSamples,
    offset: int,
) -> ResidualTraceSamples:
    """Shift sample indices in a residual trace block by a fixed offset."""

    return ResidualTraceSamples(
        residuals=traces.residuals,
        solutions=traces.solutions,
        sample_indices=traces.sample_indices + offset,
        iteration_indices=traces.iteration_indices,
    )


def _merge_residual_traces(
    blocks: list[ResidualTraceSamples],
) -> ResidualTraceSamples:
    """Concatenate multiple residual trace blocks."""

    residuals = np.vstack([block.residuals for block in blocks])
    solutions = np.vstack([block.solutions for block in blocks])
    sample_indices = np.concatenate([block.sample_indices for block in blocks])
    iteration_indices = np.concatenate([block.iteration_indices for block in blocks])
    return ResidualTraceSamples(
        residuals=residuals,
        solutions=solutions,
        sample_indices=sample_indices,
        iteration_indices=iteration_indices,
    )


def generate_mixture(
    A: np.ndarray,
    b: np.ndarray,
    mix: dict[str, float],
    total: int,
    krylov_iters: int = 15,
    residual_iters: int = 8,
    seed: int = 42,
    shuffle: bool = True,
    normalize: Literal["none", "matrix", "rhs"] = "matrix",
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, ResidualTraceSamples | None]:
    """Generate mixed training data from multiple strategies.

    Args:
        A: System matrix
        b: RHS vector (mother RHS)
        mix: Dictionary of strategy_name -> proportion
        total: Total number of samples
        krylov_iters: Number of CG iterations for Krylov strategy
        seed: Random seed
        shuffle: Whether to shuffle final dataset
        normalize: Normalization method
            - "none": no RHS norm matching
            - "matrix": no RHS norm matching (matrix already normalized)
            - "rhs": match RHS norms to mother RHS
        strategy_overrides: Optional mapping of strategy names to override
            dictionaries (e.g., {"krylov": {"krylov_iters": 10}})

    Returns:
        Tuple of (features, targets, residual_traces) arrays.
    """
    rng = rng_from_seed(seed)
    overrides: dict[str, dict[str, Any]] = {
        name: dict(options) for name, options in (strategy_overrides or {}).items()
    }

    # Validate mix proportions
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"Mix proportions must sum to 1.0, got {sum(mix.values())}")

    # Convert proportions to counts
    counts = rounded_counts(total, mix)

    # Calculate appropriate RHS scale for generation
    # Goal: Generate RHS with norms similar to what per-sample normalization expects
    # For "rhs" normalization: target RHS norms matching mother RHS so that
    # after dividing A by ||b||, we get matrix scales similar to collected data
    n = A.shape[0]
    if normalize == "rhs":
        # Use mother RHS norm as target to match the scale of collected data
        # For gaussian samples: E[||x||] ≈ scale * sqrt(n)
        # So: scale = target_norm / sqrt(n)
        target_norm = float(norm(b))  # Match mother RHS norm
        target_rhs_scale = target_norm / np.sqrt(n)
    else:
        # For other normalizations, use standard gaussian scale
        target_rhs_scale = 1.0

    match_rhs_norm = False  # Deprecated parameter, kept for compatibility

    all_features = []
    all_targets = []
    residual_blocks: list[ResidualTraceSamples] = []
    sample_offset = 0

    for strategy_name, count in counts.items():
        if count == 0:
            continue

        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy_func = STRATEGY_REGISTRY[strategy_name]

        strategy_options = overrides.get(strategy_name, {})
        override_seed = strategy_options.pop("seed", None)
        strategy_rng = rng if override_seed is None else rng_from_seed(int(override_seed))

        if strategy_name == "krylov":
            override_iters = strategy_options.pop(
                ConfigKeys.KRYLOV_ITERS,
                strategy_options.pop("krylov_iters", None),
            )
            krylov_count = int(override_iters) if override_iters is not None else krylov_iters
            output: StrategyOutput = strategy_func(
                A,
                b,
                count,
                krylov_count,
                strategy_rng,
                match_rhs_norm,
                target_rhs_scale,
            )
        elif strategy_name in {"cg_residual", "residual"}:
            override_cg_iters = strategy_options.pop(
                ConfigKeys.RESIDUAL_ITERS,
                strategy_options.pop("residual_iters", None),
            )
            cg_iterations = (
                int(override_cg_iters) if override_cg_iters is not None else residual_iters
            )
            output = strategy_func(
                A,
                b,
                count,
                cg_iterations,
                strategy_rng,
                match_rhs_norm,
                target_rhs_scale,
            )
        else:
            if strategy_options:
                unknown = ", ".join(sorted(strategy_options.keys()))
                raise ValueError(
                    f"Unsupported override keys for strategy '{strategy_name}': {unknown}"
                )
            output = strategy_func(
                A,
                b,
                count,
                strategy_rng,
                match_rhs_norm,
                target_rhs_scale,
            )

        all_features.append(output.rhs)
        all_targets.append(output.solutions)

        if output.residual_traces is not None:
            residual_blocks.append(_offset_residual_traces(output.residual_traces, sample_offset))

        sample_offset += output.rhs.shape[0]

    # Concatenate all samples
    X = np.vstack(all_features)
    Y = np.vstack(all_targets)
    residual_traces = _merge_residual_traces(residual_blocks) if residual_blocks else None

    # Shuffle if requested
    if shuffle:
        indices = rng.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        if residual_traces is not None:
            inverse = np.empty_like(indices)
            inverse[indices] = np.arange(len(indices))
            residual_traces = ResidualTraceSamples(
                residuals=residual_traces.residuals,
                solutions=residual_traces.solutions,
                sample_indices=inverse[residual_traces.sample_indices],
                iteration_indices=residual_traces.iteration_indices,
            )

    return X, Y, residual_traces
