"""Pure linear algebra utilities for the domain layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import norm

from neuralls.shared.types import MatrixNormType


def calculate_normalization_scale(A: np.ndarray) -> tuple[float, float]:
    """Calculate normalization scale from matrix A.

    Uses per-dimension matrix 1-norm: scale = ||A||_1 / dimension.
    The matrix 1-norm grows proportionally with the number of rows when the
    magnitude of entries stays constant. Dividing by the system dimension keeps
    the effective scale of the normalized matrix comparable across differently
    sized systems.

    Normalization is applied at data generation time only. All subsequent stages
    (training, inference, comparison) use pre-normalized artifacts.

    Args:
        A: System matrix

    Returns:
        Tuple of (scale, matrix_norm)
    """
    dimension = int(A.shape[0]) if A.ndim > 0 else 0
    matrix_norm = float(np.linalg.norm(A, ord=1))
    scale = matrix_norm / max(1, dimension)
    return scale, matrix_norm


def calculate_spectral_radius_bound(A: np.ndarray) -> float:
    """Calculate upper bound on spectral radius using Gershgorin theorem.

    The Gershgorin circle theorem provides an upper bound on the spectral radius
    (largest absolute eigenvalue) by taking the maximum of:
    - Maximum row sum of absolute values
    - Maximum column sum of absolute values

    This bound is used for matrix normalization to ensure the normalized matrix
    has spectral radius <= 1.

    Args:
        A: System matrix

    Returns:
        Upper bound on spectral radius
    """
    # Calculate maximum row sum of absolute values
    row_sums = np.sum(np.abs(A), axis=1)
    max_row_sum = float(np.max(row_sums))

    # Calculate maximum column sum of absolute values
    col_sums = np.sum(np.abs(A), axis=0)
    max_col_sum = float(np.max(col_sums))

    # Return the maximum of the two
    return max(max_row_sum, max_col_sum)


def calculate_spectral_norm(A: np.ndarray) -> float:
    """Calculate spectral norm (largest singular value) of matrix A.

    The spectral norm is defined as the largest singular value of A,
    which equals the square root of the largest eigenvalue of A^T A.
    This is computed using ||A||_2.

    Args:
        A: System matrix

    Returns:
        Spectral norm (largest singular value)
    """
    return float(np.linalg.norm(A, ord=2))


def calculate_frobenius_norm(A: np.ndarray) -> float:
    """Calculate Frobenius norm of matrix A.

    The Frobenius norm is defined as the square root of the sum of
    squared entries: ||A||_F = sqrt(sum(A_ij^2)).

    Args:
        A: System matrix

    Returns:
        Frobenius norm
    """
    return float(np.linalg.norm(A, ord="fro"))


def calculate_nuclear_norm(A: np.ndarray) -> float:
    """Calculate nuclear norm (trace norm) of matrix A.

    The nuclear norm is the sum of singular values of A.
    It's the dual norm of the spectral norm.

    Args:
        A: System matrix

    Returns:
        Nuclear norm (sum of singular values)
    """
    return float(np.linalg.norm(A, ord="nuc"))


def calculate_one_norm(A: np.ndarray) -> float:
    """Calculate 1-norm of matrix A.

    The 1-norm is the maximum absolute column sum:
    ||A||_1 = max_j sum_i |A_ij|

    Args:
        A: System matrix

    Returns:
        1-norm (max column sum)
    """
    return float(np.linalg.norm(A, ord=1))


def calculate_inf_norm(A: np.ndarray) -> float:
    """Calculate infinity norm of matrix A.

    The infinity norm is the maximum absolute row sum:
    ||A||_inf = max_i sum_j |A_ij|

    Args:
        A: System matrix

    Returns:
        Infinity norm (max row sum)
    """
    return float(np.linalg.norm(A, ord=np.inf))


# Norm strategy mapping - uses StrEnum as single source of truth
_NORM_STRATEGIES: dict[str, Callable[[np.ndarray], float]] = {
    MatrixNormType.SPECTRAL: calculate_spectral_norm,
    MatrixNormType.FROBENIUS: calculate_frobenius_norm,
    MatrixNormType.NUCLEAR: calculate_nuclear_norm,
    MatrixNormType.ONE: calculate_one_norm,
    MatrixNormType.INF: calculate_inf_norm,
}


def calculate_matrix_norm(
    A: np.ndarray,
    norm_type: str = "spectral",
) -> float:
    """Calculate matrix norm using specified norm type.

    Extensible dispatch function using StrEnum for type safety.
    Users pass strings - StrEnum provides validation internally.

    To add a new norm type:
    1. Add member to MatrixNormType enum above
    2. Add calculate_<name>_norm() function above
    3. Add entry to _NORM_STRATEGIES mapping

    Args:
        A: System matrix
        norm_type: Type of norm to compute (default: "spectral").
            Supported values:
            - "spectral": Spectral norm (2-norm, largest singular value)
            - "frobenius": Frobenius norm (sqrt of sum of squared entries)
            - "nuclear": Nuclear norm (sum of singular values)
            - "one": 1-norm (max column sum)
            - "inf": Infinity norm (max row sum)

    Returns:
        Matrix norm value

    Raises:
        ValueError: If norm_type is not recognized

    Examples:
        >>> A = np.array([[3, 0], [0, 4]])
        >>> calculate_matrix_norm(A, "spectral")  # Max singular value
        4.0
        >>> calculate_matrix_norm(A, "frobenius")  # sqrt(3^2 + 4^2)
        5.0
        >>> calculate_matrix_norm(A, "one")  # max(3, 4)
        4.0
    """
    if norm_type not in _NORM_STRATEGIES:
        supported = ", ".join(norm.value for norm in MatrixNormType)
        raise ValueError(f"Unknown matrix norm type: {norm_type!r}. Supported types: {supported}")

    norm_fn = _NORM_STRATEGIES[norm_type]
    return norm_fn(A)


def compute_dim_scale(dimension: int) -> float:
    """Calculate dimension-based scaling factor, sqrt(dimension).

    This scaling ensures consistent normalization across different matrix sizes.

    Args:
        dimension: Matrix dimension.

    Returns:
        sqrt(dimension).
    """
    return float(np.sqrt(dimension))


def apply_dimension_scaling(
    A: np.ndarray, b: np.ndarray, dimension_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply dimension scaling to matrix and RHS.

    Args:
        A: System matrix
        b: RHS vector
        dimension_scale: Scaling factor (typically sqrt(d))

    Returns:
        Tuple of (scaled_A, scaled_b)
    """
    return A / dimension_scale, b / dimension_scale


def normalize_by_matrix(
    A: np.ndarray, b: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize system (A, b) by matrix-based scale with dimension scaling.

    Both matrix and RHS are divided by scale factor and sqrt(d),
    preserving the solution: (A/s) x = (b/s) has same solution as Ax = b.

    This ensures consistent scaling across different matrix sizes.

    Args:
        A: System matrix
        b: RHS vector
        scale: Normalization scale (e.g., spectral radius bound)

    Returns:
        Tuple of (normalized_A, normalized_b)
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")

    dimension_scale = compute_dim_scale(A.shape[0])
    A_scaled = A / scale
    b_scaled = b / scale
    return apply_dimension_scaling(A_scaled, b_scaled, dimension_scale)


def normalize_by_rhs(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize system (A, b) by RHS norm with dimension scaling.

    Both matrix and RHS are divided by ||b|| and sqrt(d), which preserves
    the solution and results in a normalized RHS.

    Args:
        A: System matrix
        b: RHS vector

    Returns:
        Tuple of (normalized_A, normalized_b, rhs_norm)
    """
    rhs_norm = float(norm(b))
    if rhs_norm < 1e-15:
        raise ValueError(f"RHS norm too small for normalization: {rhs_norm}")

    dimension_scale = compute_dim_scale(A.shape[0])
    A_scaled = A / rhs_norm
    b_scaled = b / rhs_norm
    A_norm, b_norm = apply_dimension_scaling(A_scaled, b_scaled, dimension_scale)
    return A_norm, b_norm, rhs_norm


def normalize_by_spectral(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, spectral_norm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize system by factoring out spectral norm and RHS norm into solution.

    Transforms the system Ax = b into A_norm @ x_norm = b_norm where:
    - A_norm = A / (spectral_norm * sqrt(d))
    - b_norm = b / (||b|| * sqrt(d))
    - x_norm = x * spectral_norm / ||b||

    This normalization ensures:
    - All matrices have spectral norm bound ≈ 1/sqrt(d)
    - All RHS vectors have unit norm / sqrt(d)
    - Solution magnitude = spectral_norm / ||rhs||

    Args:
        A: System matrix
        b: RHS vector
        x: Solution vector
        spectral_norm: Spectral norm of A (from calculate_spectral_radius_bound)

    Returns:
        Tuple of (normalized_A, normalized_b, normalized_x)
    """
    if spectral_norm <= 0:
        raise ValueError(f"Spectral norm must be positive, got {spectral_norm}")

    rhs_norm = float(np.linalg.norm(b, ord=2))
    if rhs_norm < 1e-15:
        raise ValueError(f"RHS norm too small for normalization: {rhs_norm}")

    dimension_scale = compute_dim_scale(b.shape[0])

    # Normalize matrix by spectral norm and dimension
    A_scaled = A / spectral_norm
    A_norm, _ = apply_dimension_scaling(A_scaled, b, dimension_scale)

    # Normalize RHS by its norm and dimension
    b_scaled = b / rhs_norm
    _, b_norm = apply_dimension_scaling(A_scaled, b_scaled, dimension_scale)

    # Solution scaled by spectral_norm / rhs_norm
    x_norm = x * spectral_norm / rhs_norm

    return A_norm, b_norm, x_norm


def compute_condition_number(A: np.ndarray) -> float:
    """Compute condition number of matrix A.

    Args:
        A: Input matrix

    Returns:
        Condition number
    """
    return np.linalg.cond(A)


def vector_stats(v: np.ndarray) -> dict[str, float]:
    """Compute statistics for a vector.

    Args:
        v: Input vector

    Returns:
        Dictionary with vector statistics
    """
    return {
        "norm": float(norm(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def compute_rhs_norm(rhs: np.ndarray) -> float:
    """Compute L2 norm of RHS vector.

    Args:
        rhs: RHS vector

    Returns:
        L2 norm of rhs
    """
    return float(np.linalg.norm(rhs, ord=2))


def compute_rhs_norms(rhs_samples: np.ndarray) -> np.ndarray:
    """Compute L2 norms for batch of RHS vectors.

    Args:
        rhs_samples: Array of RHS vectors, shape (N, n)

    Returns:
        Array of L2 norms, shape (N,)
    """
    return np.array([compute_rhs_norm(rhs) for rhs in rhs_samples])


def extract_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Extract diagonal from matrix.

    Args:
        matrix: Square matrix

    Returns:
        Diagonal as 1D array
    """
    return np.diag(matrix).astype(np.float64, copy=False)


def validate_diagonal(diagonal: np.ndarray, threshold: float = 1e-15) -> None:
    """Validate diagonal has no near-zero entries.

    Args:
        diagonal: Diagonal array
        threshold: Minimum absolute value threshold

    Raises:
        ValueError: If any diagonal entry is below threshold
    """
    if np.any(np.abs(diagonal) < threshold):
        raise ValueError(
            "Matrix contains near-zero diagonal entries; cannot apply diagonal normalization."
        )


def compute_spectral_bound(matrix: np.ndarray) -> float:
    """Compute spectral radius bound using Gershgorin theorem.

    Args:
        matrix: System matrix

    Returns:
        Upper bound on spectral radius
    """
    return calculate_spectral_radius_bound(matrix)
