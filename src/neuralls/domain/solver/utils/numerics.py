"""Numerical utility functions for iterative solvers.

This module provides numerically stable implementations of common operations:
- stable_dot_product: Overflow-safe dot product computation
- compute_curvature: Curvature computation with breakdown detection

Extracted from helpers.py to follow Single Responsibility Principle.

Theory:
    Numerical stability is critical in iterative solvers where rounding errors
    accumulate. These utilities prevent catastrophic overflow/underflow while
    maintaining precision in normal ranges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import norm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def stable_dot_product(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> float:
    """Compute dot product with overflow prevention via balanced scaling.

    Mathematically equivalent to np.dot(a, b) but scales both vectors
    by the same factor to prevent overflow, then adjusts the result.

    Theory:
        For dot(a, b), if we compute dot(a/s, b/s), we get:
            sum((a_i/s) * (b_i/s)) = sum(a_i * b_i) / s^2 = dot(a,b) / s^2
        So: dot(a, b) = s^2 * dot(a/s, b/s)

        This is mathematically exact, just reordered for numerical stability.

    Args:
        a: First vector for dot product, shape (n,).
        b: Second vector for dot product, shape (n,).
        dtype: Data type for overflow threshold calculation (default: np.float64).

    Returns:
        Dot product a^T b, computed stably to avoid overflow.

    Example:
        >>> stable_dot_product(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        11.0

        >>> # Extreme case - automatic scaling prevents overflow
        >>> a = np.array([1e150, 1e150])
        >>> b = np.array([1e150, 1e150])
        >>> result = stable_dot_product(a, b)
        >>> np.isfinite(result)
        True
    """
    max_a = np.max(np.abs(a))
    max_b = np.max(np.abs(b))

    # Get dtype-specific overflow threshold
    from neuralls.shared.constants import get_overflow_threshold

    overflow_threshold = get_overflow_threshold(np.float64)

    # Check for overflow risk
    sqrt_threshold = np.sqrt(overflow_threshold)  # ~1.3e154 for float64

    if max_a > sqrt_threshold or max_b > sqrt_threshold:
        # At least one value is large - need to scale
        # Scale both vectors by the same factor to keep product safe
        max_val = max(max_a, max_b)
        target = np.sqrt(sqrt_threshold)  # ~1.1e77 for float64
        scale = target / max_val
        # Mathematically: dot(a,b) = (1/scale^2) * dot(a*scale, b*scale)
        return float(np.dot(a * scale, b * scale) / (scale * scale))

    return float(np.dot(a, b))


def compute_curvature(
    p: NDArray,
    q: NDArray,
    *,
    breakdown_tol: float = 1e-14,
) -> tuple[float, bool]:
    """Compute curvature d = p^T A p with breakdown detection.

    Args:
        p: Search direction vector.
        q: Matrix-vector product q = A @ p.
        breakdown_tol: Threshold for breakdown detection (default: 1e-14).

    Returns:
        Tuple (curvature, breakdown):
        - curvature: Scalar d = p^T q
        - breakdown: True if breakdown detected (d too small or negative)

    Theory:
        The curvature d = p^T A p = p^T q must be positive for descent.
        Negative or tiny d signals numerical issues or non-SPD behavior.

        Breakdown conditions:
        - d <= 0: Non-positive definite (restart needed)
        - |d| < breakdown_tol * ||p||^2: Too small (numerical precision loss)

    Example:
        >>> p = np.array([1.0, 2.0, 3.0])
        >>> q = np.array([2.0, 4.0, 6.0])
        >>> d, breakdown = compute_curvature(p, q)
        >>> breakdown
        False
    """
    # Use stable dot product to prevent overflow
    d = stable_dot_product(p, q)

    # Check for breakdown
    if not np.isfinite(d):
        return float(d), True

    # Absolute threshold
    from neuralls.shared.constants import get_breakdown_tol

    abs_threshold = get_breakdown_tol(np.float64)
    if abs(d) < abs_threshold:
        return float(d), True

    # Relative threshold
    p_norm_sq = float(norm(p)) ** 2
    rel_threshold = breakdown_tol * p_norm_sq

    if d <= 0 or d < rel_threshold:
        return float(d), True

    return float(d), False


def check_breakdown(value: float, threshold: float = 1e-14) -> bool:
    """Check if value indicates numerical breakdown.

    Args:
        value: Value to check.
        threshold: Threshold for breakdown detection.

    Returns:
        True if breakdown detected (NaN, Inf, or too small).

    Example:
        >>> check_breakdown(1e-20)
        True
        >>> check_breakdown(1.0)
        False
    """
    return not np.isfinite(value) or abs(value) < threshold
