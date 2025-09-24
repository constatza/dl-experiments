"""Mathematical utilities for linear algebra operations."""

from __future__ import annotations
import numpy as np
from scipy.linalg import norm


def normalize_system(A: np.ndarray, b: np.ndarray, *, enabled: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Normalize linear system A*x = b by ||A||_1.

    Args:
        A: System matrix
        b: Right-hand side vector
        enabled: Whether to apply normalization

    Returns:
        Tuple of (normalized_A, normalized_b)
    """
    if not enabled:
        return A, b
    scale = np.linalg.norm(A, ord=1)
    if scale == 0:
        return A, b
    return A / scale, b / scale


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


def _auto_device(device: str | None = None) -> str:
    """Automatically determine device to use.

    Args:
        device: Specified device or None for auto-detection

    Returns:
        Device string ('cpu' or 'cuda')
    """
    if device is not None:
        return device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_csc(A: np.ndarray | object) -> object:
    """Convert matrix to CSC format for scipy operations.

    Args:
        A: Input matrix (dense or sparse)

    Returns:
        Matrix in CSC format
    """
    try:
        from scipy.sparse import csc_matrix, issparse
        if issparse(A):
            return A.tocsc()
        else:
            return csc_matrix(A)
    except ImportError:
        return A