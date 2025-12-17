"""Synthetic benchmark generation and diagnostics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from loguru import logger
import pandas as pd

from src.io.comparison import load_npz_entry


def load_matrix(path: Path) -> np.ndarray:
    """Load matrix A from various file formats."""
    if path.suffix == ".npz":
        return load_npz_entry(path, "matrix")
    elif path.suffix == ".npy":
        return np.load(path)
    else:
        return np.loadtxt(path)


def generate_synthetic_test_case(
    matrix_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic test case where x_true is all ones.

    Args:
        matrix_path: Path to the matrix file (NPZ, NPY, or TXT).

    Returns:
        Tuple of (A, x_true, b) where b = A @ x_true.
    """
    logger.info(f"Generating synthetic test case from matrix: {matrix_path}")
    
    A = load_matrix(matrix_path)
    
    if A.ndim != 2:
        # If we accidentally loaded a 1D array or 3D tensor, try to handle or fail
        if A.ndim == 3:
             logger.warning("Loaded 3D matrix; using first slice.")
             A = A[0]
        else:
             raise ValueError(f"Loaded matrix must be 2D, got shape {A.shape}")
        
    n = A.shape[1]
    x_true = np.ones(n, dtype=A.dtype)
    b = A @ x_true
    
    logger.debug(f"Generated system size {n}x{n}")
    return A, x_true, b


def compute_synthetic_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute error metrics between true and predicted solutions."""
    if y_true.shape != y_pred.shape:
        # Try flattening if shapes mismatch but sizes match
        if y_true.size == y_pred.size:
             y_true = y_true.ravel()
             y_pred = y_pred.ravel()
        else:
            raise ValueError(
                f"Shape mismatch: true {y_true.shape} vs pred {y_pred.shape}"
            )
        
    error = y_pred - y_true
    
    # L2 Norms
    norm_true = np.linalg.norm(y_true)
    norm_error = np.linalg.norm(error)
    
    # Relative Error
    rel_error = norm_error / norm_true if norm_true > 0 else float("inf")
    
    return {
        "l2_error": float(norm_error),
        "l2_true": float(norm_true),
        "relative_error": float(rel_error),
        "max_error": float(np.max(np.abs(error))),
        "mean_error": float(np.mean(np.abs(error))),
    }


def save_synthetic_results(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identifier: str,
) -> Path:
    """Save synthetic benchmark results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"independent_{identifier}.csv"
    output_path = output_dir / filename
    
    # Compute error array
    error_array = y_pred - y_true
    
    # Create a DataFrame
    df = pd.DataFrame({
        "y_true": y_true.ravel(), # Ensure 1D
        "y_pred": y_pred.ravel(), # Ensure 1D
        "error": error_array.ravel() # Ensure 1D
    })
    
    df.to_csv(output_path, index=False) # Save to CSV

    logger.info(f"Saved synthetic results to {output_path}")
    
    # Also log the key metric
    metrics = compute_synthetic_error(y_true, y_pred)
    logger.info(f"Synthetic Relative Error: {metrics['relative_error']:.6e}")
    
    return output_path
