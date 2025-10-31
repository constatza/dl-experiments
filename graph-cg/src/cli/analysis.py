"""Dataset analysis helpers for exploratory scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def analyze_dataset(data_dir: Path, name: str) -> Dict[str, Any]:
    """Compute descriptive statistics for a processed dataset."""

    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"Path: {data_dir}")
    print(f"{'=' * 70}")

    A = np.load(data_dir / "matrix.npy")
    rhs_samples = np.load(data_dir / "rhs-samples.npy")
    sol_samples = np.load(data_dir / "sol-samples.npy")
    rhs_mother = np.load(data_dir / "rhs-mother.npy")

    num_samples = rhs_samples.shape[0]
    dimension = A.shape[0]

    print("\nBasic Info:")
    print(f"  Samples: {num_samples}")
    print(f"  Dimension: {dimension}")
    print("  All samples use same matrix: True")

    print("\nMatrix A statistics:")
    print(f"  Norm (Frobenius): {np.linalg.norm(A, 'fro'):.6e}")
    print(f"  Norm (1-norm): {np.linalg.norm(A, 1):.6e}")
    print(f"  Norm (inf-norm): {np.linalg.norm(A, np.inf):.6e}")
    print(f"  Max absolute value: {np.max(np.abs(A)):.6e}")
    non_zero = np.abs(A[A != 0])
    if non_zero.size:
        print(f"  Min absolute value (non-zero): {np.min(non_zero):.6e}")
    print(f"  Condition number: {np.linalg.cond(A):.6e}")

    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(A, axis=0)
    print(
        "  Row norms - min: {:.6e}, max: {:.6e}, mean: {:.6e}".format(
            np.min(row_norms), np.max(row_norms), np.mean(row_norms)
        )
    )
    print(
        "  Col norms - min: {:.6e}, max: {:.6e}, mean: {:.6e}".format(
            np.min(col_norms), np.max(col_norms), np.mean(col_norms)
        )
    )

    rhs_norms = np.linalg.norm(rhs_samples, axis=1)
    mother_norm = float(np.linalg.norm(rhs_mother))
    print("\nRHS statistics:")
    print(f"  Mother RHS norm: {mother_norm:.6e}")
    print(f"  Sample RHS norms - min: {np.min(rhs_norms):.6e}, max: {np.max(rhs_norms):.6e}")
    print(f"  Sample RHS norms - mean: {np.mean(rhs_norms):.6e}, std: {np.std(rhs_norms):.6e}")
    print(f"  All RHS norms close to mother? {np.allclose(rhs_norms, mother_norm, rtol=1e-10)}")

    sol_norms = np.linalg.norm(sol_samples, axis=1)
    print("\nSolution statistics:")
    print(f"  Solution norms - min: {np.min(sol_norms):.6e}, max: {np.max(sol_norms):.6e}")
    print(f"  Solution norms - mean: {np.mean(sol_norms):.6e}, std: {np.std(sol_norms):.6e}")

    residuals = [np.linalg.norm(A @ sol_samples[i] - rhs_samples[i]) for i in range(min(100, num_samples))]
    print("\nResidual (||A @ x - b||) for first 100 samples:")
    print(f"  Min: {np.min(residuals):.6e}, Max: {np.max(residuals):.6e}")
    print(f"  Mean: {np.mean(residuals):.6e}, Median: {np.median(residuals):.6e}")

    gradients_from_zero = [
        np.linalg.norm(-2 * A.T @ rhs_samples[i]) for i in range(min(100, num_samples))
    ]
    print("\nGradient magnitude (if we start from x=0):")
    print(f"  Min: {np.min(gradients_from_zero):.6e}, Max: {np.max(gradients_from_zero):.6e}")
    print(f"  Mean: {np.mean(gradients_from_zero):.6e}, Std: {np.std(gradients_from_zero):.6e}")

    losses_at_zero = [np.linalg.norm(rhs_samples[i]) ** 2 for i in range(min(100, num_samples))]
    print("\nLoss at x=0 (||b||^2):")
    print(f"  Min: {np.min(losses_at_zero):.6e}, Max: {np.max(losses_at_zero):.6e}")
    print(f"  Mean: {np.mean(losses_at_zero):.6e}, Std: {np.std(losses_at_zero):.6e}")

    return {
        "name": name,
        "num_samples": num_samples,
        "dimension": dimension,
        "matrix_norm": np.linalg.norm(A, "fro"),
        "rhs_norm_mean": float(np.mean(rhs_norms)),
        "rhs_norm_std": float(np.std(rhs_norms)),
        "sol_norm_mean": float(np.mean(sol_norms)),
        "sol_norm_std": float(np.std(sol_norms)),
        "gradient_mean": float(np.mean(gradients_from_zero)),
        "gradient_std": float(np.std(gradients_from_zero)),
        "loss_at_zero_mean": float(np.mean(losses_at_zero)),
        "condition_number": float(np.linalg.cond(A)),
    }
