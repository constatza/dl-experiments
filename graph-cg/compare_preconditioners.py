# !/usr/bin/env python3
"""Created on Mon Jun 16 00:22:39 2025

@author: ioannis
"""

import numpy as np
from pathlib import Path
from scipy.sparse.linalg import spilu
from scipy.linalg import norm


def preconditioned_cg(A, b, x0, tol=1e-6, max_iter=1000, preconditioner=None):
    n = len(b)
    x = x0.copy()

    # Preconditioner setup
    if preconditioner == "ilu":
        ilu = spilu(A)
        Mx = lambda x: ilu.solve(x)
    elif preconditioner == "jacobi":
        D_inv = 1.0 / A.diagonal()
        Mx = lambda x: D_inv * x
    elif preconditioner is None or preconditioner == "none":
        Mx = lambda x: x
    else:
        raise ValueError("Unknown preconditioner type. Use 'ilu', 'jacobi', or None.")

    # Initialize
    r = b - A.dot(x)
    z = Mx(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    residuals = [norm(r)]

    for k in range(max_iter):
        Ap = A.dot(p)
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        residual_norm = norm(r)
        residuals.append(residual_norm)

        if residual_norm < tol:
            print(f"Converged at iteration {k + 1}, residual: {residual_norm:.2e}")
            break

        z = Mx(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x, {
        "converged": residual_norm < tol,
        "iterations": k + 1,
        "residual": residual_norm,
        "residual_history": residuals,
    }


def main(*, matrix_path: Path | str):
    matrix = np.loadtxt(matrix_path)
    # !! TEMPORARY FIX !!
    rhs = np.ones_like(matrix[:, 0])

    x0 = np.zeros_like(rhs)

    # Run with no preconditioner
    x, info = preconditioned_cg(
        matrix, rhs, x0, tol=1e-8, max_iter=1000, preconditioner=None
    )

    # Run with Jacobi
    x_jacobi, info_jacobi = preconditioned_cg(
        matrix, rhs, x0, tol=1e-8, max_iter=1000, preconditioner="jacobi"
    )

    # Run with ILU
    x_ilu, info_ilu = preconditioned_cg(
        matrix, rhs, x0, tol=1e-8, max_iter=1000, preconditioner="ilu"
    )


# Example usage
if __name__ == "__main__":
    matrix_file = "PlaneStress20x20dofSystem.txt"
    io_dir = Path(r"M:\shared\graph-cg\raw")
    matrix_path = io_dir / matrix_file
    main(matrix_path=matrix_path)
