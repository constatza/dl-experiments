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


def generate_training_data(
    A, b, x0, tol=1e-6, cg_iter=10, noOfRandomVectors_Krylov=100, noOfRandomVectors=100
):
    n = len(b)
    x = x0.copy()

    # Initialize
    r = b - A.dot(x)
    z = r
    p = z.copy()
    rz_old = np.dot(r, z)

    matrixOfKrylovVectors = np.zeros((n, cg_iter))
    matrixOfKrylovVectors[:, 0] = p / norm(p)  # probably these shouldn't be normalized

    residuals = [norm(r)]

    for k in range(cg_iter - 1):
        Ap = A.dot(p)
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        residual_norm = norm(r)
        residuals.append(residual_norm)

        if residual_norm < tol:
            print(f"Converged at iteration {k + 1}, residual: {residual_norm:.2e}")
            break

        z = r
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        matrixOfKrylovVectors[:, k + 1] = p / norm(
            p
        )  # probably these shouldn't be normalized

    inputVectors_Krylov = np.zeros((n, noOfRandomVectors_Krylov))
    outputVectors_Krylov = np.zeros((n, noOfRandomVectors_Krylov))
    for k in range(noOfRandomVectors_Krylov):
        epsilon = np.random.normal(0, 1, size=cg_iter)
        inputVectors_Krylov[:, k] = matrixOfKrylovVectors @ epsilon
        outputVectors_Krylov[:, k] = A @ (matrixOfKrylovVectors @ epsilon)

    inputVectors_Random = np.zeros((n, noOfRandomVectors))
    outputVectors_Random = np.zeros((n, noOfRandomVectors))
    for k in range(noOfRandomVectors_Krylov):
        epsilon = np.random.normal(0, 1, size=n)
        inputVectors_Random[:, k] = epsilon
        outputVectors_Random[:, k] = A @ epsilon

    inputTrainingData = np.hstack((inputVectors_Krylov, inputVectors_Random))
    outputTrainingData = np.hstack((outputVectors_Krylov, outputVectors_Random))
    return inputTrainingData, outputTrainingData


def main():
    output_dir = Path(r"M:\shared\graph-cg\raw")
    output_dir.mkdir(exist_ok=True)
    matrix = np.loadtxt(output_dir / "matrix.txt")

    n = len(matrix)
    rhs = np.loadtxt(output_dir / "RHSs.txt")
    rhs = rhs[:, 0]
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

    cg_iter = 10
    noOfRandomVectors_Krylov = 100
    noOfTotallyRandomVectors = 100
    tol = 1e-6

    NN_RHS_output, NN_Solution_input = generate_training_data(
        matrix,
        rhs,
        x0,
        tol,
        cg_iter,
        noOfRandomVectors_Krylov,
        noOfTotallyRandomVectors,
    )

    np.save(output_dir / "rhs.npy", NN_RHS_output.T)
    np.save(output_dir / "solution.npy", NN_Solution_input.T)
    np.save(output_dir / "matrix.npy", matrix)


# Example usage
if __name__ == "__main__":
    main()
