#!/usr/bin/env python3
"""Created on Mon Jul 21 23:49:07 2025

@author: ioannis
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from pathlib import Path
from dynaconf import Dynaconf


def eigenspace_projection(
    matrix: np.ndarray, rhs: np.ndarray, num_vectors: int
) -> np.ndarray:
    # Generate a small symmetric positive definite (SPD) matrix A
    n = matrix.shape[0]

    # Generate a right-hand side vector b

    # === Dimensionality Reduction via Eigenvectors ===
    # Compute the r smallest eigenvectors of A
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(
        matrix, k=num_vectors, which="SM"
    )  # Smallest magnitude

    # Reduced basis
    V_eig = eigvecs  # n x r
    A_r_eig = V_eig.T @ matrix @ V_eig  # r x r reduced matrix
    b_r_eig = V_eig.T @ rhs
    x_r_eig = V_eig @ np.linalg.solve(A_r_eig, b_r_eig)  # approximate solution
    return V_eig


# === Dimensionality Reduction via CG-based Krylov Subspace ===
def conjugate_gradient_krylov(matrix: np.ndarray, rhs: np.ndarray, num_iterations: int):
    """Returns the CG basis vectors up to num_iterations (orthonormalized).

    Args:
        matrix (np.ndarray): The matrix A.
        rhs (np.ndarray): The right-hand side vector b.
        num_iterations (int): The number of iterations.

    Returns:
        np.ndarray: The CG basis vectors.
    """
    n = len(rhs)
    Q = []
    P = []
    x = np.zeros_like(rhs)
    r_vec = rhs.copy()
    p = r_vec.copy()
    Q.append(r_vec / np.linalg.norm(r_vec))
    P.append(p / np.linalg.norm(p))

    for k in range(num_iterations - 1):
        Ap = matrix @ p
        alpha = (r_vec @ r_vec) / (p @ Ap)
        x = x + alpha * p
        r_new = r_vec - alpha * Ap
        beta = (r_new @ r_new) / (r_vec @ r_vec)
        p = r_new + beta * p
        r_vec = r_new
        q_new = r_vec / np.linalg.norm(r_vec)
        Q.append(q_new)
        P.append(p / np.linalg.norm(p))

    return np.column_stack(Q), np.column_stack(P)


def generate_krylov_samples(
    *,
    matrix: np.ndarray,
    rhs: np.ndarray,
    num_samples: int,
    num_krylov_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:  # np.ndarray:
    """Generate samples from the approximate inverse of a matrix using CG-based Krylov vectors.

    Args:
        matrix (np.ndarray): The matrix A.
        rhs (np.ndarray): The right-hand side vector b.
        num_samples (int): The number of samples to generate.
        num_krylov_iterations (int): The number of iterations for CG-based Krylov vectors.
    """
    # Generate standard multivariate Gaussian samples ε ~ N(0, I)
    n = matrix.shape[0]
    epsilon_samples = np.random.randn(n, num_samples)
    V_cg, _ = conjugate_gradient_krylov(matrix, rhs, num_krylov_iterations)  # n x r
    A_r_cg = V_cg.T @ matrix @ V_cg

    # Approximate inverse from CG-based Krylov vectors
    M_r = V_cg @ np.linalg.inv(A_r_cg) @ V_cg.T  # Low-rank approximation of A^{-1}

    # Generate samples x = M_r * ε
    x_samples = M_r @ epsilon_samples  # Shape: (n, num_samples)
    rhs_samples = matrix @ x_samples

    return x_samples, rhs_samples


def generate_simple_samples(
    *,
    matrix: np.ndarray,
    num_samples: int,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Generate standard multivariate Gaussian samples ε ~ N(0, I)
    n = matrix.shape[0]
    epsilon_samples = np.random.randn(n, num_samples)

    # Generate samples x = A  * ε
    rhs_samples = matrix @ epsilon_samples  # Shape: (n, num_samples)

    return epsilon_samples, rhs_samples


def generate_samples(
    *,
    matrix: np.ndarray,
    rhs: np.ndarray,
    num_samples_krylov: int,
    num_samples_simple: int,
    num_krylov_iterations: int,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = num_samples_krylov + num_samples_simple
    x_krylov, rhs_krylov = generate_krylov_samples(
        matrix=matrix,
        rhs=rhs,
        num_samples=num_samples_krylov,
        num_krylov_iterations=num_krylov_iterations,
    )
    x_simple, rhs_simple = generate_simple_samples(
        matrix=matrix, num_samples=num_samples_simple
    )
    # stack samples
    x_all = np.hstack((x_krylov, x_simple)).T
    rhs_all = np.hstack((rhs_krylov, rhs_simple)).T

    if shuffle:
        indices = np.random.permutation(num_samples)
        x_all = x_all[indices, :]
        rhs_all = rhs_all[indices, :]
    return x_all, rhs_all


def main(rhs_generated_path, x_generated_path, matrix_path, rhs_path):
    # Parameters
    num_samples_simple = 3000
    num_samples_krylov = 3000
    num_krylov_iterations = 15

    matrix = np.loadtxt(matrix_path)
    # rhs = np.loadtxt(rhs_path)
    size = matrix.shape[0]
    rhs = np.ones(size)
    print(f"Matrix shape: {matrix.shape}")
    scale = np.linalg.norm(matrix, ord=1)
    matrix = matrix / scale

    rhs = rhs / scale

    x_samples, rhs_samples = generate_samples(
        matrix=matrix,
        rhs=rhs,
        num_samples_simple=num_samples_simple,
        num_samples_krylov=num_samples_krylov,
        num_krylov_iterations=num_krylov_iterations,
        shuffle=True,
    )

    np.save(rhs_generated_path, rhs_samples)
    np.save(x_generated_path, x_samples)
    print(f"Saved to {rhs_generated_path}")


# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    name = ""

    settings = Dynaconf(
        settings_file="./config.toml",
    )
    # Paths
    matrix_path = Path(settings.PATHS.matrix_template)
    rhs_path = Path(settings.PATHS.rhs_template)

    generated_dir = Path(settings.PATHS.generated_dir)
    generated_dir.mkdir(exist_ok=True)
    if name is None or name == "":
        rhs_generated_path = generated_dir / f"{matrix_path.stem}-rhs.npy"
        x_generated_path = generated_dir / f"{matrix_path.stem}-solution.npy"
    else:
        rhs_generated_path = generated_dir / f"{name}-rhs.npy"
        x_generated_path = generated_dir / f"{name}-solution.npy"
    main(rhs_generated_path, x_generated_path, matrix_path, rhs_path)
