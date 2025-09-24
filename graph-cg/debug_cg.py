#!/usr/bin/env python3
"""Debug the CG algorithm to find why Jacobi doesn't improve."""

import numpy as np
from scipy.linalg import norm

def debug_preconditioned_cg(A, b, x0, preconditioner, tol=1e-6, max_iter=50):
    """Debug version of preconditioned CG with detailed output."""
    print(f"Starting CG with tol={tol}, max_iter={max_iter}")

    x = x0.copy()
    r = b - A.dot(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    print(f"Initial: ||r||={norm(r):.3e}, r^T z={rz_old:.3e}")

    residual_norm = norm(r)
    residuals = [residual_norm]

    for k in range(max_iter):
        print(f"\n--- Iteration {k+1} ---")

        Ap = A.dot(p)
        pAp = np.dot(p, Ap)
        print(f"p^T A p = {pAp:.3e}")

        if abs(pAp) < 1e-15:
            print(f"Breakdown: p^T A p = {pAp:.3e} < 1e-15")
            break

        alpha = rz_old / pAp
        print(f"alpha = {alpha:.3e}")

        x += alpha * p
        r -= alpha * Ap
        residual_norm = norm(r)
        residuals.append(residual_norm)

        print(f"After update: ||r|| = {residual_norm:.3e}")

        if residual_norm < tol:
            print(f"Converged! ||r|| = {residual_norm:.3e} < {tol:.3e}")
            return x, {"converged": True, "iterations": k + 1, "residual": residual_norm, "residuals": residuals}

        z = preconditioner(r)
        rz_new = np.dot(r, z)
        print(f"r^T z_new = {rz_new:.3e}")

        if abs(rz_new) < 1e-15:
            print(f"Breakdown: r^T z = {rz_new:.3e} < 1e-15")
            break

        beta = rz_new / rz_old
        print(f"beta = {beta:.3e}")

        p = z + beta * p
        rz_old = rz_new

    print(f"Max iterations reached: {max_iter}")
    return x, {"converged": False, "iterations": max_iter, "residual": residual_norm, "residuals": residuals}

def test_debug():
    """Test both identity and Jacobi with debug output."""
    np.random.seed(42)
    n = 5  # Small for detailed output

    # Simple diagonally dominant matrix
    A = np.diag([10, 8, 6, 4, 2]) + 0.1 * np.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric
    b = np.ones(n)
    x0 = np.zeros(n)

    print("Matrix A:")
    print(A)
    print(f"Condition number: {np.linalg.cond(A):.3f}")
    print(f"True solution: {np.linalg.solve(A, b)}")

    # Identity preconditioner
    print("\n" + "="*50)
    print("IDENTITY PRECONDITIONER")
    print("="*50)
    identity = lambda x: x
    x1, info1 = debug_preconditioned_cg(A, b, x0, identity, tol=1e-8, max_iter=20)

    # Jacobi preconditioner
    print("\n" + "="*50)
    print("JACOBI PRECONDITIONER")
    print("="*50)
    D_inv = 1.0 / np.diag(A)
    jacobi = lambda x: D_inv * x
    x2, info2 = debug_preconditioned_cg(A, b, x0, jacobi, tol=1e-8, max_iter=20)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Identity: {info1['iterations']} iterations, ||r||={info1['residual']:.3e}")
    print(f"Jacobi:   {info2['iterations']} iterations, ||r||={info2['residual']:.3e}")

    if info2['iterations'] >= info1['iterations']:
        print("🐛 BUG: Jacobi should be better than identity!")
    else:
        print("✅ Jacobi improves over identity")

if __name__ == "__main__":
    test_debug()