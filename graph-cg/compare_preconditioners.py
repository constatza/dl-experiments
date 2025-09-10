# !/usr/bin/env python3
"""Created on Mon Jun 16 00:22:39 2025

@author: ioannis
"""

import numpy as np
import torch

from functools import partial
from collections.abc import Callable
from typing import Literal
from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from scipy.sparse.linalg import spilu
from scipy.linalg import norm

from dlkit.gnn.wrap import GraphNetwork
from dlkit.gnn.transforms import SpectralRadiusNorm
from dlkit.settings import Settings

TOL = 1e-8
MAX_ITER = 1000


# Preconditioner setup
def ilu_preconditioner(A):
    ilu = spilu(A)
    return lambda x: ilu.solve(x)


def jacobi_preconditioner(A):
    D_inv = 1.0 / A.diagonal()
    return lambda x: D_inv * x


def gnn_preconditioner(
    A: np.ndarray,
    checkpoint_path: Path | str,
    device: str = "cuda",
    mode: Literal["training", "inference"] = "inference",
):
    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(A).float().to(device))
    data = partial(Data, edge_index=edge_index, edge_attr=edge_attr)
    transform = SpectralRadiusNorm()
    model = GraphNetwork.load_from_checkpoint(checkpoint_path)
    model.eval()

    def preconditioner(x, data=data):
        x = torch.from_numpy(x).float().to(device).view(-1, 1)
        data = transform(data(x=x))
        with torch.inference_mode():
            y_hat = (
                model(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                .cpu()
                .numpy()
            )
        return y_hat.squeeze()

    return preconditioner


def preconditioned_cg(
    A,
    b,
    x0,
    tol=1e-6,
    max_iter=1000,
    preconditioner: Callable[[np.ndarray], np.ndarray] = lambda x: x,
):
    n = len(b)
    x = x0.copy()

    # Initialize
    r = b - A.dot(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    residual_norm = norm(r)
    residuals = [residual_norm]

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

        z = preconditioner(r)
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


def prepare_preconditioners(matrix: np.ndarray):
    return {
        "ilu": ilu_preconditioner(matrix),
        "jacobi": jacobi_preconditioner(matrix),
        "gnn": gnn_preconditioner(matrix, mode="inference"),
    }


def main(*, matrix: np.ndarray, rhs: np.ndarray, checkpoint_path: Path | str):
    # !! TEMPORARY FIX !!

    x0 = np.zeros_like(rhs)

    # Run with no preconditioner
    base_cg = partial(preconditioned_cg, x0=x0, tol=TOL, max_iter=MAX_ITER)

    x, info = base_cg(matrix, rhs, preconditioner=lambda x: x)
    x_jacobi, info_jacobi = base_cg(
        matrix, rhs, preconditioner=jacobi_preconditioner(matrix)
    )
    x_ilu, info_ilu = base_cg(matrix, rhs, preconditioner=ilu_preconditioner(matrix))
    x_gnn, info_gnn = base_cg(
        matrix, rhs, preconditioner=gnn_preconditioner(matrix, checkpoint_path)
    )


# Example usage
if __name__ == "__main__":
    settings = Settings.from_file("./config.toml")

    matrix = np.loadtxt(settings.PATHS.matrix_template)
    rhs = np.loadtxt(settings.PATHS.rhs_template)
    checkpoint_path = settings.PIPELINE.checkpoint
    main(matrix=matrix, rhs=rhs, checkpoint_path=checkpoint_path)
