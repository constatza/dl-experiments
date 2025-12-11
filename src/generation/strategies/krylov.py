"""Krylov strategy implementation using Lanczos basis."""

from __future__ import annotations

import numpy as np
from scipy.linalg import norm

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy
from ..runner import register_strategy


@register_strategy
class KrylovStrategy(IMatrixOnlyGenerationStrategy):
    name = "krylov"

    def requires_rhs(self) -> bool:
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict,
    ) -> GeneratedSamples:
        count = int(cfg.get("samples", 0))
        m = int(cfg.get("krylov_iters", 15))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))

        n = matrix.shape[0]
        V = np.zeros((n, m), dtype=np.float64)
        alpha = np.zeros(m, dtype=np.float64)
        beta = np.zeros(m + 1, dtype=np.float64)

        v = rng.normal(size=n).astype(np.float64, copy=False)
        v = v / norm(v)
        V[:, 0] = v

        v_prev = np.zeros(n, dtype=np.float64)
        beta[0] = 0.0

        m_eff = m
        for j in range(m):
            w = matrix @ V[:, j] - beta[j] * v_prev
            alpha[j] = np.dot(V[:, j], w)
            w = w - alpha[j] * V[:, j]
            beta[j + 1] = norm(w)
            if beta[j + 1] <= 1e-14:
                m_eff = j + 1
                V = V[:, :m_eff]
                alpha = alpha[:m_eff]
                beta = beta[: m_eff + 1]
                break
            v_prev = V[:, j].copy()
            if j + 1 < m:
                V[:, j + 1] = w / beta[j + 1]

        T = (
            np.diag(alpha[:m_eff])
            + np.diag(beta[1:m_eff], k=-1)
            + np.diag(beta[1:m_eff], k=1)
        )

        Lambda, Q = np.linalg.eigh(T)
        Lambda_inv = 1.0 / Lambda

        rhs_blocks: list[np.ndarray] = []
        sol_blocks: list[np.ndarray] = []

        for _ in range(count):
            eps = rng.normal(size=m_eff).astype(np.float64, copy=False)
            x = V @ (Q @ (Lambda_inv * eps))
            b_sample = matrix @ x
            rhs_blocks.append(b_sample)
            sol_blocks.append(x)

        rhs_out = np.array(rhs_blocks, dtype=np.float64)
        sol_out = np.array(sol_blocks, dtype=np.float64)
        return GeneratedSamples(matrix=matrix, rhs=rhs_out, solutions=sol_out)
