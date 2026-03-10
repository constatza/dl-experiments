"""Sparse RHS Strategy using SOLID provider + transform pattern.

Architecture:
    Layer 1 (Input): Broadcast sparse template to (N, n) matrix
    Layer 2 (Transform): SolveTransform computes x = A^-1 @ b
    Layer 3 (Strategy): Orchestrates template construction and solving
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..interfaces import ArchiveData, GeneratedSamples
from ..runner import register_strategy
from ..strategy_configs import SparseRhsConfig
from ..transforms import SolveTransform


@register_strategy
class SparseRhsStrategy:
    """Generate sparse RHS vectors and solve for solutions.

    SOLID Pattern:
        - Template builder: Constructs sparse (N, n) broadcast matrix (Layer 1)
        - SolveTransform: Computes x = A^-1 @ b (Layer 2)
        - Strategy: Orchestrates template construction and solving (Layer 3)

    Each RHS vector is identical: b[idx] = val for each (idx, val) pair.
    Negative indices are resolved relative to vector dimension n.
    The number of samples determines how many identical systems are solved,
    useful for testing preconditioner consistency across repeated solves.

    Configuration:
        - indices (list[int]): Positions of nonzero entries (negative supported)
        - values (list[float]): Values at specified positions
        - samples (int): Number of RHS vectors to generate
        - solve_config (SolveConfig): Solver configuration

    Examples:
        >>> cfg = {
        ...     "indices": [0, 1, -1],
        ...     "values": [90.0, 100.0, -100.0],
        ...     "samples": 1000,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "sparse_rhs"
    ConfigType = SparseRhsConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Build sparse RHS vectors and solve A x = b for each.

        Args:
            matrix: System matrix (already normalized by orchestrator).
            cfg: Configuration dictionary with keys:
                - indices: Positions of nonzero entries
                - values: Values at specified positions
                - samples: Number of RHS vectors to generate
                - solve_config: Optional SolveConfig parameters
            archive: Optional pre-loaded archive data (ignored).

        Returns:
            GeneratedSamples with sparse RHS vectors and solved solutions.

        Raises:
            ValueError: If indices and values lengths differ (checked at config time).
        """
        config = SparseRhsConfig(**cfg)

        n = matrix.shape[0]
        logger.info(
            f"Building {config.samples} sparse RHS vectors "
            f"(n={n}, {len(config.indices)} nonzeros)"
        )

        # Layer 1: broadcast sparse template to (N, n)
        rhs = _build_sparse_rhs(n, config.samples, config.indices, config.values)

        # Layer 2: solve x = A^-1 @ b for each sparse RHS
        solve_cfg = config.solve_config
        logger.info(f"Solving {config.samples} linear systems using {solve_cfg.method}...")
        transform = SolveTransform(
            matrix,
            method=solve_cfg.method,
            rtol=solve_cfg.rtol,
            atol=solve_cfg.atol,
            max_iters=solve_cfg.max_iters,
            assume_pos_def=solve_cfg.assume_pos_def,
        )
        solutions = transform.transform(rhs)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs,
            solutions=solutions,
            residual_traces=None,
            error_traces=None,
        )


def _build_sparse_rhs(
    n: int,
    samples: int,
    indices: list[int],
    values: list[float],
) -> np.ndarray:
    """Build a (samples, n) array where every row is the same sparse vector.

    Args:
        n: Dimension of each RHS vector.
        samples: Number of RHS vectors to generate.
        indices: Positions of nonzero entries (negative indices supported).
        values: Values at the specified positions.

    Returns:
        Array of shape (samples, n) with nonzeros set at given positions.
    """
    template = np.zeros(n, dtype=np.float64)
    for idx, val in zip(indices, values):
        template[idx] = val
    return np.broadcast_to(template, (samples, n)).copy()
