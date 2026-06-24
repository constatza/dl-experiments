"""Multigrid cycle implementations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .hierarchy import MultigridHierarchy
from .protocols import MultigridSmoother


class VCycle:
    """Standard V-cycle: pre-smooth, recurse, prolongate, post-smooth.

    Args:
        smoother: Smoother applied before and after the coarse-grid correction.
        n_pre: Pre-smoothing steps.
        n_post: Post-smoothing steps.
    """

    def __init__(
        self,
        smoother: MultigridSmoother,
        n_pre: int = 2,
        n_post: int = 2,
    ) -> None:
        self._smoother = smoother
        self._n_pre = n_pre
        self._n_post = n_post

    def apply(self, hierarchy: MultigridHierarchy, rhs: NDArray) -> NDArray:
        """Compute one V-cycle starting from the finest level.

        Args:
            hierarchy: Pre-built multigrid hierarchy.
            rhs: Right-hand side vector on the finest grid.

        Returns:
            Approximate solution after one V-cycle (zero initial guess).
        """
        return self._vcycle(hierarchy, rhs, level_idx=0)

    def _vcycle(self, hierarchy: MultigridHierarchy, rhs: NDArray, level_idx: int) -> NDArray:
        level = hierarchy.levels[level_idx]
        A = level.matrix

        # Coarsest level: direct solve
        if level.transfer is None:
            return spsolve(csr_matrix(A), rhs)

        # Pre-smooth
        x = self._smoother.smooth(A, rhs, np.zeros_like(rhs), self._n_pre)

        # Restrict residual to coarse grid
        r_fine = rhs - A @ x
        r_coarse = level.transfer.restrict(r_fine)

        # Recursive coarse-grid correction
        e_coarse = self._vcycle(hierarchy, r_coarse, level_idx + 1)

        # Prolongate and correct
        x = x + level.transfer.prolongate(e_coarse)

        # Post-smooth
        return self._smoother.smooth(A, rhs, x, self._n_post)
