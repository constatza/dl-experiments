"""Multigrid cycle implementations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .hierarchy import MultigridHierarchy
from .protocols import MultigridSmoother


class VCycle:
    """Standard V-cycle multigrid preconditioner (γ = 1).

    Descends the hierarchy pre-smoothing, computes a coarse-grid correction,
    then ascends post-smoothing. One recursive call per level.

    Args:
        smoother: Smoother applied before and after the coarse-grid correction.
        n_pre: Pre-smoothing steps.
        n_post: Post-smoothing steps.

    References:
        - Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000).
          *A Multigrid Tutorial* (2nd ed.). SIAM. Algorithm 3.7, §3.2.
        - Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001).
          *Multigrid*. Academic Press. §2.2.1.
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

        # Coarsest level: direct dense solve (A stored as dense array)
        if level.transfer is None:
            return np.linalg.solve(A, rhs)

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


class WCycle:
    """W-cycle multigrid preconditioner (γ = 2).

    Applies two sequential coarse-grid corrections per level. After the first
    correction the fine-grid residual is recomputed and restricted again before
    the second coarse solve. This doubles the coarse-grid work relative to
    V-cycle and yields a more robust convergence factor, especially for
    anisotropic or indefinite-leaning problems.

    Args:
        smoother: Smoother applied before and after the coarse-grid corrections.
        n_pre: Pre-smoothing steps.
        n_post: Post-smoothing steps.

    References:
        - Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000).
          *A Multigrid Tutorial* (2nd ed.). SIAM. §3.3, γ-cycle definition.
        - Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001).
          *Multigrid*. Academic Press. §2.2.2, Algorithm 2.2 (μ = 2).
        - Stuben, K. (2001). A review of algebraic multigrid.
          *J. Comput. Appl. Math.*, 128(1–2), 281–309.
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
        """Compute one W-cycle starting from the finest level.

        Args:
            hierarchy: Pre-built multigrid hierarchy.
            rhs: Right-hand side vector on the finest grid.

        Returns:
            Approximate solution after one W-cycle (zero initial guess).
        """
        return self._wcycle(hierarchy, rhs, level_idx=0)

    def _wcycle(self, hierarchy: MultigridHierarchy, rhs: NDArray, level_idx: int) -> NDArray:
        level = hierarchy.levels[level_idx]
        A = level.matrix

        # Coarsest level: direct dense solve (A stored as dense array)
        if level.transfer is None:
            return np.linalg.solve(A, rhs)

        # Pre-smooth from zero initial guess
        x = self._smoother.smooth(A, rhs, np.zeros_like(rhs), self._n_pre)

        # γ = 2: two coarse-grid corrections, recomputing residual each time
        for _ in range(2):
            r_fine = rhs - A @ x
            r_coarse = level.transfer.restrict(r_fine)
            e_coarse = self._wcycle(hierarchy, r_coarse, level_idx + 1)
            x = x + level.transfer.prolongate(e_coarse)

        # Post-smooth
        return self._smoother.smooth(A, rhs, x, self._n_post)
