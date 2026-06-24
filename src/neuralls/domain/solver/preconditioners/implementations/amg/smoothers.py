"""Multigrid smoothers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SmootherBase(ABC):
    """Base class for fixed-step iterative error dampers used in multigrid cycles.

    Smoothers are stationary iterative methods applied for a fixed number of steps
    rather than to convergence. Their role is to damp high-frequency error components
    on a single grid level, complementing the coarse-grid correction. They satisfy
    the ``MultigridSmoother`` protocol structurally (parallel nominative/structural
    hierarchies; no import dependency between them).

    Note:
        This ABC and ``MultigridSmoother`` (Protocol) are intentionally separate:
        the Protocol provides structural typing for cycle extension points; this ABC
        provides the concrete implementation hierarchy for shared documentation and
        type-safe subclassing.
    """

    @abstractmethod
    def smooth(self, A: NDArray, rhs: NDArray, x: NDArray, steps: int) -> NDArray:
        """Apply ``steps`` smoothing iterations to the system Ax = rhs.

        Args:
            A: System matrix on this level (n × n).
            rhs: Right-hand side vector (n,).
            x: Current iterate (n,).
            steps: Number of sweeps (fixed, not convergence-driven).

        Returns:
            Updated iterate after ``steps`` sweeps, same shape as ``x``.
        """


class JacobiSmoother(SmootherBase):
    """Weighted Jacobi smoother: x ← x + ω D⁻¹ (rhs − Ax).

    Applies ``steps`` stationary Jacobi iterations with damping ω. For smoothed
    aggregation AMG, ω ≈ 2/3 is optimal: it annihilates the high-frequency error
    components that the coarse-grid correction cannot reach.

    Args:
        omega: Damping factor ω ∈ (0, 1). Default 0.67 ≈ 2/3 is the standard
            choice for isotropic SPD problems where ρ(D⁻¹A) ≈ 2 (see References).

    References:
        - Young, D. M. (1954). Iterative methods for solving partial difference
          equations of elliptic type. *Trans. Amer. Math. Soc.*, 76, 92–111.
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          *Computing*, 56(3), 179–196. §3: ω = 4 / (3 · ρ(D⁻¹A)).
    """

    def __init__(self, omega: float = 0.67) -> None:
        self._omega = omega

    def smooth(self, A: NDArray, rhs: NDArray, x: NDArray, steps: int) -> NDArray:
        """Apply ``steps`` weighted Jacobi iterations.

        Args:
            A: System matrix (n × n), dense.
            rhs: Right-hand side vector (n,).
            x: Current iterate (n,).
            steps: Number of sweeps.

        Returns:
            Updated iterate after ``steps`` sweeps.
        """
        diag = np.diag(A)
        diag_inv = np.where(np.abs(diag) > 1e-14, self._omega / diag, 0.0)
        x = x.copy()
        for _ in range(steps):
            x += diag_inv * (rhs - A @ x)
        return x
