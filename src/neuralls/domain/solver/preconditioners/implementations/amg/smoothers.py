"""Multigrid smoothers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class JacobiSmoother:
    """Weighted Jacobi smoother: x ← x + ω D⁻¹ (rhs − Ax).

    Args:
        omega: Damping factor. 0.67 is standard for AMG with smoothed aggregation.
    """

    def __init__(self, omega: float = 0.67) -> None:
        self._omega = omega

    def smooth(self, A: NDArray, rhs: NDArray, x: NDArray, steps: int) -> NDArray:
        """Apply ``steps`` weighted Jacobi iterations.

        Args:
            A: System matrix (n × n).
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
