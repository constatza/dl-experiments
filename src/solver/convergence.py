"""Convergence criteria abstractions for iterative solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray
from scipy.linalg import norm


class IConvergenceCriterion(ABC):
    """Abstract base class for convergence checks."""

    @abstractmethod
    def threshold(self, rhs_norm: float) -> float:
        """Return the stopping threshold for a given RHS norm."""

    @abstractmethod
    def has_converged(self, residual: NDArray, rhs_norm: float) -> bool:
        """Determine convergence given residual and RHS norm."""


@dataclass(frozen=True)
class CombinedToleranceCriterion(IConvergenceCriterion):
    """SciPy-style criterion: ||r|| <= max(rtol * ||b||, atol)."""

    rtol: float
    atol: float

    def threshold(self, rhs_norm: float) -> float:
        return max(self.rtol * rhs_norm, self.atol)

    def has_converged(self, residual: NDArray, rhs_norm: float) -> bool:
        return bool(norm(residual) <= self.threshold(rhs_norm))
