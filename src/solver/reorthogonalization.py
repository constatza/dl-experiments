"""Reorthogonalization strategies for Flexible PCG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from scipy.linalg import norm

from ..constants import (
    REORTHOG_MAX_COEFF,
    REORTHOG_ZERO_NORM_TOL,
)

# NOTE: Reorthogonalization helpers live here to keep a single source of truth
# across iterative solvers. Each solver can opt in via apply_reorthogonalization.


class ReorthogonalizationStrategy(ABC):
    """Base class for reorthogonalization strategies."""

    @abstractmethod
    def reorthogonalize(
        self,
        vector: np.ndarray,
        basis_vectors: list[np.ndarray],
        iteration: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reorthogonalize vector against basis vectors."""
        raise NotImplementedError


class FullReorthogonalization(ReorthogonalizationStrategy):
    """Full A-conjugate reorthogonalization against all previous search directions."""

    def __init__(self, A: np.ndarray | None = None):
        """Initialize with system matrix for A-conjugate reorthogonalization."""
        self.A = A

    def reorthogonalize(
        self,
        vector: np.ndarray,
        basis_vectors: list[np.ndarray],
        iteration: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        v = vector.copy()
        v_norm_initial = norm(v)
        if v_norm_initial < REORTHOG_ZERO_NORM_TOL:
            return v, {
                "breakdown": True,
                "reason": "near-zero input norm",
                "norm_initial": v_norm_initial,
            }

        coeffs: list[float] = []

        for i, basis_vec in enumerate(basis_vectors):
            if self.A is not None:
                Ap = self.A.dot(basis_vec)
                pAp = np.dot(basis_vec, Ap)
                if abs(pAp) < REORTHOG_ZERO_NORM_TOL:
                    return v, {
                        "breakdown": True,
                        "reason": "near-zero A-norm basis vector",
                        "pAp": float(pAp),
                        "basis_index": i,
                    }
                coeff = np.dot(v, Ap) / pAp
            else:
                basis_norm_sq = np.dot(basis_vec, basis_vec)
                if basis_norm_sq < REORTHOG_ZERO_NORM_TOL:
                    return v, {
                        "breakdown": True,
                        "reason": "near-zero basis norm",
                        "basis_norm_sq": float(basis_norm_sq),
                        "basis_index": i,
                    }
                coeff = np.dot(v, basis_vec) / basis_norm_sq

            if not np.isfinite(coeff):
                return v, {
                    "breakdown": True,
                    "reason": "non-finite coeff",
                    "coeff": float(coeff),
                    "basis_index": i,
                }
            if abs(coeff) > REORTHOG_MAX_COEFF:
                return v, {
                    "breakdown": True,
                    "reason": "coeff too large",
                    "coeff": float(coeff),
                    "basis_index": i,
                }

            v = v - coeff * basis_vec
            coeffs.append(float(coeff))

        v_norm_final = norm(v)
        if v_norm_final < REORTHOG_ZERO_NORM_TOL * v_norm_initial:
            return v, {
                "breakdown": True,
                "reason": "near-zero output norm",
                "norm_initial": v_norm_initial,
                "norm_final": v_norm_final,
            }

        return v, {
            "reorthog_count": len(coeffs),
            "reorthog_coeffs": coeffs,
        }


class PartialReorthogonalization(ReorthogonalizationStrategy):
    """Reorthogonalize against a sliding window of search directions."""

    def __init__(self, window_size: int | None = 10):
        """Initialize with window size (None uses all previous directions)."""
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive or None")
        self.window_size = window_size

    def reorthogonalize(
        self,
        vector: np.ndarray,
        basis_vectors: list[np.ndarray],
        iteration: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        v = vector.copy()
        info: dict[str, Any] = {
            "reorthog_count": 0,
            "iteration": iteration,
            "window_size": 0,
        }

        basis = (
            basis_vectors
            if self.window_size is None
            else basis_vectors[-self.window_size :]
        )
        info["window_size"] = len(basis)

        for basis_vec in basis:
            norm_basis = norm(basis_vec)
            if norm_basis < REORTHOG_ZERO_NORM_TOL:
                raise ValueError("basis vector has near-zero norm")
            coeff = np.dot(basis_vec, v) / (norm_basis**2)
            v = v - coeff * basis_vec
            info["reorthog_count"] += 1

        return v, info


class SelectiveReorthogonalization(ReorthogonalizationStrategy):
    """Reorthogonalize when deviation from orthogonality exceeds threshold."""

    def __init__(self, threshold: float = 0.7, A: np.ndarray | None = None):
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold
        self.A = A

    def reorthogonalize(
        self,
        vector: np.ndarray,
        basis_vectors: list[np.ndarray],
        iteration: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        v = vector.copy()
        if self.A is not None:
            Av = self.A.dot(v)
            v_norm_sq = float(np.dot(v, Av))
            v_norm = float(np.sqrt(abs(v_norm_sq)))
        else:
            v_norm = norm(v)
            v_norm_sq = v_norm * v_norm
        if v_norm == 0:
            return v, {
                "reorthog_count": 0,
                "reorthog_coeffs": [],
                "triggered": False,
                "threshold": self.threshold,
            }

        coeffs: list[float] = []
        needs_reorthog = False

        for basis_vec in basis_vectors:
            if self.A is not None:
                Ap = self.A.dot(basis_vec)
                pAp = float(np.dot(basis_vec, Ap))
                if pAp <= 0.0:
                    continue
                basis_norm = float(np.sqrt(abs(pAp)))
                inner_prod = abs(np.dot(v, Ap)) / max(v_norm * basis_norm, 1e-16)
            else:
                basis_norm = norm(basis_vec)
                if basis_norm == 0:
                    continue
                inner_prod = abs(np.dot(v, basis_vec)) / (v_norm * basis_norm)
            if inner_prod > self.threshold:
                needs_reorthog = True
                break

        if needs_reorthog:
            for basis_vec in basis_vectors:
                if self.A is not None:
                    Ap = self.A.dot(basis_vec)
                    pAp = float(np.dot(basis_vec, Ap))
                    if abs(pAp) < 1e-14:
                        continue
                    coeff = np.dot(v, Ap) / pAp
                else:
                    basis_norm_sq = np.dot(basis_vec, basis_vec)
                    if basis_norm_sq < 1e-14:
                        continue
                    coeff = np.dot(v, basis_vec) / basis_norm_sq
                if np.isfinite(coeff):
                    v = v - coeff * basis_vec
                    coeffs.append(float(coeff))

        return v, {
            "reorthog_count": len(coeffs),
            "reorthog_coeffs": coeffs,
            "triggered": needs_reorthog,
            "threshold": self.threshold,
        }


def create_reorthogonalization_strategy(
    strategy: Literal["none", "full", "partial", "selective"] | None,
    **kwargs: Any,
) -> ReorthogonalizationStrategy | None:
    """Factory for reorthogonalization strategies."""
    if strategy is None or strategy == "none":
        return None
    if strategy == "full":
        A = kwargs.get("A", None)
        return FullReorthogonalization(A=A)
    if strategy == "partial":
        window_size = kwargs.get("window_size", 10)
        return PartialReorthogonalization(window_size=window_size)
    if strategy == "selective":
        threshold = kwargs.get("threshold", 0.7)
        A = kwargs.get("A", None)
        return SelectiveReorthogonalization(threshold=threshold, A=A)
    raise ValueError(f"Unknown reorthogonalization strategy: {strategy}")


def apply_reorthogonalization(
    strategy: ReorthogonalizationStrategy | None,
    vector: np.ndarray,
    basis_vectors: list[np.ndarray],
    iteration: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply reorthogonalization if a strategy is provided."""
    if strategy is None or not basis_vectors:
        return vector, {"skipped": True, "reorthog_count": 0}
    return strategy.reorthogonalize(vector, basis_vectors, iteration)
