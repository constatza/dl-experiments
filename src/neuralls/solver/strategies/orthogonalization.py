"""Orthogonalization strategies for Krylov methods (Single Responsibility).

This module provides orthogonalization strategies for maintaining conjugacy/orthogonality
in Krylov methods. Previously scattered across fcg_base.py and direction_strategies.py,
now consolidated following the Strategy pattern.

Design:
    - OrthogonalizationStrategy: Abstract base for all orthogonalization methods
    - TruncatedGramSchmidt: FCG-style truncated Gram-Schmidt (window-based)
    - ModifiedGramSchmidt: More stable variant (full orthogonalization)
    - FullOrthogonalization: Orthogonalize against all previous directions

Theory:
    Flexible CG maintains approximate A-conjugacy via Gram-Schmidt orthogonalization:
        p_i = z_i - Σ_{k=i-m}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

    The coefficient (z_i, q_k) / (p_k, q_k) enforces A-conjugacy:
        p_i^T A p_k ≈ 0

    This is NOT Euclidean orthogonality! A-conjugacy accelerates convergence
    for linear systems, while Euclidean orthogonality does not.

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...constants import REORTHOG_ZERO_NORM_TOL

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class OrthogonalizationReport:
    """Report from orthogonalization operation.

    Attributes:
        coefficients: Orthogonalization coefficients [(z, q_k) / (p_k, q_k)].
        breakdown: Whether breakdown detected (near-linear dependence).
        num_skipped: Number of terms skipped (small denominators).

    Example:
        >>> report = OrthogonalizationReport(
        ...     coefficients=[0.5, 0.3, 0.1],
        ...     breakdown=False,
        ...     num_skipped=0,
        ... )
        >>> len(report.coefficients)
        3
    """

    coefficients: list[float]
    """Orthogonalization coefficients computed."""

    breakdown: bool = False
    """Whether breakdown detected (orthogonalized vector too small)."""

    num_skipped: int = 0
    """Number of orthogonalization terms skipped (small denominators)."""


class OrthogonalizationStrategy(ABC):
    """Abstract base for orthogonalization strategies.

    Defines the interface for all orthogonalization methods used in
    Krylov solvers. Subclasses implement specific algorithms.

    Theory:
        Orthogonalization maintains conjugacy or orthogonality of search
        directions, preventing loss of linear independence and numerical
        breakdown.
    """

    @abstractmethod
    def orthogonalize(
        self,
        vector: NDArray,
        p_vectors: list[NDArray],
        q_vectors: list[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize vector against basis vectors.

        Args:
            vector: Vector to orthogonalize (typically z_i = M^{-1} r_i).
            p_vectors: Previous search directions [p_{i-m}, ..., p_{i-1}].
            q_vectors: Previous matrix-vector products [q_{i-m}, ..., q_{i-1}].

        Returns:
            Tuple (orthogonalized_vector, report):
            - orthogonalized_vector: Result of orthogonalization
            - report: OrthogonalizationReport with diagnostics

        Theory:
            For FCG, we compute:
                p_i = z_i - Σ_k [(z_i, q_k) / (p_k, q_k)] p_k

            This enforces approximate A-conjugacy: p_i^T A p_k ≈ 0
        """
        ...


class TruncatedGramSchmidt(OrthogonalizationStrategy):
    """Truncated Gram-Schmidt orthogonalization for FCG.

    Orthogonalizes against a sliding window of previous directions to
    maintain approximate A-conjugacy while bounding memory and cost.

    Attributes:
        window_size: Maximum number of directions to orthogonalize against.
        epsilon: Threshold for small denominators (default: 1e-14).

    Theory:
        FCG formula:
            p_i = z_i - Σ_{k=i-m}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

        where m = min(i, window_size) is the active window.

        The coefficient (z_i, q_k) / (p_k, q_k) = (z_i, A p_k) / (p_k^T A p_k)
        is designed to enforce A-conjugacy: p_i^T A p_k ≈ 0

    Example:
        >>> orthog = TruncatedGramSchmidt(window_size=10)
        >>> p, report = orthog.orthogonalize(z, p_vectors, q_vectors)
        >>> report.breakdown
        False
    """

    def __init__(self, window_size: int, epsilon: float = 1e-14):
        """Initialize truncated Gram-Schmidt orthogonalization.

        Args:
            window_size: Maximum window size (m_max). Typical: 5-50.
            epsilon: Threshold for skipping small denominators.

        Raises:
            ValueError: If window_size < 1 or epsilon <= 0.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        self.window_size = window_size
        self.epsilon = epsilon

    def orthogonalize(
        self,
        vector: NDArray,
        p_vectors: list[NDArray],
        q_vectors: list[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize via truncated Gram-Schmidt.

        Args:
            vector: Vector to orthogonalize (z_i).
            p_vectors: Previous search directions.
            q_vectors: Previous matrix-vector products.

        Returns:
            Tuple (orthogonalized_vector, report).

        Theory:
            Orthogonalizes z_i against last min(m_max, len(p_vectors)) directions.
            Skips terms with small denominators to avoid numerical instability.
        """
        # Copy vector to avoid mutation
        result = np.asarray(vector, copy=True)

        # Determine orthogonalization window
        n_history = len(p_vectors)
        m = min(n_history, self.window_size)

        if m == 0:
            # No history to orthogonalize against
            return result, OrthogonalizationReport(coefficients=[], breakdown=False)

        # Orthogonalize against last m directions
        coefficients: list[float] = []
        num_skipped = 0

        for j in range(n_history - m, n_history):
            p_j = p_vectors[j]
            q_j = q_vectors[j]

            # Compute orthogonalization coefficient
            numerator = float(np.dot(vector, q_j))  # (z_i, q_k)
            denominator = float(np.dot(p_j, q_j))   # (p_k, q_k)

            # Check for NaN/Inf
            if not np.isfinite(numerator) or not np.isfinite(denominator):
                num_skipped += 1
                continue

            # Skip if denominator too small (avoid division by near-zero)
            if abs(denominator) < self.epsilon:
                num_skipped += 1
                continue

            # Compute coefficient and update
            coeff = numerator / denominator
            coefficients.append(coeff)

            if not np.isfinite(coeff):
                num_skipped += 1
                continue

            # Orthogonalize: p -= coeff * p_j
            result -= coeff * p_j

        # Check for breakdown (orthogonalized vector too small)
        result_norm = float(np.linalg.norm(result))
        vector_norm = float(np.linalg.norm(vector))
        breakdown = result_norm < REORTHOG_ZERO_NORM_TOL * max(vector_norm, 1.0)

        return result, OrthogonalizationReport(
            coefficients=coefficients,
            breakdown=breakdown,
            num_skipped=num_skipped,
        )


class ModifiedGramSchmidt(OrthogonalizationStrategy):
    """Modified Gram-Schmidt orthogonalization (more stable).

    Modified Gram-Schmidt is more numerically stable than classical
    Gram-Schmidt, especially for nearly linearly dependent vectors.

    Attributes:
        epsilon: Threshold for small denominators.

    Theory:
        Modified GS updates the vector after each orthogonalization step:
            p_i^{(0)} = z_i
            For k = i-m, ..., i-1:
                coeff_k = (p_i^{(k)}, q_k) / (p_k, q_k)
                p_i^{(k+1)} = p_i^{(k)} - coeff_k * p_k
            p_i = p_i^{(m)}

        This differs from classical GS which computes all coefficients
        before updating, leading to better numerical stability.

    Example:
        >>> orthog = ModifiedGramSchmidt()
        >>> p, report = orthog.orthogonalize(z, p_vectors, q_vectors)
    """

    def __init__(self, epsilon: float = 1e-14):
        """Initialize modified Gram-Schmidt.

        Args:
            epsilon: Threshold for skipping small denominators.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = epsilon

    def orthogonalize(
        self,
        vector: NDArray,
        p_vectors: list[NDArray],
        q_vectors: list[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize via modified Gram-Schmidt.

        Args:
            vector: Vector to orthogonalize.
            p_vectors: Previous search directions.
            q_vectors: Previous matrix-vector products.

        Returns:
            Tuple (orthogonalized_vector, report).
        """
        # Copy vector
        result = np.asarray(vector, copy=True)

        if len(p_vectors) == 0:
            return result, OrthogonalizationReport(coefficients=[], breakdown=False)

        coefficients: list[float] = []
        num_skipped = 0

        # Modified GS: update result after each step
        for p_j, q_j in zip(p_vectors, q_vectors):
            # Use updated result for numerator (key difference from classical GS)
            numerator = float(np.dot(result, q_j))
            denominator = float(np.dot(p_j, q_j))

            if not np.isfinite(numerator) or not np.isfinite(denominator):
                num_skipped += 1
                continue

            if abs(denominator) < self.epsilon:
                num_skipped += 1
                continue

            coeff = numerator / denominator
            coefficients.append(coeff)

            if not np.isfinite(coeff):
                num_skipped += 1
                continue

            result -= coeff * p_j

        # Check breakdown
        result_norm = float(np.linalg.norm(result))
        vector_norm = float(np.linalg.norm(vector))
        breakdown = result_norm < REORTHOG_ZERO_NORM_TOL * max(vector_norm, 1.0)

        return result, OrthogonalizationReport(
            coefficients=coefficients,
            breakdown=breakdown,
            num_skipped=num_skipped,
        )


class FullOrthogonalization(OrthogonalizationStrategy):
    """Full orthogonalization against all previous directions (no truncation).

    Orthogonalizes against all available history vectors. More accurate
    but higher memory (O(n * iterations)) and cost (O(n * iterations)).

    Use this for:
    - Short runs where memory is not a concern
    - Maximum accuracy requirements
    - Debugging truncated methods

    Attributes:
        epsilon: Threshold for small denominators.

    Example:
        >>> orthog = FullOrthogonalization()
        >>> p, report = orthog.orthogonalize(z, p_vectors, q_vectors)
    """

    def __init__(self, epsilon: float = 1e-14):
        """Initialize full orthogonalization.

        Args:
            epsilon: Threshold for skipping small denominators.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = epsilon

    def orthogonalize(
        self,
        vector: NDArray,
        p_vectors: list[NDArray],
        q_vectors: list[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize against all directions.

        Args:
            vector: Vector to orthogonalize.
            p_vectors: All previous search directions.
            q_vectors: All previous matrix-vector products.

        Returns:
            Tuple (orthogonalized_vector, report).
        """
        # Delegate to ModifiedGramSchmidt (no truncation)
        mgs = ModifiedGramSchmidt(epsilon=self.epsilon)
        return mgs.orthogonalize(vector, p_vectors, q_vectors)
