"""Orthogonalization strategies for Krylov methods (Single Responsibility).

This module provides orthogonalization strategies for maintaining conjugacy/orthogonality
in Krylov methods. The implementation is consolidated in this package's strategy layer.

Design:
    - OrthogonalizationStrategy: Abstract base for all orthogonalization methods
    - TruncatedGramSchmidt: FCG-style truncated Gram-Schmidt (window-based)
    - ModifiedGramSchmidt: More stable variant (full orthogonalization)
    - FullOrthogonalization: Orthogonalize against all previous directions

Theory (Notay 2000):
    Flexible CG maintains approximate A-conjugacy via Gram-Schmidt orthogonalization:
        d_i = w_i - Σ_{j=i-m}^{i-1} [(w_i, A d_j) / (d_j, A d_j)] d_j

    where:
        - w_i = M^{-1}(r_i) is the preconditioned residual
        - d_i is the search direction
        - A d_j is the matrix-vector product (q_j in code)

    The coefficient (w_i, A d_j) / (d_j, A d_j) enforces A-conjugacy:
        d_i^T A d_j ≈ 0

    This is NOT Euclidean orthogonality! A-conjugacy accelerates convergence
    for linear systems, while Euclidean orthogonality does not.

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...constants import DEFAULT_M_MAX, REORTHOG_ZERO_NORM_TOL

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class OrthogonalizationReport:
    """Report from orthogonalization operation.

    Attributes:
        coefficients: Orthogonalization coefficients [(w_i, q_j) / (d_j, q_j)] (Notay 2000).
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

    All strategies must implement:
    - orthogonalize(): Core orthogonalization algorithm
    - window_size property: Window size for direction history

    The window_size property enables uniform configuration across
    different orthogonalization strategies. It returns:
    - int: Maximum directions to retain (bounded window)
    - None: Unlimited orthogonalization (full Gram-Schmidt)

    Theory:
        Orthogonalization maintains conjugacy or orthogonality of search
        directions, preventing loss of linear independence and numerical
        breakdown.
    """

    @property
    @abstractmethod
    def window_size(self) -> int | None:
        """Return window size for direction history.

        Returns:
            int: Maximum number of directions to retain
            None: Unlimited (full orthogonalization)
        """
        ...

    @abstractmethod
    def orthogonalize(
        self,
        vector: NDArray,
        d_vectors: Sequence[NDArray],
        q_vectors: Sequence[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize vector against basis vectors.

        Args:
            vector: Vector to orthogonalize w_i = M^{-1} r_i (Notay 2000).
            d_vectors: Previous search directions [d_{i-m}, ..., d_{i-1}] (Notay 2000).
            q_vectors: Previous matrix-vector products [A d_{i-m}, ..., A d_{i-1}].

        Returns:
            Tuple (orthogonalized_vector, report):
            - orthogonalized_vector: Result of orthogonalization d_i (Notay 2000)
            - report: OrthogonalizationReport with diagnostics

        Theory (Notay 2000):
            For FCG, we compute:
                d_i = w_i - Σ_j [(w_i, A d_j) / (d_j, A d_j)] d_j

            This enforces approximate A-conjugacy: d_i^T A d_j ≈ 0
        """
        ...


class PeriodicRestartOrthogonalization(OrthogonalizationStrategy):
    """FCG(m_max) orthogonalization with periodic restart per Notay (2000).

    Implements the paper's recommended truncation strategy with cyclic restart:
        m_i = max(1, mod(i, m_max + 1))

    This creates a periodic pattern where the orthogonalization window
    resets to size 1 every (m_max + 1) iterations.

    Special case: Setting m_max=np.inf implements FCG(∞), which orthogonalizes
    against ALL previous directions without truncation or restart. This is
    equivalent to full reorthogonalization and is used for ill-conditioned
    problems requiring exact A-conjugacy.

    Attributes:
        m_max: Maximum window size before reset. Use np.inf for FCG(∞).
        epsilon: Threshold for small denominators.

    Theory (Notay 2000, Algorithm 2.1):
        The periodic restart formula m_i = max(1, i mod (m_max + 1)) produces:
        - For i < m_max: m_i = i (accumulate history)
        - For i = m_max: m_i = m_max (full window)
        - For i = m_max + 1: m_i = 1 (RESTART - discard old directions)
        - Repeats with period m_max + 1

        When m_max = ∞ (FCG(∞)): m_i = i (use all history, no restart)

        This periodic reset:
        1. Prevents accumulation of rounding errors
        2. Bounds memory at O(m_max * n)
        3. Is "generally more cost efficient" than pure truncation (Notay 2000, Table 1)

    Examples:
        >>> orthog = PeriodicRestartOrthogonalization(m_max=20)
        >>> # At iteration 11, will only use 1 direction (restart)
        >>> p, report = orthog.orthogonalize(z, p_vectors, q_vectors)

        >>> orthog_full = PeriodicRestartOrthogonalization(m_max=np.inf)  # FCG(∞)
        >>> # Always uses all history, never restarts
    """

    def __init__(self, m_max: float, epsilon: float = 1e-14):
        """Initialize FCG orthogonalization with periodic restart.

        Args:
            m_max: Maximum window size (m_max). Use np.inf for full orthogonalization (FCG(∞)).
                   At iteration m_max+1, resets to 1 (unless m_max is infinite).
            epsilon: Threshold for skipping small denominators.

        Raises:
            ValueError: If m_max < 1 (and not np.inf) or epsilon <= 0.
        """
        if not (m_max >= 1 or np.isinf(m_max)):
            raise ValueError(f"m_max must be >= 1 or np.inf, got {m_max}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        self.m_max = m_max
        self.epsilon = epsilon
        self._iteration_count = 0  # Track calls to compute m_i

    @property
    def window_size(self) -> int | None:
        """Return m_max as window size for direction history.

        Returns:
            int: Maximum number of directions (m_max from Notay 2000)
            None: Unlimited orthogonalization when m_max is np.inf (FCG(∞))
        """
        if np.isinf(self.m_max):
            return None
        return int(self.m_max)

    def orthogonalize(
        self,
        vector: NDArray,
        d_vectors: Sequence[NDArray],
        q_vectors: Sequence[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize via FCG with periodic restart (Notay 2000, Algorithm 2.1).

        Args:
            vector: Vector to orthogonalize w_i (Notay 2000).
            d_vectors: Previous search directions d_j (Notay 2000).
            q_vectors: Previous matrix-vector products A d_j.

        Returns:
            Tuple (orthogonalized_vector, report).

        Theory:
            Computes m_i = max(1, i mod (m_max + 1)) and orthogonalizes
            against the last m_i directions only.
        """
        result = np.asarray(vector, copy=True)

        n_history = len(d_vectors)

        if n_history == 0:
            self._iteration_count += 1
            return result, OrthogonalizationReport(coefficients=[], breakdown=False)

        # Handle infinity: use all history (no periodic restart)
        if np.isinf(self.m_max):
            m_i = n_history  # Use ALL history for FCG(∞)
        else:
            # Compute m_i using FCG periodic restart formula
            # m_i = max(1, i mod (m_max + 1))
            i = self._iteration_count
            m_i = max(1, i % (int(self.m_max) + 1))
            # Clamp to available history
            m_i = min(m_i, n_history)

        # Orthogonalize against last m_i directions
        coefficients: list[float] = []
        num_skipped = 0

        start_idx = n_history - m_i
        for j in range(start_idx, n_history):
            d_j = d_vectors[j]
            q_j = q_vectors[j]

            # Notay (2000) Algorithm 2.1: d_i = w_i - Σ[(w_i, A d_j)/(d_j, A d_j)] d_j
            # Classical Gram-Schmidt as per the summation in the original paper.
            numerator = float(np.dot(vector, q_j))  # (w_i, A d_j)
            denominator = float(np.dot(d_j, q_j))  # (d_j, A d_j)

            coeff = numerator / denominator
            coefficients.append(coeff)

            result -= coeff * d_j

        self._iteration_count += 1

        # Check for breakdown
        result_norm = float(np.linalg.norm(result))
        vector_norm = float(np.linalg.norm(vector))
        breakdown = result_norm < REORTHOG_ZERO_NORM_TOL * max(vector_norm, 1.0)

        return result, OrthogonalizationReport(
            coefficients=coefficients,
            breakdown=breakdown,
            num_skipped=num_skipped,
        )


class TruncatedGramSchmidt(OrthogonalizationStrategy):
    """Tr-FCG: Truncated Gram-Schmidt with sliding window (no restart).

    Orthogonalizes against a sliding window of previous directions to
    maintain approximate A-conjugacy while bounding memory and cost.

    Note: This implements Tr-FCG (pure truncation), NOT FCG.
    For FCG with periodic restart, use PeriodicRestartOrthogonalization.

    Attributes:
        window_size: Maximum number of directions to orthogonalize against.
        epsilon: Threshold for small denominators (default: 1e-14).

    Theory (Notay 2000):
        Tr-FCG formula:
            d_i = w_i - Σ_{j=i-m}^{i-1} [(w_i, A d_j) / (d_j, A d_j)] d_j

        where m = min(i, window_size) is the active window.
        Unlike FCG, this uses a sliding window without periodic resets.

    Example:
        >>> orthog = TruncatedGramSchmidt(window_size=10)
        >>> d, report = orthog.orthogonalize(w, d_vectors, q_vectors)
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

        self._window_size = window_size
        self.epsilon = epsilon

    @property
    def window_size(self) -> int:
        """Return configured window size for direction history.

        Returns:
            Maximum number of directions to retain
        """
        return self._window_size

    def orthogonalize(
        self,
        vector: NDArray,
        d_vectors: Sequence[NDArray],
        q_vectors: Sequence[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize via truncated Gram-Schmidt.

        Args:
            vector: Vector to orthogonalize w_i (Notay 2000).
            d_vectors: Previous search directions d_j (Notay 2000).
            q_vectors: Previous matrix-vector products A d_j.

        Returns:
            Tuple (orthogonalized_vector, report).

        Theory (Notay 2000):
            Orthogonalizes w_i against last min(m_max, len(d_vectors)) directions.
            Skips terms with small denominators to avoid numerical instability.
        """
        # Copy vector to avoid mutation
        result = np.asarray(vector, copy=True)

        # Determine orthogonalization window
        n_history = len(d_vectors)
        m = min(n_history, self._window_size)

        if m == 0:
            # No history to orthogonalize against
            return result, OrthogonalizationReport(coefficients=[], breakdown=False)

        # Orthogonalize against last m directions
        coefficients: list[float] = []
        num_skipped = 0

        for j in range(n_history - m, n_history):
            d_j = d_vectors[j]
            q_j = q_vectors[j]

            # Compute orthogonalization coefficient (Notay 2000)
            numerator = float(np.dot(vector, q_j))  # (w_i, A d_j)
            denominator = float(np.dot(d_j, q_j))  # (d_j, A d_j)

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

            # Orthogonalize: d_i -= coeff * d_j
            result -= coeff * d_j

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
    """Modified Gram-Schmidt orthogonalization (internal use).

    Modified Gram-Schmidt is more numerically stable than classical
    Gram-Schmidt, especially for nearly linearly dependent vectors.

    Note: This class is primarily for internal use. For FCG(∞) (full
    orthogonalization), use PeriodicRestartOrthogonalization(m_max=np.inf).

    Attributes:
        epsilon: Threshold for small denominators.

    Theory (Notay 2000):
        Modified GS updates the vector after each orthogonalization step:
            d_i^{(0)} = w_i
            For j = i-m, ..., i-1:
                coeff_j = (d_i^{(j)}, A d_j) / (d_j, A d_j)
                d_i^{(j+1)} = d_i^{(j)} - coeff_j * d_j
            d_i = d_i^{(m)}

        This differs from classical GS which computes all coefficients
        before updating, leading to better numerical stability.

    Example:
        >>> orthog = ModifiedGramSchmidt()
        >>> d, report = orthog.orthogonalize(w, d_vectors, q_vectors)
    """

    def __init__(self, epsilon: float = 1e-14):
        """Initialize modified Gram-Schmidt.

        Args:
            epsilon: Threshold for skipping small denominators.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = epsilon

    @property
    def window_size(self) -> None:
        """Return None for unlimited orthogonalization.

        Returns:
            None: Indicates full orthogonalization (no truncation)
        """
        return None

    def orthogonalize(
        self,
        vector: NDArray,
        d_vectors: Sequence[NDArray],
        q_vectors: Sequence[NDArray],
    ) -> tuple[NDArray, OrthogonalizationReport]:
        """Orthogonalize via modified Gram-Schmidt.

        Args:
            vector: Vector to orthogonalize w_i (Notay 2000).
            d_vectors: Previous search directions d_j (Notay 2000).
            q_vectors: Previous matrix-vector products A d_j.

        Returns:
            Tuple (orthogonalized_vector, report).
        """
        # Copy vector
        result = np.asarray(vector, copy=True)

        if len(d_vectors) == 0:
            return result, OrthogonalizationReport(coefficients=[], breakdown=False)

        coefficients: list[float] = []
        num_skipped = 0

        # Modified GS: update result after each step
        for d_j, q_j in zip(d_vectors, q_vectors):
            # Use updated result for numerator (key difference from classical GS)
            numerator = float(np.dot(result, q_j))
            denominator = float(np.dot(d_j, q_j))

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

            result -= coeff * d_j

        # Check breakdown
        result_norm = float(np.linalg.norm(result))
        vector_norm = float(np.linalg.norm(vector))
        breakdown = result_norm < REORTHOG_ZERO_NORM_TOL * max(vector_norm, 1.0)

        return result, OrthogonalizationReport(
            coefficients=coefficients,
            breakdown=breakdown,
            num_skipped=num_skipped,
        )


def create_fcg_orthogonalization(
    m_max: float | int = DEFAULT_M_MAX,
) -> OrthogonalizationStrategy:
    """Create FCG(m) orthogonalization strategy following Notay (2000).

    Factory function that creates the appropriate orthogonalization strategy
    based on the m_max parameter. This simplifies configuration by mapping
    a single parameter to the correct strategy class.

    Args:
        m_max: Window size for orthogonalization (FCG(m) from Notay 2000).
            - m_max > 0: FCG(m) with periodic restart
            - m_max = -1 or np.inf: FCG(∞) - full reorthogonalization

        Note: FCG(∞) means "infinite" orthogonalization window, orthogonalizing
        against ALL previous directions. The sentinel value m_max=-1 is converted
        to np.inf for backward compatibility.

    Returns:
        PeriodicRestartOrthogonalization: Configured strategy instance.

    Raises:
        ValueError: If m_max < -1 or m_max == 0.

    Examples:
        >>> orthog = create_fcg_orthogonalization(m_max=20)  # FCG(20)
        >>> isinstance(orthog, PeriodicRestartOrthogonalization)
        True
        >>> orthog.window_size
        10

        >>> orthog = create_fcg_orthogonalization(m_max=-1)  # FCG(∞) via sentinel
        >>> isinstance(orthog, PeriodicRestartOrthogonalization)
        True
        >>> orthog.window_size is None
        True

        >>> orthog = create_fcg_orthogonalization(m_max=np.inf)  # FCG(∞) explicit
        >>> orthog.window_size is None
        True

    References:
        Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    """
    if m_max == 0:
        raise ValueError("m_max cannot be 0. Use m_max=1 for minimal orthogonalization.")
    if m_max < -1 and not np.isinf(m_max):
        raise ValueError(f"m_max must be >= -1 or np.inf, got {m_max}")

    # Convert sentinel -1 to np.inf for backward compatibility
    if m_max == -1:
        m_max = np.inf

    return PeriodicRestartOrthogonalization(m_max=m_max)
