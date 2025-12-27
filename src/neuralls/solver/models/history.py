"""History tracking for iterative solvers.

This module defines immutable history dataclasses for tracking algorithm state:
- DirectionHistory: Search directions and matrix-vector products (for FCG)
- ResidualHistory: Residual norms (absolute and relative)

All classes use frozen dataclasses for immutability, following functional programming principles.

Theory:
    Krylov methods like FCG require maintaining a window of search directions
    for orthogonalization. Using immutable tuples ensures no accidental mutation
    and simplifies reasoning about algorithm correctness.

Design:
    - Immutable (frozen=True) for thread safety and predictable behavior
    - Tuple-based storage (immutable sequences)
    - Factory methods for creating empty instances
    - Update methods return new instances (functional style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class DirectionHistory:
    """Immutable history of search directions and matrix-vector products.

    Maintains a sliding window of search directions (p_k) and their corresponding
    matrix-vector products (q_k = A p_k) for algorithms like Flexible Conjugate
    Gradient that require orthogonalization against previous directions.

    Theory:
        FCG orthogonalization formula:
            p_i = z_i - Σ_{k=i-m}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

        This requires storing the last m directions and products, where m is
        the orthogonalization window size.

    Attributes:
        p_vectors: Tuple of previous search directions [p_{i-m}, ..., p_{i-1}].
            Immutable tuple ensures no accidental modification.

        q_vectors: Tuple of previous matrix-vector products [q_{i-m}, ..., q_{i-1}].
            Each q_k = A @ p_k. Used in orthogonalization coefficients.

        max_size: Maximum window size. Vectors are truncated to keep only last max_size.

        total_updates: Total number of directions added (monotonic, not truncated).
            Used for debugging and statistics.

    Example:
        >>> history = DirectionHistory.empty(max_size=10)
        >>> history = history.add(p, q)
        >>> history = history.add(p2, q2)
        >>> len(history.p_vectors)  # 2
    """

    p_vectors: tuple[NDArray, ...] = ()
    """Previous search directions (immutable tuple)."""

    q_vectors: tuple[NDArray, ...] = ()
    """Previous matrix-vector products q_k = A @ p_k (immutable tuple)."""

    max_size: int = 10
    """Maximum window size for truncation."""

    total_updates: int = 0
    """Total number of directions ever added (monotonic)."""

    @classmethod
    def empty(cls, max_size: int = 10) -> DirectionHistory:
        """Create empty history with specified maximum size.

        Args:
            max_size: Maximum number of directions to store. Defaults to 10.

        Returns:
            New DirectionHistory with no vectors.

        Example:
            >>> history = DirectionHistory.empty(max_size=20)
            >>> history.p_vectors
            ()
        """
        return cls(p_vectors=(), q_vectors=(), max_size=max_size, total_updates=0)

    def add(self, p: NDArray, q: NDArray) -> DirectionHistory:
        """Return new history with vectors added (immutable update).

        Args:
            p: New search direction to add.
            q: New matrix-vector product (A @ p) to add.

        Returns:
            New DirectionHistory with vector added and old vectors possibly truncated.

        Theory:
            Truncation maintains only the last max_size vectors to:
            1. Bound memory usage at O(max_size * n)
            2. Limit orthogonalization cost to O(max_size * n) per iteration
            3. Prevent numerical error accumulation

        Example:
            >>> history = DirectionHistory.empty(max_size=2)
            >>> history = history.add(p1, q1)
            >>> history = history.add(p2, q2)
            >>> history = history.add(p3, q3)  # Drops p1, q1
            >>> len(history.p_vectors)  # 2
        """
        import numpy as np

        # Append new vectors (create new tuples)
        new_p = self.p_vectors + (np.asarray(p, copy=True),)
        new_q = self.q_vectors + (np.asarray(q, copy=True),)

        # Truncate to max_size (keep last max_size)
        if len(new_p) > self.max_size:
            new_p = new_p[-self.max_size :]
        if len(new_q) > self.max_size:
            new_q = new_q[-self.max_size :]

        return DirectionHistory(
            p_vectors=new_p,
            q_vectors=new_q,
            max_size=self.max_size,
            total_updates=self.total_updates + 1,
        )

    def __len__(self) -> int:
        """Return number of directions stored (not total_updates).

        Returns:
            Current window size (len(p_vectors)).
        """
        return len(self.p_vectors)


@dataclass(frozen=True)
class ResidualHistory:
    """Immutable history of residual norms.

    Tracks absolute and relative residual norms across iterations for
    convergence monitoring and post-analysis.

    Attributes:
        norms_abs: Tuple of absolute residual norms ||r_k||_2.
        norms_rel: Tuple of relative residual norms ||r_k||_2 / ||b||_2.

    Example:
        >>> history = ResidualHistory.empty()
        >>> history = history.add(norm_abs=1.0, norm_rel=0.1)
        >>> history = history.add(norm_abs=0.5, norm_rel=0.05)
        >>> history.norms_abs
        (1.0, 0.5)
    """

    norms_abs: tuple[float, ...] = ()
    """Absolute residual norms ||r_k||_2 (immutable tuple)."""

    norms_rel: tuple[float, ...] = ()
    """Relative residual norms ||r_k||_2 / ||b||_2 (immutable tuple)."""

    @classmethod
    def empty(cls) -> ResidualHistory:
        """Create empty residual history.

        Returns:
            New ResidualHistory with no norms.

        Example:
            >>> history = ResidualHistory.empty()
            >>> history.norms_abs
            ()
        """
        return cls(norms_abs=(), norms_rel=())

    def add(self, norm_abs: float, norm_rel: float) -> ResidualHistory:
        """Return new history with norms added (immutable update).

        Args:
            norm_abs: Absolute residual norm ||r_k||_2.
            norm_rel: Relative residual norm ||r_k||_2 / ||b||_2.

        Returns:
            New ResidualHistory with norms appended.

        Example:
            >>> history = ResidualHistory.empty()
            >>> history = history.add(1.0, 0.1)
            >>> history.norms_abs[-1]
            1.0
        """
        return ResidualHistory(
            norms_abs=self.norms_abs + (float(norm_abs),),
            norms_rel=self.norms_rel + (float(norm_rel),),
        )

    def __len__(self) -> int:
        """Return number of residual norms stored.

        Returns:
            Length of norms_abs (same as norms_rel).
        """
        return len(self.norms_abs)
