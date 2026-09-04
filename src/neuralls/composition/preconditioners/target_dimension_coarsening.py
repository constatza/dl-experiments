"""Cross-config-cached drop-in for torchalg's `TargetDimensionCoarsening`.

`TargetDimensionCoarsening._search` (torchalg) exhaustively scans a `theta`
grid, running a full `AggregationCoarsening.build_transfer` per candidate, to
find the realized coarse dimension closest to a target. A comparison run
commonly declares several `target_dim` AMG preconditioners against the same
matrix (e.g. `amg-dim100`/`amg-dim200`/`amg-dim300` in
`configs/cases/rectangular-high-condition/default.toml`) — each gets its own
`TargetDimensionCoarsening` instance, so the identical `theta` grid is
rescanned from scratch per config even though every candidate's
`build_transfer(A, theta, omega)` result only depends on the matrix and
`theta`/`omega`, not on the target dimension being searched for.

This module memoizes exactly that per-candidate computation with
`functools.lru_cache`, so the first `target_dim` config against a given
matrix populates the cache and every sibling config's search becomes cache
hits. `_search` is torchalg's documented extension point for swapping the
search algorithm without touching `build_transfer` or any caller, so this
subclasses `TargetDimensionCoarsening` and overrides only that method —
`isinstance` checks and the `_theta`/`_realized_coarse_dim`/
`_target_coarse_dim` attributes read by
`platform/reporting/preconditioner_labels.py` stay exactly as torchalg
defines them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torchalg.preconditioners.implementations.amg import TargetDimensionCoarsening

if TYPE_CHECKING:
    from torchalg.preconditioners.implementations.amg.transfer import DenseTransferOperator


@lru_cache(maxsize=1024)
def _cached_aggregation_build_transfer(
    matrix: torch.Tensor, theta: float, omega: float
) -> tuple[torch.Tensor, DenseTransferOperator]:
    """Build (and cache) one `AggregationCoarsening` candidate for `(matrix, theta, omega)`.

    `matrix` hashes and compares by object identity (the default for
    `torch.Tensor`), so this only dedupes calls sharing the exact matrix
    object already resident from one comparison run — never a false hit
    across two distinct, coincidentally-equal matrices.
    """
    from torchalg.preconditioners.implementations.amg import AggregationCoarsening

    return AggregationCoarsening(theta=theta, omega=omega).build_transfer(matrix)


class CachedTargetDimensionCoarsening(TargetDimensionCoarsening):
    """`TargetDimensionCoarsening` with its per-theta candidate builds cache-shared."""

    def _search(self, A: torch.Tensor) -> tuple[float, torch.Tensor, DenseTransferOperator]:
        """Same exhaustive scan as the base class, routed through the shared cache."""
        n_steps = round((self._theta_max - self._theta_min) / self._step) + 1
        candidates = [self._theta_min + i * self._step for i in range(n_steps)]
        results = [
            (theta, *_cached_aggregation_build_transfer(A, theta, self._omega))
            for theta in candidates
        ]
        return min(results, key=lambda result: abs(result[1].shape[0] - self._target_coarse_dim))
