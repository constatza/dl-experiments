"""Multigrid hierarchy data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .protocols import TransferOperator


@dataclass(frozen=True)
class MultigridLevel:
    """One level in a multigrid hierarchy.

    Attributes:
        matrix: System matrix on this grid level (n_k × n_k).
        transfer: Operator that moves vectors to the next coarser level.
            ``None`` on the coarsest level — direct solve is used there.
    """

    matrix: NDArray
    transfer: TransferOperator | None


@dataclass(frozen=True)
class MultigridHierarchy:
    """Full multigrid hierarchy built by ``AMGPreconditioner``.

    Attributes:
        levels: Tuple of levels from finest (index 0) to coarsest (index -1).
    """

    levels: tuple[MultigridLevel, ...]
