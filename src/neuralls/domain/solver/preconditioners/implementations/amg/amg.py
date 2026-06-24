"""AMG preconditioner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neuralls.domain.solver.preconditioners.base import (
    BindableInputs,
    Preconditioner,
    PreconditionerContext,
)

from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import CoarseningStrategy, MultigridCycle

if TYPE_CHECKING:
    pass


class AMGPreconditioner(Preconditioner):
    """Algebraic Multigrid preconditioner.

    Builds a multigrid hierarchy from a system matrix and applies a multigrid
    cycle as the preconditioner.  Structurally implements ``BindableInputs``
    (always safe to call ``bind_inputs``/``extra_input_names``); when the
    coarsening strategy has no extra inputs, these are no-ops.

    Hierarchy construction is **lazy**: it happens on the first ``apply()``
    call.  For classical coarsening (no ``bind_inputs`` needed), this is
    effectively immediate.  For neural coarsening, ``bind_inputs`` must be
    called before the first ``apply()``.

    Args:
        matrix: System matrix A (n × n).
        coarsening: Strategy that builds each coarse level.
        cycle: Multigrid cycle to apply as preconditioner.
        n_levels: Total number of levels (2 = one coarse grid).
        linear: Whether the preconditioner is linear (False for neural AMG).
            Controls ``requires_flexible_cg``.
    """

    def __init__(
        self,
        matrix: NDArray,
        coarsening: CoarseningStrategy,
        cycle: MultigridCycle,
        n_levels: int = 2,
        linear: bool = True,
    ) -> None:
        self._matrix = matrix
        self._coarsening = coarsening
        self._cycle = cycle
        self._n_levels = n_levels
        self._linear = linear
        self._hierarchy: MultigridHierarchy | None = None

    # BindableInputs structural implementation ---------------------------------

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Extra input names required by the coarsening strategy, if any."""
        if isinstance(self._coarsening, BindableInputs):
            return self._coarsening.extra_input_names
        return ()

    def bind_inputs(self, **inputs: NDArray) -> None:
        """Forward extra domain inputs to the coarsening strategy.

        Also invalidates the cached hierarchy so it is rebuilt on the next
        ``apply()`` call with the newly bound inputs.

        Args:
            **inputs: Named arrays (e.g., positions, θ).
        """
        if isinstance(self._coarsening, BindableInputs):
            self._coarsening.bind_inputs(**inputs)
        self._hierarchy = None

    # Preconditioner -----------------------------------------------------------

    @property
    def requires_flexible_cg(self) -> bool:
        """True for neural AMG (non-linear cycle); False for classical AMG."""
        return not self._linear

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply one V-cycle (or configured cycle) to the residual.

        Args:
            residual: Current residual vector r_k.
            context: Ignored (AMG cycle is stateless w.r.t. solver iteration).

        Returns:
            Approximate solution to Az = r from one multigrid cycle.
        """
        if self._hierarchy is None:
            self._hierarchy = self._build_hierarchy(self._matrix)
        return np.asarray(self._cycle.apply(self._hierarchy, residual), dtype=np.float64)

    # Private ------------------------------------------------------------------

    def _build_hierarchy(self, matrix: NDArray) -> MultigridHierarchy:
        """Iteratively apply coarsening to build all levels.

        Args:
            matrix: Finest-level system matrix.

        Returns:
            Frozen hierarchy of levels from fine to coarse.
        """
        levels: list[MultigridLevel] = []
        A_current = matrix
        for _ in range(self._n_levels - 1):
            A_coarse, transfer = self._coarsening.build_transfer(A_current)
            levels.append(MultigridLevel(matrix=A_current, transfer=transfer))
            A_current = A_coarse
        levels.append(MultigridLevel(matrix=A_current, transfer=None))
        return MultigridHierarchy(levels=tuple(levels))
