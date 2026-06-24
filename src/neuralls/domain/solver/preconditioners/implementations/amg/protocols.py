"""AMG protocols — minimal extension points for multigrid components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .hierarchy import MultigridHierarchy


class TransferOperator(Protocol):
    """Moves vectors between a fine and its coarse grid.

    OCP hook: implement to add sparse, dense, or neural P/R operators.
    """

    def prolongate(self, coarse: NDArray) -> NDArray:
        """Map coarse-grid vector to fine grid (interpolation).

        Args:
            coarse: Coarse-grid vector of length n_coarse.

        Returns:
            Fine-grid vector of length n_fine.
        """
        ...

    def restrict(self, fine: NDArray) -> NDArray:
        """Map fine-grid vector to coarse grid (restriction).

        Args:
            fine: Fine-grid vector of length n_fine.

        Returns:
            Coarse-grid vector of length n_coarse.
        """
        ...


class MultigridSmoother(Protocol):
    """Damps high-frequency error on one grid level.

    OCP hook: implement to add Gauss-Seidel, polynomial, or other smoothers.
    """

    def smooth(self, A: NDArray, rhs: NDArray, x: NDArray, steps: int) -> NDArray:
        """Apply ``steps`` smoothing iterations to the system Ax = rhs.

        Args:
            A: System matrix on this level.
            rhs: Right-hand side vector.
            x: Current iterate (modified in-place semantics, but returns new array).
            steps: Number of smoothing steps.

        Returns:
            Updated iterate after smoothing.
        """
        ...


class CoarseningStrategy(Protocol):
    """Builds one coarse level from a fine-grid matrix.

    OCP hook: implement to add classical AMG, aggregation, or neural coarsening.
    Neural variants additionally implement ``BindableInputs`` so that spatial
    data (positions, parameters) can be bound before ``build_transfer`` is called.
    """

    def build_transfer(self, A: NDArray) -> tuple[NDArray, TransferOperator]:
        """Build a coarse grid and the associated transfer operator.

        The matrix A is the only argument because it is always available at
        hierarchy-construction time.  Neural strategies pre-bind any other
        needed arrays (positions, θ) via ``bind_inputs`` before this call.

        Args:
            A: Fine-grid matrix (n × n).

        Returns:
            ``(A_coarse, transfer)`` where ``A_coarse`` is the Galerkin coarse
            matrix (n_c × n_c) and ``transfer`` maps between the two grids.
        """
        ...


class MultigridCycle(Protocol):
    """Performs one multigrid correction cycle (V, W, F, …).

    OCP hook: implement to add W-cycle, F-cycle, or custom cycles.
    """

    def apply(self, hierarchy: MultigridHierarchy, rhs: NDArray) -> NDArray:
        """Compute an approximate solution to the system on the finest grid.

        Args:
            hierarchy: Pre-built multigrid hierarchy.
            rhs: Right-hand side vector on the finest grid.

        Returns:
            Approximate solution vector on the finest grid.
        """
        ...
