"""POD basis construction (snapshot method, via thin SVD)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from neuralls.domain.solver.preconditioners.tensor_utils import numpy_to_tensor, tensor_to_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _resolve_rank(singular_values: torch.Tensor, rank: int | float, max_rank: int) -> int:
    """Resolve a mode count from a fixed count or an energy-capture threshold.

    Args:
        singular_values: Singular values, descending (from ``torch.linalg.svd``).
        rank: Fixed mode count (int) or minimum cumulative captured energy
            (float in (0, 1], e.g. 0.9999 — matches Nikolopoulos et al. 2022,
            §4.1: "r=8 ... over 99.99% of the variance").
        max_rank: Upper bound (``min(n_samples, n_dofs)``).

    Returns:
        Resolved mode count, clamped to ``max_rank``.

    Raises:
        ValueError: If ``rank`` is a float outside (0, 1], or an int exceeding
            ``max_rank``.
    """
    if isinstance(rank, float):
        if not (0.0 < rank <= 1.0):
            raise ValueError(f"rank as an energy threshold must be in (0, 1], got {rank}")
        energy = singular_values.pow(2)
        cumulative = torch.cumsum(energy, dim=0) / energy.sum()
        resolved = int((cumulative < rank).sum().item()) + 1
        return min(resolved, max_rank)

    if rank > max_rank:
        raise ValueError(f"rank ({rank}) cannot exceed min(n_samples, n_dofs) ({max_rank})")
    return rank


def compute_pod_basis(
    snapshots: NDArray,
    rank: int | float,
    dtype: type | None = None,
) -> NDArray:
    """Build a truncated POD basis from a snapshot ensemble (snapshot method).

    Implements the reduced-basis construction of Nikolopoulos et al. (2022),
    §3.3, eq. 26-27, via the thin SVD of the snapshot matrix rather than
    eigendecomposing a Gram matrix: forming UUᵀ or UᵀU squares the condition
    number of U, which hurts precision precisely when POD is most useful
    (highly correlated/redundant snapshots). ``torch.linalg.svd`` computes the
    same POD modes without ever forming that Gram matrix, at the same
    asymptotic cost as the snapshot method for n_samples << n_dofs (the
    paper's own justification for avoiding the full n_dofs x n_dofs
    eigenproblem still holds), and returns singular values already sorted
    descending and non-negative — no manual sort or clamp needed. Runs on
    torch (GPU-ready, differentiable) but takes and returns plain NDArrays,
    matching the numpy boundary used elsewhere in ``domain/solver``.

    Args:
        snapshots: Solution snapshot ensemble, shape (n_samples, n_dofs) — one
            row per high-fidelity solution vector, matching the convention
            used by ``FileInputProvider``/``GeneratedSamples`` elsewhere in
            the repo.
        rank: Either a fixed number of POD modes to retain (int, largest
            singular values first), or a minimum fraction of cumulative
            captured energy to retain (float in (0, 1] — e.g. 0.9999 retains
            the smallest r with sum(σ_1²..σ_r²) / sum(σ_i²) >= rank, matching
            how the paper itself selects r).
        dtype: Numpy scalar type for the SVD (e.g. ``np.float32``). Defaults
            to ``snapshots.dtype`` — the caller's own precision is preserved
            unless an explicit override is given.

    Returns:
        POD basis Φ_r, shape (n_dofs, r), in the resolved dtype.

    Raises:
        ValueError: If ``rank`` is a float outside (0, 1], or an int
            exceeding ``min(n_samples, n_dofs)``.
    """
    n_samples, n_dofs = snapshots.shape
    max_rank = min(n_samples, n_dofs)

    resolved_dtype = dtype if dtype is not None else snapshots.dtype.type
    snapshots_t = numpy_to_tensor(snapshots, device="cpu", dtype=resolved_dtype)

    _, s, vh = torch.linalg.svd(snapshots_t, full_matrices=False)  # vh: (max_rank, n_dofs)
    resolved_rank = _resolve_rank(s, rank, max_rank)
    basis = vh[:resolved_rank].T  # (n_dofs, r), columns orthonormal by construction

    return tensor_to_numpy(basis, dtype=resolved_dtype)
