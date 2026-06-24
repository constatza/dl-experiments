"""Coarsening strategies for AMG hierarchy construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, issparse

from .transfer import NeuralTransferOperator, SparseTransferOperator

if TYPE_CHECKING:
    from neuralls.domain.solver.preconditioners.ports import ExtraInputPredictorPort


def _to_csr(A: NDArray) -> csr_matrix:
    """Convert dense or sparse matrix to CSR format."""
    return cast(csr_matrix, A) if issparse(A) else csr_matrix(A)


def _strength_of_connection(A: csr_matrix, theta: float) -> csr_matrix:
    """Build binary strength-of-connection matrix.

    Entry (i, j) is strong if |a_ij| >= theta * sqrt(|a_ii| * |a_jj|).
    Diagonal entries are excluded.
    """
    diag_abs = np.abs(A.diagonal())
    diag_safe = np.where(diag_abs > 0, diag_abs, 1.0)
    S = A.copy().astype(float)
    rows, cols = S.nonzero()
    strengths = np.abs(S.data) / np.sqrt(diag_safe[rows] * diag_safe[cols])
    S.data = (strengths >= theta).astype(float)
    S.setdiag(0.0)
    S.eliminate_zeros()
    return S


def _greedy_aggregation(n: int, S: csr_matrix) -> NDArray:
    """Assign each node to an aggregate via greedy graph traversal.

    Returns an array of length n where entry i is the aggregate index of node i.
    """
    aggregate = np.full(n, -1, dtype=np.intp)
    n_agg = 0
    for i in range(n):
        if aggregate[i] >= 0:
            continue
        aggregate[i] = n_agg
        _, neighbors = S[i].nonzero()
        for j in neighbors:
            if aggregate[j] < 0:
                aggregate[j] = n_agg
        n_agg += 1
    # Any isolated nodes get their own aggregate
    for i in range(n):
        if aggregate[i] < 0:
            aggregate[i] = n_agg
            n_agg += 1
    return aggregate


def _piecewise_constant_prolongation(aggregate: NDArray) -> csr_matrix:
    """Build piecewise-constant prolongation P (n_fine × n_coarse)."""
    n = len(aggregate)
    n_coarse = int(aggregate.max()) + 1
    rows = np.arange(n, dtype=np.intp)
    return csr_matrix((np.ones(n), (rows, aggregate)), shape=(n, n_coarse))


def _smooth_prolongation(A: csr_matrix, P: csr_matrix, omega: float) -> csr_matrix:
    """Apply one Jacobi smoothing step to the tentative prolongation P₀.

    Computes P = (I − ω D⁻¹ A) P₀ (Vanek et al. 1996, Eq. 3.2). The damping
    factor ω should equal 4 / (3 · ρ(D⁻¹A)). For isotropic SPD problems
    ρ(D⁻¹A) ≈ 2, giving ω ≈ 2/3 ≈ 0.67 — the fixed default used here.
    Anisotropic problems may need ω computed via power iteration on D⁻¹A.

    Args:
        A: Fine-grid matrix in CSR format.
        P: Piecewise-constant tentative prolongation P₀ (n_fine × n_coarse).
        omega: Jacobi damping factor ω. Default 0.67 assumes ρ(D⁻¹A) ≈ 2.

    Returns:
        Smoothed prolongation matrix P = (I − ω D⁻¹ A) P₀.
    """
    diag = A.diagonal()
    diag_safe = np.where(np.abs(diag) > 1e-14, diag, 1.0)
    D_inv = diags(1.0 / diag_safe, format="csr")
    return P - omega * (D_inv @ A @ P)


class SparseAggregationCoarsening:
    """Smoothed aggregation AMG coarsening using scipy.sparse (SA-AMG).

    Implements the five-step coarsening algorithm from Vanek, Mandel & Brezina
    (1996):

    1. Strength-of-connection: mark strong off-diagonal entries via
       ``|a_ij| / sqrt(|a_ii| · |a_jj|) ≥ θ`` (Stüben 2001, §2.1).
    2. Greedy aggregation: partition nodes into non-overlapping aggregates
       using the strong-connection graph.
    3. Tentative prolongation P₀: piecewise-constant indicator matrix
       (aggregate membership).
    4. Smoothed prolongation: P = (I − ω D⁻¹ A) P₀, ω ≈ 2/3 for isotropic
       SPD (Vanek et al. 1996, Eq. 3.2).
    5. Galerkin coarse matrix: Aᶜ = Pᵀ A P.

    Args:
        theta: Strength-of-connection threshold θ ∈ (0, 1). Default 0.25
            follows Stüben (2001) §2.1.
        omega: Jacobi-smoothing damping ω ∈ (0, 1) for the prolongation
            smoother. Default 0.67 ≈ 2/3 assumes ρ(D⁻¹A) ≈ 2 (isotropic SPD).

    References:
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          *Computing*, 56(3), 179–196.
        - Stuben, K. (2001). A review of algebraic multigrid.
          *J. Comput. Appl. Math.*, 128(1–2), 281–309.
    """

    def __init__(self, theta: float = 0.25, omega: float = 0.67) -> None:
        self._theta = theta
        self._omega = omega

    def build_transfer(self, A: NDArray) -> tuple[NDArray, SparseTransferOperator]:
        """Build one coarse level from fine-grid matrix A.

        Args:
            A: Fine-grid matrix (n × n), dense or sparse.

        Returns:
            ``(A_coarse, transfer)`` where A_coarse is a dense (n_c × n_c) array.
        """
        A_csr = _to_csr(A)
        n = A_csr.shape[0]

        S = _strength_of_connection(A_csr, self._theta)
        aggregate = _greedy_aggregation(n, S)
        P0 = _piecewise_constant_prolongation(aggregate)
        P = _smooth_prolongation(A_csr, P0, self._omega)

        A_coarse = (P.T @ A_csr @ P).toarray()
        return A_coarse, SparseTransferOperator(P)


class NeuralCoarseningStrategy:
    """Coarsening with neural prolongation/restriction operators.

    Implements both ``CoarseningStrategy`` and ``BindableInputs`` so that
    static domain data (positions, θ) can be bound before the hierarchy
    is built.  Injected predictors are already loaded; this class does not
    load models (that is the factory's job).

    Args:
        prolongator: Neural predictor for coarse→fine mapping.
        restrictor: Neural predictor for fine→coarse mapping, or ``None``
            to use a placeholder (``build_transfer`` will raise).

    Note:
        ``build_transfer`` raises ``NotImplementedError`` until the neural
        operators are fully tested against a specific checkpoint format.
        The class structure is stable; only the forward pass needs wiring.
    """

    def __init__(
        self,
        prolongator: ExtraInputPredictorPort,
        restrictor: ExtraInputPredictorPort | None,
    ) -> None:
        self._prolongator = prolongator
        self._restrictor = restrictor
        self._bound: dict[str, NDArray] = {}

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Union of required inputs from both predictors, deduplicated."""
        names: list[str] = list(self._prolongator.required_inputs)
        if self._restrictor is not None:
            names.extend(self._restrictor.required_inputs)
        return tuple(dict.fromkeys(names))

    def bind_inputs(self, **inputs: NDArray) -> None:
        """Pre-bind domain data (positions, θ, …) before ``build_transfer``.

        Args:
            **inputs: Named arrays; only those in ``extra_input_names`` are kept.
        """
        self._bound = {k: v for k, v in inputs.items() if k in self.extra_input_names}

    def build_transfer(self, A: NDArray) -> tuple[NDArray, NeuralTransferOperator]:
        """Build neural transfer operator for this coarse level.

        Args:
            A: Fine-grid matrix (n × n).

        Returns:
            ``(A_coarse, transfer)`` — A_coarse uses Galerkin projection;
            transfer applies the neural P/R operators.

        Raises:
            NotImplementedError: Until the neural forward pass is wired to
                a specific checkpoint format and the coarse matrix computation
                is verified.
        """
        raise NotImplementedError(
            "NeuralCoarseningStrategy.build_transfer is not yet implemented. "
            "Wire the neural prolongator/restrictor to produce P, then compute "
            "A_coarse = P.T @ A @ P and return NeuralTransferOperator."
        )
