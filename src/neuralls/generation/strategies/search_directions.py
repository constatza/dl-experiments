"""Search directions strategy for neural preconditioner training.

Collects (A @ p_k, p_k) pairs from CG iterations without requiring exact solutions.
Training mapping: NN(A @ p_k) ≈ p_k, meaning NN ≈ A^{-1}
"""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IDataGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import SearchDirectionsConfig
from ..helpers import (
    _load_or_generate_solutions,
    _load_or_compute_rhs,
    _build_trace_indices,
)
from ...normalization import SearchDirectionsSamples


@register_strategy
class SearchDirectionsStrategy(IDataGenerationStrategy):
    """Collect search direction pairs from CG for neural preconditioner training.

    This strategy generates training data for learning A^{-1} without requiring
    exact solutions. It collects (A @ p_k, p_k) pairs from CG iterations where:
    - p_k: search direction at iteration k
    - A @ p_k: search direction product (computed in CG anyway)

    The neural network learns: NN(A @ p_k) ≈ p_k, i.e., NN ≈ A^{-1}

    This is useful for:
    - Training preconditioners for a single matrix A (run CG from different u_0)
    - Training preconditioners for a family of matrices (run on multiple systems)
    - Learning inverse operators without expensive exact solves
    """

    name = "search_directions"
    ConfigType = SearchDirectionsConfig

    def requires_rhs(self) -> bool:
        return True

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate (A @ p_k, p_k) pairs from CG iterations.

        Args:
            matrix: System matrix
            rhs: Mother RHS vector (required)
            cfg: Configuration dictionary
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with search_directions_traces populated
        """
        if rhs is None:
            raise ValueError("search_directions requires rhs input")

        # Validate and convert to typed config
        config = SearchDirectionsConfig(**cfg)

        # config.samples represents number of base systems to generate
        num_base_systems = config.samples
        cg_iters = config.cg_iters
        rng = np.random.default_rng(config.seed)

        n = matrix.shape[0]
        # Generate base systems
        sols = _load_or_generate_solutions(num_base_systems, n, rng, 1.0, archive)
        rhs_samples = _load_or_compute_rhs(matrix, sols, archive)

        direction_blocks: list[np.ndarray] = []
        product_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        # Import scipy CG solver
        from ...solver.solvers.scipy_cg_solver import SciPyCGSolver
        from ...solver.monitoring.trace_recorder import TraceRecorder
        from ...solver.monitoring.events import EventType

        for sample_idx, rhs_vec in enumerate(rhs_samples):
            # Run classical CG (scipy) for fixed number of iterations
            event_log = TraceRecorder()
            solver = SciPyCGSolver(event_logger=event_log)

            _, info = solver.solve(
                A=matrix,
                b=rhs_vec,
                x0=np.zeros(n, dtype=np.float64),
                maxiter=cg_iters,
                rtol=1e-20,  # Very tight to prevent early convergence
                atol=1e-20,  # Very tight to run full iterations
                trace_mode="full",  # Collect all intermediate vectors
            )

            # Extract search directions and products from scipy CG event log
            assert info.event_log is not None
            direction_seq = info.event_log.get_vectors(EventType.SEARCH_DIRECTION)
            product_seq = info.event_log.get_vectors(EventType.SEARCH_DIRECTION_PRODUCT)

            num_pairs = direction_seq.shape[0]

            direction_blocks.append(direction_seq)
            product_blocks.append(product_seq)
            sidx, iidx = _build_trace_indices(num_pairs, sample_idx)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        # Build search directions traces: (A @ p_k, p_k) pairs
        search_directions_traces = SearchDirectionsSamples(
            search_direction_products=np.vstack(product_blocks),  # A @ p_k (inputs)
            search_directions=np.vstack(direction_blocks),  # p_k (targets)
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
        )

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_samples,
            solutions=sols,
            search_directions_traces=search_directions_traces,
        )
