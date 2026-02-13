"""Search directions strategy using SOLID architecture.

Architecture:
    Layer 1: RandomInputProvider or ArchiveInputProvider → generate solutions
    Layer 2: ComputeRhsTransform → compute RHS = A @ x
    Layer 3: SciPyCGSolver → run CG with search direction tracking
    Layer 4: Collect (A @ p_k, p_k) pairs from iteration history

This strategy collects (A @ p_k, p_k) pairs from CG iterations without requiring
exact solutions. Training mapping: NN(A @ p_k) ≈ p_k, meaning NN ≈ A^{-1}
"""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import SearchDirectionsConfig
from ..helpers import _build_trace_indices
from ..providers import HybridInputProvider
from ..transforms import ComputeRhsTransform
from ...normalization import SearchDirectionsSamples


@register_strategy
class SearchDirectionsStrategy:
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

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate (A @ p_k, p_k) pairs from CG iterations.

        Supports two modes:
        - Single RHS mode: If single_rhs provided, run CG multiple times on the same RHS
        - Multiple RHS mode: If single_rhs is None, generate N different RHS vectors

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            single_rhs: Optional single RHS vector. If provided, all samples solve A @ x = single_rhs
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with search_directions_traces populated
        """

        # Validate and convert to typed config
        config = SearchDirectionsConfig(**cfg)

        # config.samples represents number of base systems to generate
        num_base_systems = config.samples
        cg_iters = config.cg_iters
        rng = np.random.default_rng(config.seed)

        n = matrix.shape[0]

        # Choose mode based on single_rhs parameter
        if single_rhs is not None:
            # Mode 1: Single RHS - run CG multiple times on the SAME RHS
            # Create array of identical RHS vectors for processing
            rhs_samples = np.tile(single_rhs, (num_base_systems, 1))
            # Note: We don't generate solutions in this mode (CG will solve from x0=0)
            sols = None
        else:
            # Mode 2: Multiple RHS - generate N different RHS vectors (SOLID pattern)
            # Layer 1: Input provision (archive with random fallback)
            solution_provider = HybridInputProvider(
                archive=archive, field="solutions", scale=1.0
            )
            sols = solution_provider.provide(matrix, count=num_base_systems, rng=rng)

            # Layer 2: Transform (compute RHS or load from archive)
            rhs_provider = HybridInputProvider(
                archive=archive, field="rhs_vectors", scale=1.0
            )
            rhs_from_archive = (
                archive is not None
                and archive.rhs_vectors is not None
                and archive.rhs_vectors.shape[0] >= num_base_systems
            )

            if rhs_from_archive:
                # Use RHS directly from archive
                rhs_samples = rhs_provider.provide(matrix, count=num_base_systems, rng=rng)
            else:
                # Compute RHS = A @ x
                transform = ComputeRhsTransform(matrix)
                rhs_samples = transform.transform(sols)

        direction_blocks: list[np.ndarray] = []
        product_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        # Import scipy CG solver
        from ...solver.scipy_wrapper import SciPyCGSolver
        from ...solver.monitoring.iteration_history import IterationHistory
        from ...solver.monitoring.trace_mode import TraceMode

        for sample_idx, rhs_vec in enumerate(rhs_samples):
            # Run classical CG (scipy) for fixed number of iterations
            iteration_history = IterationHistory(mode=TraceMode.FULL)
            solver = SciPyCGSolver(iteration_history=iteration_history)

            _, info = solver.solve(
                A=matrix,
                b=rhs_vec,
                x0=np.zeros(n, dtype=np.float64),
                maxiter=cg_iters,
                rtol=1e-20,  # Very tight to prevent early convergence
                atol=1e-20,  # Very tight to run full iterations
                trace_mode="full",  # Collect all intermediate vectors
            )

            # Extract search directions from iteration history
            # Guard: Check if directions are available
            if (
                info.iteration_history is None
                or info.iteration_history.directions is None
            ):
                raise RuntimeError(
                    "Search directions not available from scipy CG solver.\n"
                    "The scipy.sparse.linalg.cg implementation does not expose "
                    "search directions (p_k vectors). To use search direction collection:\n"
                    "  1. Implement a custom CG solver that tracks search directions, OR\n"
                    "  2. Use a different strategy (e.g., residual_trace) for neural preconditioner training.\n"
                    "\nTODO: Implement custom CG with search direction collection."
                )

            direction_seq = info.iteration_history.directions.to_array()

            # Guard: Check if directions array is empty (silent failure)
            if direction_seq.size == 0:
                raise RuntimeError(
                    f"Search directions array is empty for sample {sample_idx + 1}.\n"
                    "This indicates the scipy CG solver did not collect search directions.\n"
                    "See error message above for solutions."
                )

            direction_seq = direction_seq[::config.every_n]

            # Compute search direction products: A @ p_k
            # (scipy doesn't provide these, so we compute them)
            product_seq = np.array([matrix @ p for p in direction_seq], dtype=np.float64)

            num_pairs = direction_seq.shape[0]

            direction_blocks.append(direction_seq)
            product_blocks.append(product_seq)
            sidx, iidx = _build_trace_indices(num_pairs, sample_idx, every_n=config.every_n)
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
