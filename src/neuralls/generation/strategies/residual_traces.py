"""Residual trace strategy (cg_residual/residual)."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IDataGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import ResidualTraceConfig
from ..helpers import (
    _load_or_generate_solutions,
    _load_or_compute_rhs,
    _build_trace_indices,
)
from ...normalization import ResidualTraceSamples


@register_strategy
class ResidualTraceStrategy(IDataGenerationStrategy):
    name = "cg_residual"
    ConfigType = ResidualTraceConfig

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
        """Generate samples with residual traces (no true error).

        Args:
            matrix: System matrix
            rhs: Mother RHS vector (required)
            cfg: Configuration dictionary
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with residual_traces populated
        """
        if rhs is None:
            raise ValueError("cg_residual requires rhs input")

        # Validate and convert to typed config
        config = ResidualTraceConfig(**cfg)

        # config.samples represents number of base systems to generate
        # (orchestration layer handles counts_represent_final_pairs conversion)
        num_base_systems = config.samples
        cg_iters = config.cg_iters
        rng = np.random.default_rng(config.seed)

        n = matrix.shape[0]
        # Generate base systems
        sols = _load_or_generate_solutions(num_base_systems, n, rng, 1.0, archive)
        rhs_samples = _load_or_compute_rhs(matrix, sols, archive)

        residual_blocks: list[np.ndarray] = []
        solution_blocks: list[np.ndarray] = []
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

            # Extract vectors from scipy CG event log
            assert info.event_log is not None
            residual_seq = info.event_log.get_vectors(EventType.RESIDUAL)
            solution_seq = info.event_log.get_vectors(EventType.SOLUTION)

            num_pairs = residual_seq.shape[0]

            residual_blocks.append(residual_seq)
            solution_blocks.append(solution_seq)
            sidx, iidx = _build_trace_indices(num_pairs, sample_idx)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        # Note: scipy CG doesn't provide search directions
        # Search directions are optional and not needed for preconditioner training
        residual_traces = ResidualTraceSamples(
            residuals=np.vstack(residual_blocks),
            solutions=np.vstack(solution_blocks),
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
            search_directions=None,
            search_direction_products=None,
        )

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_samples,
            solutions=sols,
            residual_traces=residual_traces,
        )


# Alias
@register_strategy
class ResidualAliasStrategy(ResidualTraceStrategy):
    name = "residual"
