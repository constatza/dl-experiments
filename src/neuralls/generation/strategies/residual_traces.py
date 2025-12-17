"""Residual trace strategy (cg_residual/residual)."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IDataGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import ResidualTraceConfig
from ...solver import SolverResult, flexible_cg
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

        count = config.samples
        cg_iters = config.residual_iters
        rng = np.random.default_rng(config.seed)
        
        n = matrix.shape[0]
        sols = _load_or_generate_solutions(count, n, rng, 1.0, archive)
        rhs_samples = _load_or_compute_rhs(matrix, sols, archive)

        residual_blocks: list[np.ndarray] = []
        solution_blocks: list[np.ndarray] = []
        search_direction_blocks: list[np.ndarray] = []
        search_direction_product_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        for sample_idx, rhs_vec in enumerate(rhs_samples):
            _, info_result = flexible_cg(
                matrix,
                rhs_vec,
                x0=np.zeros(n, dtype=np.float64),
                max_iter=cg_iters,
                preconditioner=None,
                stopping_criterion="fixed_iterations",
                tol=1e-12,
            )
            info = (
                SolverResult(**info_result)
                if isinstance(info_result, dict)
                else info_result
            )

            assert info.event_log is not None
            residual_list = info.event_log.get_history("residual")
            solution_list = info.event_log.get_history("solution")
            search_direction_list = info.event_log.get_history("search_direction")

            residual_seq = np.array(residual_list)
            solution_seq = np.array(solution_list)
            search_direction_seq = np.array(search_direction_list)
            search_direction_products_seq = np.array(
                [matrix @ search_direction for search_direction in search_direction_seq]
            )

            num_pairs = residual_seq.shape[0]

            residual_blocks.append(residual_seq)
            solution_blocks.append(solution_seq)
            search_direction_blocks.append(search_direction_seq)
            search_direction_product_blocks.append(search_direction_products_seq)
            sidx, iidx = _build_trace_indices(num_pairs, sample_idx)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        residual_traces = ResidualTraceSamples(
            residuals=np.vstack(residual_blocks),
            solutions=np.vstack(solution_blocks),
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
            search_directions=np.vstack(search_direction_blocks),
            search_direction_products=np.vstack(search_direction_product_blocks),
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
