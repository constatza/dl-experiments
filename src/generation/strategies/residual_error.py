"""Residual error strategy (cg_residual_error/residual_error)."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IDataGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import ResidualErrorConfig
from ...solver import SolverResult, flexible_cg
from ..helpers import (
    _load_or_generate_solutions,
    _load_or_compute_rhs,
    _build_trace_indices,
)
from ...normalization import ErrorTraceSamples


@register_strategy
class ResidualErrorStrategy(IDataGenerationStrategy):
    name = "cg_residual_error"
    ConfigType = ResidualErrorConfig

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
        """Generate samples with full residual error traces.

        Args:
            matrix: System matrix
            rhs: Mother RHS vector (required)
            cfg: Configuration dictionary
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with error_traces populated
        """
        if rhs is None:
            raise ValueError("cg_residual_error requires rhs input")

        # Validate and convert to typed config
        config = ResidualErrorConfig(**cfg)

        count = config.samples
        cg_iters = config.residual_iters
        rng = np.random.default_rng(config.seed)
        
        # Ensure archive is available if requested?
        # For now, just use what is passed.
        
        n = matrix.shape[0]
        sols = _load_or_generate_solutions(count, n, rng, 1.0, archive)
        rhs_samples = _load_or_compute_rhs(matrix, sols, archive)

        residual_blocks: list[np.ndarray] = []
        solution_current_blocks: list[np.ndarray] = []
        error_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        for sample_idx, (rhs_vec, true_sol) in enumerate(zip(rhs_samples, sols)):
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

            residual_seq = np.array(residual_list)
            solution_seq = np.array(solution_list)
            num_pairs = residual_seq.shape[0]

            error_seq = np.array(
                [true_sol - x_k for x_k in solution_seq], dtype=np.float64
            )

            residual_blocks.append(residual_seq)
            solution_current_blocks.append(solution_seq)
            error_blocks.append(error_seq)
            sidx, iidx = _build_trace_indices(num_pairs, sample_idx)
            sample_indices.append(sidx)
            iteration_indices.append(iidx)

        error_traces = ErrorTraceSamples(
            residuals=np.vstack(residual_blocks),
            solutions_current=np.vstack(solution_current_blocks),
            errors=np.vstack(error_blocks),
            true_solutions=sols,
            sample_indices=np.concatenate(sample_indices),
            iteration_indices=np.concatenate(iteration_indices),
        )

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_samples,
            solutions=sols,
            error_traces=error_traces,
        )


@register_strategy
class ResidualErrorAliasStrategy(ResidualErrorStrategy):
    name = "residual_error"
