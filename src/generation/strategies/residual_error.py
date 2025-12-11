"""Residual error strategy (cg_residual_error/residual_error)."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IDataGenerationStrategy
from ..runner import register_strategy
from ..types import ArchiveData
from ...solver import SolverResult, flexible_cg
from ..helpers import (
    _load_or_generate_solutions,
    _load_or_compute_rhs,
    _build_trace_indices,
)
from ...normalization import ErrorTraceSamples


def _build_archive(
    solutions: np.ndarray,
    rhs: np.ndarray | None,
) -> ArchiveData:
    """Normalize archive inputs into ArchiveData."""
    sols = np.asarray(solutions, dtype=np.float64)
    rhs_vectors = None if rhs is None else np.asarray(rhs, dtype=np.float64)
    return ArchiveData(solutions=sols, rhs_vectors=rhs_vectors)


@register_strategy
class ResidualErrorStrategy(IDataGenerationStrategy):
    name = "cg_residual_error"

    def requires_rhs(self) -> bool:
        return True

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict,
    ) -> GeneratedSamples:
        if rhs is None:
            raise ValueError("cg_residual_error requires rhs input")
        count = int(cfg.get("samples", 0))
        cg_iters = int(cfg.get("residual_iters", 8))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        archive_solutions = cfg.get("archive_solutions")
        archive_rhs = cfg.get("archive_rhs")
        archive = None
        if archive_solutions is not None:
            archive = _build_archive(archive_solutions, archive_rhs)

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

            error_seq = np.array([true_sol - x_k for x_k in solution_seq], dtype=np.float64)

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
