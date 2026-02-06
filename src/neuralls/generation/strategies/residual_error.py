"""Residual error strategy (cg_residual_error/residual_error)."""

from __future__ import annotations

import warnings

import numpy as np

from ..interfaces import GeneratedSamples, IDataGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import ResidualErrorConfig
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

        # config.samples represents number of base systems to generate
        # (orchestration layer handles counts_represent_final_pairs conversion)
        num_base_systems = config.samples
        cg_iters = config.cg_iters
        rng = np.random.default_rng(config.seed)

        n = matrix.shape[0]

        # Warn if using random generation (error traces are more meaningful with known solutions)
        if archive is None or archive.solutions is None:
            warnings.warn(
                "residual_error strategy is generating random solutions. "
                "Error traces are most useful when computed from known archive solutions. "
                "Consider passing archive_solutions to generate_mixture().",
                UserWarning,
                stacklevel=2,
            )

        # Generate base systems
        sols = _load_or_generate_solutions(num_base_systems, n, rng, 1.0, archive)
        rhs_samples = _load_or_compute_rhs(matrix, sols, archive)

        residual_blocks: list[np.ndarray] = []
        solution_current_blocks: list[np.ndarray] = []
        error_blocks: list[np.ndarray] = []
        sample_indices: list[np.ndarray] = []
        iteration_indices: list[np.ndarray] = []

        # Import scipy CG solver
        from ...solver.scipy_wrapper import SciPyCGSolver
        from ...solver.monitoring.iteration_history import IterationHistory
        from ...solver.monitoring.trace_mode import TraceMode

        for sample_idx, (rhs_vec, true_sol) in enumerate(zip(rhs_samples, sols)):
            # Run classical CG (scipy) for fixed number of iterations
            # Use scipy.sparse.linalg.cg wrapper for standard CG implementation
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

            # Extract vectors from iteration history
            assert info.iteration_history is not None and info.iteration_history.residuals is not None
            residual_seq = info.iteration_history.residuals.to_array()
            solution_seq = info.iteration_history.solutions.to_array() if info.iteration_history.solutions else np.array([])
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
