"""Residual error strategy (cg_residual_error/residual_error) using SOLID architecture.

Architecture:
    Layer 1: RandomInputProvider or ArchiveInputProvider → generate solutions
    Layer 2: ComputeRhsTransform → compute RHS = A @ x
    Layer 3: SciPyCGSolver → run CG with iteration history tracking
    Layer 4: Collect residual and error traces from iteration history

This strategy runs CG solver and collects both residual vectors and error vectors
at each iteration, useful for training preconditioners with true error information.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import ResidualErrorConfig
from ..helpers import _build_trace_indices, resolve_trace_generation_counts
from ..providers import provide_solutions, HybridInputProvider
from ..transforms import ComputeRhsTransform
from ...normalization import ErrorTraceSamples
from ..trace_utils import _referenced_sample_count, _trim_error_traces


@register_strategy
class ResidualErrorStrategy:
    name = "cg_residual_error"
    ConfigType = ResidualErrorConfig

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples with full residual error traces.

        Supports two modes:
        - Single RHS mode: If single_rhs provided, run CG multiple times on the same RHS
        - Multiple RHS mode: If single_rhs is None, generate N different RHS vectors

        Args:
            matrix: System matrix
            cfg: Configuration dictionary
            single_rhs: Optional single RHS vector. If provided, all samples solve A @ x = single_rhs
            archive: Optional archive data to seed generation

        Returns:
            GeneratedSamples with error_traces populated
        """

        # Validate and convert to typed config
        config = ResidualErrorConfig(**cfg)

        cg_iters = config.cg_iters
        rng = np.random.default_rng(config.seed)
        available_systems: int | None = None

        if single_rhs is None:
            if config.solutions_glob is not None:
                available_systems = len(
                    provide_solutions(
                        matrix,
                        -1,
                        rng,
                        solutions_glob=config.solutions_glob,
                        archive=archive,
                        shuffle=config.shuffle,
                        seed=config.seed,
                        strategy_name="cg_residual_error",
                    )
                )
            elif archive is not None and archive.solutions is not None:
                available_systems = int(archive.solutions.shape[0])

        num_base_systems, final_rows = resolve_trace_generation_counts(
            config.samples,
            cg_iters=cg_iters,
            every_n=config.every_n,
            available_systems=available_systems,
            strategy_name="cg_residual_error",
        )

        n = matrix.shape[0]

        # Choose mode based on single_rhs parameter
        if single_rhs is not None:
            # Mode 1: Single RHS - run CG multiple times on the SAME RHS
            # Note: Error traces require true solution, which we don't have in single RHS mode
            # So we'll solve once to get approximate solution, then use it as "true" for comparison
            from ...solver.scipy_wrapper import SciPyCGSolver

            # Solve once with high accuracy to get "true" solution
            solver = SciPyCGSolver()
            true_sol, _ = solver.solve(
                A=matrix,
                b=single_rhs,
                x0=np.zeros(n, dtype=np.float64),
                maxiter=10000,
                rtol=1e-12,
                atol=1e-12,
            )

            # Create array of identical RHS and solution vectors
            rhs_samples = np.tile(single_rhs, (num_base_systems, 1))
            sols = np.tile(true_sol, (num_base_systems, 1))
        else:
            # Mode 2: Multiple RHS - load solutions from glob, archive, or fail fast.
            sols = provide_solutions(
                matrix, num_base_systems, rng,
                solutions_glob=config.solutions_glob,
                archive=archive,
                shuffle=config.shuffle,
                seed=config.seed,
                strategy_name="cg_residual_error",
            )

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

            residual_seq = residual_seq[::config.every_n]
            if solution_seq.size > 0:
                solution_seq = solution_seq[::config.every_n]
            num_pairs = residual_seq.shape[0]

            error_seq = np.array(
                [true_sol - x_k for x_k in solution_seq], dtype=np.float64
            )

            residual_blocks.append(residual_seq)
            solution_current_blocks.append(solution_seq)
            error_blocks.append(error_seq)
            sidx, iidx = _build_trace_indices(num_pairs, sample_idx, every_n=config.every_n)
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
        error_traces = _trim_error_traces(error_traces, final_rows)
        referenced_samples = _referenced_sample_count(error_traces.sample_indices)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_samples[:referenced_samples],
            solutions=sols[:referenced_samples],
            error_traces=error_traces,
        )


@register_strategy
class ResidualErrorAliasStrategy(ResidualErrorStrategy):
    name = "residual_error"
