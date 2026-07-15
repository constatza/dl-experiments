"""Residual trace strategies with error targets.

Architecture:
    Layer 1: Archive/File or Gaussian providers → generate true solutions
    Layer 2: ComputeRhsTransform → compute RHS = A @ x
    Layer 3: SciPyCGSolver → run CG with iteration history tracking
    Layer 4: Collect residual and error traces from iteration history

These strategies run CG and collect residual vectors together with exact error
targets dx_k = x_true - x_k at each iteration.
"""

from __future__ import annotations

import numpy as np
import torch

from ..interfaces import GeneratedSamples, ArchiveData, TracingSolverCallable
from ..runner import register_single_rhs_strategy
from ..strategy_configs import ResidualErrorConfig
from ..helpers import _build_trace_indices, resolve_trace_generation_counts
from ..providers import HybridInputProvider, RandomInputProvider, provide_solutions
from ..transforms import ComputeRhsTransform
from neuralls.domain.normalization import ErrorTraceSamples
from ..trace_utils import _referenced_sample_count, _trim_error_traces


def _tensor_trace_to_numpy(value: torch.Tensor | None) -> np.ndarray:
    if value is None:
        return np.array([])
    return value.detach().cpu().numpy()


class _BaseResidualsStrategy:
    ConfigType = ResidualErrorConfig
    name: str

    def _resolve_available_systems(
        self,
        matrix: np.ndarray,
        config: ResidualErrorConfig,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> int | None:
        if config.solutions_glob is not None:
            return len(
                provide_solutions(
                    matrix,
                    -1,
                    rng,
                    solutions_glob=config.solutions_glob,
                    archive=archive,
                    shuffle=config.shuffle,
                    seed=config.seed,
                    strategy_name=self.name,
                )
            )
        if archive is not None and archive.solutions is not None:
            return int(archive.solutions.shape[0])
        return None

    def _provide_true_solutions(
        self,
        matrix: np.ndarray,
        count: int,
        config: ResidualErrorConfig,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return provide_solutions(
            matrix,
            count,
            rng,
            solutions_glob=config.solutions_glob,
            archive=archive,
            shuffle=config.shuffle,
            seed=config.seed,
            strategy_name=self.name,
        )

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict,
        solver: TracingSolverCallable,
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
            available_systems = self._resolve_available_systems(matrix, config, archive, rng)

        num_base_systems, final_rows = resolve_trace_generation_counts(
            config.samples,
            cg_iters=cg_iters,
            every_n=config.every_n,
            available_systems=available_systems,
            strategy_name=self.name,
        )

        n = matrix.shape[0]

        # Choose mode based on single_rhs parameter
        if single_rhs is not None:
            # Mode 1: Single RHS - run CG multiple times on the SAME RHS
            # Solve exactly to get "true" solution for error target computation
            true_sol = np.linalg.solve(matrix, single_rhs)

            # Create array of identical RHS and solution vectors
            rhs_samples = np.tile(single_rhs, (num_base_systems, 1))
            sols = np.tile(true_sol, (num_base_systems, 1))
        else:
            sols = self._provide_true_solutions(
                matrix,
                num_base_systems,
                config,
                archive,
                rng,
            )

            # Layer 2: Transform (compute RHS or load from archive)
            rhs_provider = HybridInputProvider(archive=archive, field="rhs_vectors", scale=1.0)
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

        for sample_idx, (rhs_vec, true_sol) in enumerate(zip(rhs_samples, sols)):
            _, info = solver(
                matrix,
                rhs_vec,
                np.zeros(n, dtype=np.float64),
                maxiter=cg_iters,
                rtol=1e-20,
                atol=1e-20,
            )

            residual_seq = _tensor_trace_to_numpy(info.residual_vectors)
            solution_seq = _tensor_trace_to_numpy(info.solution_vectors)

            residual_seq = residual_seq[:: config.every_n]
            if solution_seq.size > 0:
                solution_seq = solution_seq[:: config.every_n]
            num_pairs = residual_seq.shape[0]

            error_seq = np.array([true_sol - x_k for x_k in solution_seq], dtype=np.float64)

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


@register_single_rhs_strategy(supports_matrix_replacement=True)
class ResidualsStrategy(_BaseResidualsStrategy):
    name = "residuals"


@register_single_rhs_strategy(supports_matrix_replacement=True)
class GaussianResidualsStrategy(_BaseResidualsStrategy):
    name = "gaussian_residuals"

    def _provide_true_solutions(
        self,
        matrix: np.ndarray,
        count: int,
        config: ResidualErrorConfig,
        archive: ArchiveData | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        provider = RandomInputProvider(seed=config.seed, scale=1.0)
        return provider.provide(matrix, count=count, rng=rng)
