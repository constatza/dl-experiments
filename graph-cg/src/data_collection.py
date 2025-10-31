#!/usr/bin/env python3
"""Data collection and generation utilities for graph-cg project.

This module provides unified functions for:
1. Collecting data from existing sources (SpectralData)
2. Generating synthetic data with various strategies
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .constants import DEFAULT_RESIDUAL_TRACE_ITERS, ConfigKeys
from .data_pipeline import (
    DataContext,
    RawSamples,
    normalize_samples,
    persist_normalized_samples,
)
from .data_generation import rounded_counts, rng_from_seed
from .sample_builders import (
    build_rhs_archive_samples,
    build_generated_samples,
    build_solution_archive_samples,
)
from .normalization import ResidualTraceSamples


def _merge_residual_traces(
    base: ResidualTraceSamples | None,
    addition: ResidualTraceSamples | None,
    offset: int,
) -> ResidualTraceSamples | None:
    """Merge residual trace blocks, adjusting sample indices for appended samples."""

    if base is None:
        if addition is None:
            return None
        return ResidualTraceSamples(
            residuals=addition.residuals.copy(),
            solutions=addition.solutions.copy(),
            sample_indices=addition.sample_indices.copy(),
            iteration_indices=addition.iteration_indices.copy(),
        )

    if addition is None:
        return ResidualTraceSamples(
            residuals=base.residuals.copy(),
            solutions=base.solutions.copy(),
            sample_indices=base.sample_indices.copy(),
            iteration_indices=base.iteration_indices.copy(),
        )

    return ResidualTraceSamples(
        residuals=np.vstack([base.residuals, addition.residuals]),
        solutions=np.vstack([base.solutions, addition.solutions]),
        sample_indices=np.concatenate(
            [base.sample_indices, addition.sample_indices + offset]
        ),
        iteration_indices=np.concatenate(
            [base.iteration_indices, addition.iteration_indices]
        ),
    )


def _append_raw_samples(base: RawSamples | None, addition: RawSamples) -> RawSamples:
    """Append RawSamples together while keeping matrices aligned."""

    if base is None:
        return addition

    if base.matrix.shape != addition.matrix.shape:
        raise ValueError(
            "Cannot merge samples with different matrix shapes: "
            f"{base.matrix.shape} vs {addition.matrix.shape}"
        )

    if not np.allclose(base.matrix, addition.matrix):
        raise ValueError("Matrix entries differ between sample blocks; ensure shared system matrix.")

    rhs = np.vstack([base.rhs, addition.rhs])
    solutions = np.vstack([base.solutions, addition.solutions])
    offset = base.rhs.shape[0]
    residual_traces = _merge_residual_traces(base.residual_traces, addition.residual_traces, offset)

    mother_rhs = base.mother_rhs
    return RawSamples(
        matrix=base.matrix,
        rhs=rhs,
        solutions=solutions,
        mother_rhs=mother_rhs,
        residual_traces=residual_traces,
    )


def _shuffle_raw_samples(samples: RawSamples, seed: int) -> RawSamples:
    """Shuffle samples and update residual trace indices consistently."""

    rng = rng_from_seed(seed)
    num_samples = samples.rhs.shape[0]
    permutation = rng.permutation(num_samples)

    rhs = samples.rhs[permutation]
    solutions = samples.solutions[permutation]
    mother_rhs = samples.mother_rhs.copy()

    residual_traces = samples.residual_traces
    if residual_traces is not None:
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(num_samples)
        residual_traces = ResidualTraceSamples(
            residuals=residual_traces.residuals.copy(),
            solutions=residual_traces.solutions.copy(),
            sample_indices=inverse[residual_traces.sample_indices],
            iteration_indices=residual_traces.iteration_indices.copy(),
        )

    return RawSamples(
        matrix=samples.matrix,
        rhs=rhs,
        solutions=solutions,
        mother_rhs=mother_rhs,
        residual_traces=residual_traces,
    )

def collect_case(
    matrix_path: str | Path,
    rhs_path: str | Path,
    dataset_dir: str | Path,
    normalize: Literal["none", "matrix", "rhs", "spectral"] = "matrix",
    solve_systems: bool = True,
    cg_tolerance: float = 1e-12,
    cg_max_iters: int = 500,
) -> Path:
    """Collect data from existing source and solve linear systems.

    Args:
        matrix_path: Absolute path to matrix file
        rhs_path: Absolute glob pattern for RHS files (supports wildcards)
        dataset_dir: Final dataset directory (unique per flow/dataset)
        normalize: Normalization method
            - "none": no normalization applied
            - "matrix": normalizes by spectral radius bound (recommended)
            - "spectral": normalizes matrix by spectral norm, embeds scale into solution
        solve_systems: Whether to solve systems (True) or just collect RHS
        cg_tolerance: CG solver relative tolerance (default: 1e-12 for near-exact solutions)
        cg_max_iters: CG solver max iterations

    Returns:
        Path to created data directory
    """
    from prefect.concurrency.sync import concurrency

    matrix_path = Path(matrix_path)
    rhs_path = Path(rhs_path)
    dataset_dir = Path(dataset_dir)

    # Validate normalize parameter
    if isinstance(normalize, bool):
        raise ValueError(
            f"Invalid normalize value: {normalize} (bool). "
            f"The 'normalize' parameter no longer accepts boolean values. "
            f"Please update your config to use: 'matrix' (recommended), 'rhs', or 'none'. "
            f"Migration: True -> 'matrix', False -> 'none'"
        )

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    rhs_dir = rhs_path.parent
    rhs_pattern = rhs_path.name

    if not rhs_dir.exists():
        raise FileNotFoundError(f"RHS directory not found: {rhs_dir}")

    # Prefect concurrency control: only 1 data collection at a time
    # Prevents concurrent writes to the same dataset directory
    # If "data-generation" limit doesn't exist, this just logs a warning (no enforcement)
    # To enforce strict locking: prefect gcl create data-generation --limit 1
    with concurrency("data-generation", occupy=1):
        print(f"Collecting data from matrix: {matrix_path}")

        rhs_files = sorted(rhs_dir.glob(rhs_pattern))
        if not rhs_files:
            raise FileNotFoundError(
                f"No RHS files found matching: {rhs_dir / rhs_pattern}"
            )

        print(f"Found {len(rhs_files)} RHS files")

        context = DataContext(
            matrix_path=matrix_path,
            dataset_dir=dataset_dir,
            normalize=normalize,
        extras={
            "source_type": ConfigKeys.TYPE_RHS_ARCHIVE,
            "solve_systems": solve_systems,
            "cg_tolerance": cg_tolerance,
            "cg_max_iters": cg_max_iters,
        },
        )

        raw_samples = build_rhs_archive_samples(
            matrix_path=matrix_path,
            rhs_glob=rhs_files,
            solve_systems=solve_systems,
            cg_tolerance=cg_tolerance,
            cg_max_iters=cg_max_iters,
        )

        dimension = raw_samples.matrix.shape[0]
        num_samples = raw_samples.rhs.shape[0]
        print(f"Matrix dimension: {dimension}")
        print(f"Collected {num_samples} RHS samples, shape: {raw_samples.rhs.shape}")

        normalization_result = normalize_samples(context, raw_samples)

        print(f"Saving to: {dataset_dir}")
        persist_normalized_samples(
            dataset_dir=dataset_dir,
            normalized=normalization_result,
            mother_rhs_vector=raw_samples.mother_rhs,
        )

        scale = normalization_result.matrix_scale
        spectral_radius_bound = normalization_result.spectral_radius_bound
        spectral_norm_value = normalization_result.spectral_norm

        # Save metadata
        matrix_dir = matrix_path.parent
        rhs_dir_resolved = rhs_dir
        matrix_ancestors = {ancestor for ancestor in matrix_path.parents}
        case_path = matrix_dir
        for ancestor in rhs_dir_resolved.parents:
            if ancestor in matrix_ancestors:
                case_path = ancestor
                break

        metadata = {
            "source_type": ConfigKeys.TYPE_RHS_ARCHIVE,
            "case_path": str(case_path),
            "dimension": int(dimension),
            "num_samples": int(num_samples),
            "normalize": str(normalize),
            "matrix_shape": list(raw_samples.matrix.shape),
            "matrix_scale": float(scale),
            "spectral_radius_bound": float(spectral_radius_bound) if spectral_radius_bound is not None else None,
            "spectral_norm": float(spectral_norm_value) if spectral_norm_value is not None else None,
            "solved": bool(solve_systems),
            "created_at": datetime.now().isoformat(),
        }

        metadata_path = dataset_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Collection complete for dataset: {dataset_dir.name}")
        return dataset_dir


def generate_case(
    matrix_path: str | Path,
    rhs_path: str | Path,
    num_samples: int,
    dataset_dir: str | Path,
    mix: dict[str, float] | None = None,
    krylov_iters: int = 15,
    residual_iters: int = DEFAULT_RESIDUAL_TRACE_ITERS,
    seed: int = 42,
    normalize: Literal["none", "matrix", "rhs", "spectral"] = "matrix",
    shuffle: bool = True,
    rhs_archive_glob: str | None = None,
    solution_archive_options: Mapping[str, Any] | None = None,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Generate synthetic training data.

    Args:
        matrix_path: Path to matrix file
        rhs_path: Path to RHS file
        num_samples: Number of samples to generate
        dataset_dir: Final dataset directory (unique per flow/dataset)
        mix: Strategy mix (e.g., {"normal": 0.5, "krylov": 0.5})
        krylov_iters: CG iterations for krylov strategy
        residual_iters: Number of CG iterations to capture residual traces
        seed: Random seed
        normalize: Normalization method
            - "none": no normalization applied
            - "matrix": normalizes by spectral radius bound (recommended)
            - "spectral": normalizes matrix by spectral norm, embeds scale into solution
        shuffle: Whether to shuffle final dataset
        rhs_archive_glob: Optional glob pattern for incorporating archived RHS samples
            when mix contains a "rhs_archive" key.
        strategy_overrides: Optional per-strategy overrides applied when generating
            synthetic samples (e.g., {"krylov": {"krylov_iters": 10}})

    Returns:
        Path to created data directory
    """
    from prefect.concurrency.sync import concurrency

    dataset_dir = Path(dataset_dir)
    matrix_path = Path(matrix_path)
    rhs_path = Path(rhs_path)

    # Validate normalize parameter
    if isinstance(normalize, bool):
        raise ValueError(
            f"Invalid normalize value: {normalize} (bool). "
            f"The 'normalize' parameter no longer accepts boolean values. "
            f"Please update your config to use: 'matrix' (recommended), 'rhs', or 'none'. "
            f"Migration: True -> 'matrix', False -> 'none'"
        )

    if mix is None:
        mix = {"normal": 1.0}

    # Prefect concurrency control: only 1 data generation at a time
    # Prevents concurrent writes to the same dataset directory
    # If "data-generation" limit doesn't exist, this just logs a warning (no enforcement)
    # To enforce strict locking: prefect gcl create data-generation --limit 1
    with concurrency("data-generation", occupy=1):

        print("Generating synthetic data...")
        print(f"  Total samples: {num_samples}")
        print(f"  Mix: {mix}")

        if abs(sum(mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"Mix fractions must sum to 1.0, got {sum(mix.values())}")

        counts = rounded_counts(num_samples, mix)
        rhs_archive_count = counts.get(ConfigKeys.TYPE_RHS_ARCHIVE, 0)
        solution_archive_count = counts.get(ConfigKeys.TYPE_SOLUTION_ARCHIVE, 0)
        synthetic_mix = {
            k: v
            for k, v in mix.items()
            if k not in {ConfigKeys.TYPE_RHS_ARCHIVE, ConfigKeys.TYPE_SOLUTION_ARCHIVE}
        }
        synthetic_total = num_samples - rhs_archive_count - solution_archive_count

        if rhs_archive_count:
            print(f"  RHS archive samples requested: {rhs_archive_count}")
        print(f"  Synthetic samples requested: {synthetic_total}")

        raw_samples: RawSamples | None = None
        synthetic_mix_normalized: dict[str, float] | None = None

        if synthetic_total > 0:
            if not synthetic_mix:
                raise ValueError(
                    "Synthetic portion requested but no synthetic strategies configured in mix"
                )
            synthetic_weight = sum(synthetic_mix.values())
            if synthetic_weight <= 0:
                raise ValueError("Synthetic strategy weights must be positive")
            synthetic_mix_normalized = {
                name: value / synthetic_weight for name, value in synthetic_mix.items()
            }

            synthetic_samples = build_generated_samples(
                matrix_path=Path(matrix_path),
                rhs_path=Path(rhs_path),
                num_samples=synthetic_total,
                mix=synthetic_mix_normalized,
                krylov_iters=krylov_iters,
                residual_iters=residual_iters,
                seed=seed,
                shuffle=shuffle,
                normalize_mode=normalize,
                strategy_overrides=strategy_overrides,
            )
            raw_samples = synthetic_samples

        if rhs_archive_count > 0:
            if not rhs_archive_glob:
                raise ValueError(
                    "Mix references 'rhs_archive' but no RHS source was configured. "
                    f"Provide 'rhs_glob' inside the strategy or set generation.{ConfigKeys.RHS_ARCHIVE_GLOB}."
                )

            rhs_pattern = Path(rhs_archive_glob)
            rhs_dir = rhs_pattern.parent
            rhs_candidates = sorted(rhs_dir.glob(rhs_pattern.name))

            if len(rhs_candidates) < rhs_archive_count:
                raise ValueError(
                    f"Requested {rhs_archive_count} rhs_archive samples but only "
                    f"{len(rhs_candidates)} available for pattern {rhs_archive_glob}"
                )

            selection_indices = rng_from_seed(seed).permutation(len(rhs_candidates))[:rhs_archive_count]
            selected_files = [rhs_candidates[idx] for idx in selection_indices]

            archive_samples = build_rhs_archive_samples(
                matrix_path=Path(matrix_path),
                rhs_glob=selected_files,
                solve_systems=True,
                cg_tolerance=1e-12,
                cg_max_iters=500,
            )

            raw_samples = _append_raw_samples(raw_samples, archive_samples)

        if solution_archive_count > 0:
            if not solution_archive_options:
                raise ValueError(
                    "Mix references 'solution_archive' but no solution archive options were provided."
                )
            solutions_glob = solution_archive_options.get(ConfigKeys.SOLUTIONS_GLOB)
            if not isinstance(solutions_glob, str) or not solutions_glob:
                raise ValueError(
                    "Solution archive strategy requires 'solutions_glob' to be specified."
                )

            shuffle_solution = bool(
                solution_archive_options.get(ConfigKeys.SHUFFLE, shuffle)
            )
            solution_seed = solution_archive_options.get(ConfigKeys.SEED, seed)
            solution_seed_int = int(solution_seed) if solution_seed is not None else None

            solution_pattern = Path(solutions_glob)
            solution_dir = solution_pattern.parent
            solution_candidates = sorted(solution_dir.glob(solution_pattern.name))
            if len(solution_candidates) < solution_archive_count:
                raise ValueError(
                    f"Requested {solution_archive_count} solution_archive samples but only "
                    f"{len(solution_candidates)} available for pattern {solutions_glob}"
                )

            if shuffle_solution:
                indices = rng_from_seed(solution_seed_int).permutation(len(solution_candidates))[
                    :solution_archive_count
                ]
                selected_solutions = [solution_candidates[idx] for idx in indices]
            else:
                selected_solutions = solution_candidates[:solution_archive_count]

            solution_samples = build_solution_archive_samples(
                matrix_path=Path(matrix_path),
                solution_files=selected_solutions,
                shuffle=False,
                seed=None,
            )

            raw_samples = _append_raw_samples(raw_samples, solution_samples)

        if raw_samples is None:
            raise RuntimeError("No samples were generated; check mix configuration")

        if shuffle and rhs_archive_count > 0 and synthetic_total > 0:
            raw_samples = _shuffle_raw_samples(raw_samples, seed)

        dimension = raw_samples.matrix.shape[0]
        print(f"  Dimension: {dimension}")
        print("Generating samples...")

        extras: dict[str, Any] = {
            "source_type": ConfigKeys.TYPE_GENERATED,
            "mix": mix.copy(),
            "krylov_iters": krylov_iters,
            "residual_iters": residual_iters,
            "seed": seed,
            "shuffle": shuffle,
            "synthetic_samples": synthetic_total,
            "rhs_archive_samples": rhs_archive_count,
            "solution_archive_samples": solution_archive_count,
        }
        if synthetic_mix_normalized is not None:
            extras["synthetic_mix_normalized"] = synthetic_mix_normalized
        if rhs_archive_glob is not None:
            extras["rhs_archive_glob"] = rhs_archive_glob
        if solution_archive_count > 0 and solution_archive_options is not None:
            extras["solution_archive_glob"] = solution_archive_options.get(ConfigKeys.SOLUTIONS_GLOB)
            extras["solution_archive_seed"] = solution_archive_options.get(ConfigKeys.SEED)
        if strategy_overrides:
            extras["strategy_overrides"] = {
                key: dict(options) for key, options in strategy_overrides.items()
            }

        context = DataContext(
            matrix_path=Path(matrix_path),
            dataset_dir=dataset_dir,
            normalize=normalize,
            extras=extras,
        )

        normalization_result = normalize_samples(context, raw_samples)

        print(f"Saving to: {dataset_dir}")
        persist_normalized_samples(
            dataset_dir=dataset_dir,
            normalized=normalization_result,
            mother_rhs_vector=raw_samples.mother_rhs,
        )

        scale = normalization_result.matrix_scale
        spectral_radius_bound = normalization_result.spectral_radius_bound
        spectral_norm_value = normalization_result.spectral_norm

        # Save metadata
        metadata = {
            "source_type": ConfigKeys.TYPE_GENERATED,
            "matrix_path": str(matrix_path),
            "rhs_path": str(rhs_path),
            "dimension": int(dimension),
            "num_samples": int(num_samples),
            "normalize": str(normalize),
            "matrix_scale": float(scale),
            "spectral_radius_bound": float(spectral_radius_bound) if spectral_radius_bound is not None else None,
            "spectral_norm": float(spectral_norm_value) if spectral_norm_value is not None else None,
            "mix": {k: float(v) for k, v in mix.items()},
            "krylov_iters": int(krylov_iters),
            "residual_iters": int(residual_iters),
            "synthetic_samples": int(synthetic_total),
            "rhs_archive_samples": int(rhs_archive_count),
            "solution_archive_samples": int(solution_archive_count),
            "seed": int(seed),
            "matrix_shape": list(raw_samples.matrix.shape),
            "created_at": datetime.now().isoformat(),
        }

        if synthetic_mix_normalized is not None:
            metadata["synthetic_mix_normalized"] = {
                k: float(v) for k, v in synthetic_mix_normalized.items()
            }
        if rhs_archive_glob is not None:
            metadata["rhs_archive_glob"] = str(rhs_archive_glob)
        if solution_archive_count > 0 and solution_archive_options is not None:
            metadata["solution_archive_glob"] = str(
                solution_archive_options.get(ConfigKeys.SOLUTIONS_GLOB)
            )
            metadata["solution_archive_seed"] = solution_archive_options.get(ConfigKeys.SEED)
        if strategy_overrides:
            metadata["strategy_overrides"] = {
                key: dict(options) for key, options in strategy_overrides.items()
            }

        metadata_path = dataset_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Generation complete for dataset: {dataset_dir.name}")
        return dataset_dir


def solution_archive_case(
    matrix_path: str | Path,
    solutions_path: str | Path,
    dataset_dir: str | Path,
    normalize: Literal["none", "matrix", "rhs", "spectral"] = "matrix",
    shuffle: bool = True,
    seed: int | None = None,
) -> Path:
    """Build dataset from an archive of pre-computed solution vectors."""

    from prefect.concurrency.sync import concurrency

    matrix_path = Path(matrix_path)
    solutions_path = Path(solutions_path)
    dataset_dir = Path(dataset_dir)

    if isinstance(normalize, bool):
        raise ValueError(
            f"Invalid normalize value: {normalize} (bool). "
            f"The 'normalize' parameter no longer accepts boolean values. "
            f"Please update your config to use: 'matrix' (recommended), 'rhs', or 'none'. "
            f"Migration: True -> 'matrix', False -> 'none'"
        )

    with concurrency("data-generation", occupy=1):
        print(f"Loading matrix from: {matrix_path}")
        print(f"Consuming solution archive from: {solutions_path}")

        solution_dir = solutions_path.parent
        solution_pattern = solutions_path.name
        if not solution_dir.exists():
            raise FileNotFoundError(f"Solutions directory not found: {solution_dir}")

        solution_files = sorted(solution_dir.glob(solution_pattern))
        if not solution_files:
            raise FileNotFoundError(
                f"No solution files found matching: {solution_dir / solution_pattern}"
            )

        context = DataContext(
            matrix_path=matrix_path,
            dataset_dir=dataset_dir,
            normalize=normalize,
            extras={
                "source_type": ConfigKeys.TYPE_SOLUTION_ARCHIVE,
                "shuffle": shuffle,
                "seed": seed,
            },
        )

        raw_samples = build_solution_archive_samples(
            matrix_path=matrix_path,
            solution_files=solution_files,
            shuffle=shuffle,
            seed=seed,
        )

        dimension = raw_samples.matrix.shape[0]
        num_samples = raw_samples.rhs.shape[0]
        print(f"Matrix dimension: {dimension}")
        print(f"Loaded {num_samples} solutions from archive")

        normalization_result = normalize_samples(context, raw_samples)

        print(f"Saving to: {dataset_dir}")
        persist_normalized_samples(
            dataset_dir=dataset_dir,
            normalized=normalization_result,
            mother_rhs_vector=raw_samples.mother_rhs,
        )

        metadata = {
            "source_type": ConfigKeys.TYPE_SOLUTION_ARCHIVE,
            "matrix_path": str(matrix_path),
            "solutions_path": str(solutions_path),
            "dimension": int(dimension),
            "num_samples": int(num_samples),
            "normalize": str(normalize),
            "matrix_scale": float(normalization_result.matrix_scale),
            "spectral_radius_bound": float(normalization_result.spectral_radius_bound)
            if normalization_result.spectral_radius_bound is not None
            else None,
            "spectral_norm": float(normalization_result.spectral_norm)
            if normalization_result.spectral_norm is not None
            else None,
            "shuffle": bool(shuffle),
            "seed": int(seed) if seed is not None else None,
            "matrix_shape": list(raw_samples.matrix.shape),
            "created_at": datetime.now().isoformat(),
        }

        metadata_path = dataset_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Solution bank ingestion complete for dataset: {dataset_dir.name}")
        return dataset_dir


def load_case_data(data_dir: str | Path) -> dict[str, np.ndarray | dict[str, Any]]:
    """Load all data from a case directory.

    Args:
        data_dir: Path to case directory

    Returns:
        Dictionary with keys: matrix, rhs_samples, sol_samples, rhs_mother, metadata
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    result = {}

    # Load single matrix file
    matrix_path = data_dir / "matrix.npy"
    if matrix_path.exists():
        result["matrix"] = np.load(matrix_path).astype(np.float64, copy=False)
    else:
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    # Load other arrays
    for name in ["rhs-samples", "sol-samples", "rhs-mother"]:
        path = data_dir / f"{name}.npy"
        if path.exists():
            result[name.replace("-", "_")] = np.load(path).astype(np.float64, copy=False)

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            result["metadata"] = json.load(f)

    # Load normalization metadata (legacy)
    norm_path = data_dir / "normalization.json"
    if norm_path.exists():
        with norm_path.open("r", encoding="utf-8") as f:
            result["normalization"] = json.load(f)

    return result
