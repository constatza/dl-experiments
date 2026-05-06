#!/usr/bin/env python3
"""Unified data processing interface across RHS archives, synthetic generation, and solution archives."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from neuralls.platform.config.models.data_models import DataConfigFile, GenerationConfig
from neuralls.domain.generation.data_types import NormalizeType
from neuralls.shared.constants import DEFAULT_RANDOM_SEED
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.plan import (
    GenerationPlan,
    StrategySpec,
    canonicalize_strategy_name,
    plan_from_specs,
)
from neuralls.domain.generation.runner import run_generation
from neuralls.composition.generation.default_services import (
    make_direction_solver,
    make_residual_solver,
)
from neuralls.domain.generation.ports import TracingSolverPort


def _make_residual_solver() -> TracingSolverPort:
    """Construct default tracing solver for residuals / residual_traces strategies."""
    return make_residual_solver()


def _make_direction_solver() -> TracingSolverPort:
    """Construct default tracing solver for search_directions strategy."""
    return make_direction_solver()


_DEFAULT_RESIDUAL_SOLVER: TracingSolverPort = _make_residual_solver()
_DEFAULT_DIRECTION_SOLVER: TracingSolverPort = _make_direction_solver()


def _plan_from_generation_config(
    generation_cfg: GenerationConfig,
) -> GenerationPlan:
    """Translate a validated GenerationConfig into a GenerationPlan.

    Converts Pydantic StrategyConfig objects directly to StrategySpec objects,
    bypassing raw-dict re-parsing.
    """
    if not generation_cfg.strategy:
        raise ValueError(
            "Missing '[[generation.strategy]]' entries in config. "
            "Add at least one strategy with 'name' and 'samples' fields."
        )
    specs: list[StrategySpec] = []
    for sc in generation_cfg.strategy:
        options = sc.model_dump(mode="python", exclude={"name", "samples"}, exclude_none=True)
        specs.append(
            StrategySpec(
                raw_name=sc.name,
                canonical_name=canonicalize_strategy_name(sc.name),
                samples=sc.samples,
                options=options,
            )
        )
    return plan_from_specs(specs)


@dataclass(frozen=True)
class DataGenerationContext:
    """Immutable context assembled from config inputs."""

    matrix_path: str
    rhs_path: str | None
    solutions_path: str | None
    sample_id_regex: str | None
    dataset_dir: Path
    normalize: NormalizeType
    seed: int
    shuffle: bool


def _build_context(
    *,
    config: DataConfigFile,
) -> tuple[DataGenerationContext, GenerationPlan]:
    """Assemble generation context, plan, and dataset directory."""
    if config.output.data_dir is None:
        raise ValueError("output.data_dir must be resolved before process_config().")
    dataset_dir = config.output.data_dir / config.id

    matrix_path = config.source.matrix_path
    if not matrix_path:
        raise ValueError(f"Missing '{'source'}.{'matrix_path'}' in config")

    rhs_path = config.source.rhs_path
    solutions_path = config.source.solutions_path
    sample_id_regex = config.source.sample_id_regex
    normalize_value = config.generation.normalize
    if isinstance(normalize_value, bool):
        raise ValueError(
            f"Invalid normalize value in config: {normalize_value} (bool). "
            f"The 'normalize' parameter expects one of "
            "'spectral', 'matrix', 'rhs', 'diagonal', or 'none'."
        )
    normalize = cast(NormalizeType, str(normalize_value))

    plan = _plan_from_generation_config(config.generation)

    return DataGenerationContext(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        solutions_path=solutions_path,
        sample_id_regex=sample_id_regex,
        dataset_dir=dataset_dir,
        normalize=normalize,
        seed=config.generation.seed,
        shuffle=config.generation.shuffle,
    ), plan


def _merge_rhs_archive_options(
    strategy: StrategySpec | None,
    generation_cfg: GenerationConfig,
) -> dict[str, Any]:
    """Combine generation-level options with strategy-specific overrides."""
    relevant_keys = ("cg_tolerance", "cg_max_iters")
    merged: dict[str, Any] = {}

    for key in relevant_keys:
        value = getattr(generation_cfg, key, None)
        if value is not None:
            merged[key] = value
        if strategy and key in strategy.options:
            merged[key] = strategy.options[key]

    return merged


def _resolve_rhs_archive_glob(
    strategy: StrategySpec | None,
    context: DataGenerationContext,
    generation_cfg: GenerationConfig,
) -> str | None:
    """Resolve the RHS glob path to use for RHS archive samples."""
    candidates: tuple[Any, ...] = (
        strategy.options.get("rhs_glob") if strategy else None,
        strategy.options.get("rhs_path") if strategy else None,
        generation_cfg.rhs_archive_glob,
        context.rhs_path,
    )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    return None


def _resolve_solution_archive_path(
    strategy: StrategySpec,
    context: DataGenerationContext,
) -> str | None:
    """Resolve the solutions glob/path for solution archive ingestion."""
    candidates: tuple[Any, ...] = (
        strategy.options.get("solutions_glob"),
        strategy.options.get("solutions_path"),
        context.solutions_path,
    )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    return None


def _derive_rhs_from_solution_archive(
    *,
    matrix: np.ndarray,
    solutions_glob: str,
    dataset_dir: Path,
) -> Path:
    """Create a fallback RHS vector by applying ``A @ x`` to a stored solution."""
    pattern = Path(solutions_glob)
    solution_dir = pattern.parent
    if not solution_dir.exists():
        raise FileNotFoundError(
            f"Solutions directory not found for provide_rhs automation: {solution_dir}"
        )

    candidates = sorted(solution_dir.glob(pattern.name))
    if not candidates:
        raise FileNotFoundError(
            "No solution files available to derive RHS for provide_rhs automation: "
            f"{solutions_glob}"
        )

    # Use solution_archive strategy to load one solution and compute RHS
    representative = candidates[0]

    # Load single solution using solution_archive strategy
    samples = run_generation(
        "solution_archive",
        matrix,
        cfg={
            "solutions_glob": str(representative),
            "samples": 1,
        },
        archive=None,
    )

    if samples.rhs is None or len(samples.rhs) == 0:
        raise ValueError(f"Failed to load solution from {representative}")

    rhs_vector = samples.rhs[0]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "mother-rhs.txt"
    np.savetxt(output_path, rhs_vector, fmt="%.18e")
    return output_path


def _resolve_rhs_source(
    context: DataGenerationContext,
    solution_archive_options: Mapping[str, Any] | None,
    provide_rhs_option: Any,
    matrix: np.ndarray,
) -> str | None:
    """Determine RHS path for synthetic generation, allowing provide_rhs overrides."""
    if context.rhs_path:
        return context.rhs_path

    if provide_rhs_option is None:
        return None

    if isinstance(provide_rhs_option, (str, Path)):
        path = Path(provide_rhs_option)
        if not path.exists():
            raise FileNotFoundError(f"provide_rhs path does not exist: {path}")
        return str(path)

    if isinstance(provide_rhs_option, bool) and provide_rhs_option:
        if not solution_archive_options:
            raise ValueError(
                "provide_rhs=True requires solution archive options to resolve a solutions glob."
            )
        solutions_glob = solution_archive_options.get("solutions_glob")
        if not isinstance(solutions_glob, str) or not solutions_glob:
            raise ValueError(
                "provide_rhs=True requires 'solutions_glob' for the solution_archive strategy."
            )
        generated_path = _derive_rhs_from_solution_archive(
            matrix=matrix,
            solutions_glob=solutions_glob,
            dataset_dir=context.dataset_dir,
        )
        return str(generated_path)

    return None


def _execute_solution_archive(
    context: DataGenerationContext,
    strategy: StrategySpec,
    generation_cfg: GenerationConfig,
) -> Path:
    """Execute solution archive using new build_dataset() with solution_archive strategy."""
    resolved_solutions_path = _resolve_solution_archive_path(strategy, context)
    if resolved_solutions_path is None:
        raise ValueError(
            "Solution archive strategy requires either "
            "a 'solutions_glob' field within [[generation.strategy]] name='solution_archive' or "
            f"'{'source'}.{'solutions_path'}'."
        )

    shuffle_value = strategy.options.get("shuffle", generation_cfg.shuffle)
    seed_value = strategy.options.get("seed", generation_cfg.seed)

    dataset_path = build_dataset(
        matrix_path=context.matrix_path,
        dataset_dir=str(context.dataset_dir),
        counts={"solution_archive": strategy.samples},
        solutions_path=context.solutions_path,
        sample_id_regex=context.sample_id_regex,
        normalize=context.normalize,
        shuffle=bool(shuffle_value),
        seed=int(seed_value) if seed_value is not None else DEFAULT_RANDOM_SEED,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": resolved_solutions_path,
                "shuffle": bool(shuffle_value),
                "seed": seed_value,
            }
        },
    )
    return Path(dataset_path)


def _execute_synthetic_generation(
    context: DataGenerationContext,
    synthetic_strategies: Mapping[str, StrategySpec],
    rhs_archive_strategy: StrategySpec | None,
    solution_archive_strategy: StrategySpec | None,
    generation_cfg: GenerationConfig,
    matrix: np.ndarray,
) -> Path:
    """Execute mixed generation using new build_dataset() with all strategies."""
    seed_value = generation_cfg.seed
    if seed_value is None:
        seed_value = DEFAULT_RANDOM_SEED
    seed = int(seed_value)
    shuffle_value = generation_cfg.shuffle

    # Build counts dict with ALL strategies (synthetic + archives)
    counts: dict[str, int] = {}
    strategy_overrides: dict[str, dict[str, Any]] = {}

    # Add synthetic strategies
    for name, spec in synthetic_strategies.items():
        counts[name] = spec.samples
        if spec.options:
            strategy_overrides[name] = dict(spec.options)

    # Add solution archive strategy
    if solution_archive_strategy is not None:
        counts["solution_archive"] = solution_archive_strategy.samples
        solutions_glob = _resolve_solution_archive_path(solution_archive_strategy, context)
        if solutions_glob:
            strategy_overrides["solution_archive"] = {
                "solutions_glob": solutions_glob,
                "shuffle": solution_archive_strategy.options.get(
                    "shuffle",
                    generation_cfg.shuffle,
                ),
                "seed": solution_archive_strategy.options.get("seed", seed_value),
            }

    # Add RHS archive strategy
    if rhs_archive_strategy is not None:
        counts["rhs_archive"] = rhs_archive_strategy.samples
        rhs_archive_glob = _resolve_rhs_archive_glob(rhs_archive_strategy, context, generation_cfg)
        if rhs_archive_glob is None:
            raise ValueError(
                "RHS archive strategy requires 'rhs_glob' within the strategy, "
                f"'{'source'}.{'rhs_path'}', "
                f"or 'generation.{'rhs_archive_glob'}'."
            )

        rhs_archive_opts = _merge_rhs_archive_options(rhs_archive_strategy, generation_cfg)
        strategy_overrides["rhs_archive"] = {
            "rhs_glob": rhs_archive_glob,
            **rhs_archive_opts,
        }

    # Resolve RHS source path (for synthetic strategies that need it)
    rhs_source_path = _resolve_rhs_source(
        context=context,
        solution_archive_options=solution_archive_strategy.options
        if solution_archive_strategy
        else None,
        provide_rhs_option=None,
        matrix=matrix,
    )

    dataset_path = build_dataset(
        matrix_path=context.matrix_path,
        dataset_dir=str(context.dataset_dir),
        counts=counts,
        rhs_path=rhs_source_path,
        solutions_path=context.solutions_path,
        sample_id_regex=context.sample_id_regex,
        normalize=context.normalize,
        shuffle=bool(shuffle_value),
        seed=seed,
        strategy_overrides=strategy_overrides,
        solver_overrides={
            "residuals": _DEFAULT_RESIDUAL_SOLVER,
            "gaussian_residuals": _DEFAULT_RESIDUAL_SOLVER,
            "residual_traces": _DEFAULT_RESIDUAL_SOLVER,
            "search_directions": _DEFAULT_DIRECTION_SOLVER,
        },
    )
    return Path(dataset_path)


def _execute_rhs_archive_only(
    context: DataGenerationContext,
    strategy: StrategySpec,
    generation_cfg: GenerationConfig,
) -> Path:
    """Execute RHS archive using new build_dataset() with rhs_archive strategy."""
    rhs_glob = _resolve_rhs_archive_glob(strategy, context, generation_cfg)
    if rhs_glob is None:
        raise ValueError(
            "RHS archive strategy requires either 'rhs_glob' in the strategy "
            f"or '{'source'}.{'rhs_path'}'."
        )

    collection_kwargs = _merge_rhs_archive_options(strategy, generation_cfg)
    seed_value = generation_cfg.seed

    dataset_path = build_dataset(
        matrix_path=context.matrix_path,
        dataset_dir=str(context.dataset_dir),
        counts={"rhs_archive": strategy.samples},
        sample_id_regex=context.sample_id_regex,
        normalize=context.normalize,
        seed=int(seed_value) if seed_value is not None else DEFAULT_RANDOM_SEED,
        strategy_overrides={
            "rhs_archive": {
                "rhs_glob": rhs_glob,
                **collection_kwargs,
            }
        },
    )
    return Path(dataset_path)


def _execute_plan(
    context: DataGenerationContext,
    generation_cfg: GenerationConfig,
    plan: GenerationPlan,
    matrix: np.ndarray,
) -> Path:
    rhs_archive_strategy = plan.rhs_archive
    solution_archive_strategy = plan.solution_archive
    synthetic_strategies = plan.synthetic

    has_solution_archive = solution_archive_strategy is not None
    has_rhs_archive = rhs_archive_strategy is not None
    has_synthetic_strategies = bool(synthetic_strategies)

    if solution_archive_strategy is not None and not (has_rhs_archive or has_synthetic_strategies):
        return _execute_solution_archive(context, solution_archive_strategy, generation_cfg)

    if has_synthetic_strategies or has_solution_archive:
        return _execute_synthetic_generation(
            context=context,
            synthetic_strategies=synthetic_strategies,
            rhs_archive_strategy=rhs_archive_strategy,
            solution_archive_strategy=solution_archive_strategy,
            generation_cfg=generation_cfg,
            matrix=matrix,
        )

    if rhs_archive_strategy is None:
        raise ValueError("No generation strategies configured")

    return _execute_rhs_archive_only(context, rhs_archive_strategy, generation_cfg)


def process_config(
    config: DataConfigFile,
    matrix: np.ndarray,
) -> Path:
    """Process a data config and execute the declared generation plan.

    After generating the main dataset, this checks for a [test] section.
    Dedicated comparison split generation is currently skipped for now.
    """
    context, plan = _build_context(config=config)
    dataset_dir = _execute_plan(context, config.generation, plan, matrix)

    # Generate comparison data if [test] section is present
    if config.test.solutions_path:
        print("\n=== Skipping comparison split generation (not yet migrated) ===")

    return dataset_dir
