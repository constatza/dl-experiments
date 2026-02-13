#!/usr/bin/env python3
"""Unified data processing interface across RHS archives, synthetic generation, and solution archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping


import numpy as np

from ..constants import (
    ConfigKeys,
    ConfigSections,
    DEFAULT_NORMALIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SHUFFLE,
)
from .plan import GenerationPlan, StrategySpec, parse_generation_plan
from ..io.base import load_matrix
from ..constants import DEFAULT_PROCESSED_DATA_DIR
from .orchestration import build_dataset
from .runner import run_generation


@dataclass(frozen=True)
class DataGenerationContext:
    """Immutable context assembled from config inputs."""

    matrix_path: str
    rhs_path: str | None
    dataset_dir: Path
    normalize: str
    source_cfg: Mapping[str, Any]
    generation_cfg: Mapping[str, Any]


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    """Convert arbitrary values to an immutable mapping."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _build_context(
    *,
    config: Mapping[str, Any],
    config_path: Path | str | None,
) -> tuple[DataGenerationContext, GenerationPlan]:
    """Assemble generation context, plan, and dataset directory."""
    source_cfg = _coerce_mapping(config.get(ConfigSections.SOURCE, {}))
    generation_cfg = _coerce_mapping(config.get(ConfigSections.GENERATION, {}))
    output_cfg = _coerce_mapping(config.get(ConfigSections.OUTPUT, {}))

    # Extract dataset_id from config or filename
    if config_path:
        dataset_id = Path(config_path).stem
    else:
        flow_section = config.get("flow", {})
        dataset_id = flow_section.get("dataset", "default")

    # Resolve processed data directory
    processed_dir_str = output_cfg.get(ConfigKeys.PROCESSED_DIR)
    if processed_dir_str:
        processed_root = Path(processed_dir_str)
    else:
        processed_root = DEFAULT_PROCESSED_DATA_DIR

    dataset_dir = processed_root / dataset_id

    matrix_path = _coerce_optional_str(source_cfg.get(ConfigKeys.MATRIX_PATH))
    if not matrix_path:
        raise ValueError(
            f"Missing '{ConfigSections.SOURCE}.{ConfigKeys.MATRIX_PATH}' in config"
        )

    rhs_path = _coerce_optional_str(source_cfg.get(ConfigKeys.RHS_PATH))
    normalize_value = generation_cfg.get(ConfigKeys.NORMALIZE, DEFAULT_NORMALIZE)
    if isinstance(normalize_value, bool):
        raise ValueError(
            f"Invalid normalize value in config: {normalize_value} (bool). "
            f"The 'normalize' parameter expects one of "
            "'spectral', 'matrix', 'rhs', 'diagonal', or 'none'."
        )
    normalize = str(normalize_value)

    plan = parse_generation_plan(generation_cfg)

    return DataGenerationContext(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        dataset_dir=dataset_dir,
        normalize=normalize,
        source_cfg=source_cfg,
        generation_cfg=generation_cfg,
    ), plan


def _merge_rhs_archive_options(
    strategy: StrategySpec | None,
    context: DataGenerationContext,
) -> dict[str, Any]:
    """Combine generation-level options with strategy-specific overrides."""
    relevant_keys = ("solve_systems", "cg_tolerance", "cg_max_iters")
    merged: dict[str, Any] = {}

    generation_cfg = context.generation_cfg

    for key in relevant_keys:
        if key in generation_cfg:
            merged[key] = generation_cfg[key]
        if strategy and key in strategy.options:
            merged[key] = strategy.options[key]

    return merged


def _resolve_rhs_archive_glob(
    strategy: StrategySpec | None,
    context: DataGenerationContext,
) -> str | None:
    """Resolve the RHS glob path to use for RHS archive samples."""
    candidates: tuple[Any, ...] = (
        strategy.options.get(ConfigKeys.RHS_GLOB) if strategy else None,
        strategy.options.get(ConfigKeys.RHS_PATH) if strategy else None,
        context.generation_cfg.get(ConfigKeys.RHS_ARCHIVE_GLOB),
        context.source_cfg.get(ConfigKeys.RHS_PATH),
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
        strategy.options.get(ConfigKeys.SOLUTIONS_GLOB),
        strategy.options.get(ConfigKeys.SOLUTIONS_PATH),
        context.source_cfg.get(ConfigKeys.SOLUTIONS_PATH),
    )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    return None


def _coerce_optional_int(value: Any) -> int | None:
    """Convert optional numeric values to ``int`` while preserving ``None``."""
    if value is None:
        return None
    return int(value)


def _derive_rhs_from_solution_archive(
    *,
    matrix_path: str,
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
    matrix = load_matrix(Path(matrix_path))

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
        solutions_glob = solution_archive_options.get(ConfigKeys.SOLUTIONS_GLOB)
        if not isinstance(solutions_glob, str) or not solutions_glob:
            raise ValueError(
                "provide_rhs=True requires 'solutions_glob' for the solution_archive strategy."
            )
        generated_path = _derive_rhs_from_solution_archive(
            matrix_path=context.matrix_path,
            solutions_glob=solutions_glob,
            dataset_dir=context.dataset_dir,
        )
        return str(generated_path)

    return None


def _execute_solution_archive(
    context: DataGenerationContext,
    strategy: StrategySpec,
) -> Path:
    """Execute solution archive using new build_dataset() with solution_archive strategy."""
    resolved_solutions_path = _resolve_solution_archive_path(strategy, context)
    if resolved_solutions_path is None:
        raise ValueError(
            "Solution archive strategy requires either "
            "a 'solutions_glob' field within [[generation.strategy]] name='solution_archive' or "
            f"'{ConfigSections.SOURCE}.{ConfigKeys.SOLUTIONS_PATH}'."
        )

    shuffle_value = strategy.options.get(
        ConfigKeys.SHUFFLE,
        context.generation_cfg.get(ConfigKeys.SHUFFLE, DEFAULT_SHUFFLE),
    )
    seed_value = strategy.options.get(
        ConfigKeys.SEED,
        context.generation_cfg.get(ConfigKeys.SEED, DEFAULT_RANDOM_SEED),
    )

    # Use new build_dataset with solution_archive strategy
    from .data_types import NormalizeType
    from typing import cast

    dataset_path = build_dataset(
        matrix_path=context.matrix_path,
        dataset_dir=str(context.dataset_dir),
        counts={"solution_archive": strategy.samples},
        normalize=cast(NormalizeType, context.normalize),
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
) -> Path:
    """Execute mixed generation using new build_dataset() with all strategies."""
    generation_cfg = context.generation_cfg

    seed_value = generation_cfg.get(ConfigKeys.SEED, DEFAULT_RANDOM_SEED)
    if seed_value is None:
        seed_value = DEFAULT_RANDOM_SEED
    seed = int(seed_value)
    shuffle_value = generation_cfg.get(ConfigKeys.SHUFFLE, DEFAULT_SHUFFLE)

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
        solutions_glob = _resolve_solution_archive_path(
            solution_archive_strategy, context
        )
        if solutions_glob:
            strategy_overrides["solution_archive"] = {
                "solutions_glob": solutions_glob,
                "shuffle": solution_archive_strategy.options.get(
                    ConfigKeys.SHUFFLE,
                    generation_cfg.get(ConfigKeys.SHUFFLE, DEFAULT_SHUFFLE),
                ),
                "seed": solution_archive_strategy.options.get(
                    ConfigKeys.SEED, seed_value
                ),
            }

    # Add RHS archive strategy
    if rhs_archive_strategy is not None:
        counts["rhs_archive"] = rhs_archive_strategy.samples
        rhs_archive_glob = _resolve_rhs_archive_glob(rhs_archive_strategy, context)
        if rhs_archive_glob is None:
            raise ValueError(
                "RHS archive strategy requires 'rhs_glob' within the strategy, "
                f"'{ConfigSections.SOURCE}.{ConfigKeys.RHS_PATH}', "
                f"or 'generation.{ConfigKeys.RHS_ARCHIVE_GLOB}'."
            )

        rhs_archive_opts = _merge_rhs_archive_options(rhs_archive_strategy, context)
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
    )

    # Use new unified build_dataset()
    from .data_types import NormalizeType
    from typing import cast

    dataset_path = build_dataset(
        matrix_path=context.matrix_path,
        dataset_dir=str(context.dataset_dir),
        counts=counts,
        rhs_path=rhs_source_path,
        normalize=cast(NormalizeType, context.normalize),
        shuffle=bool(shuffle_value),
        seed=seed,
        strategy_overrides=strategy_overrides,
    )
    return Path(dataset_path)


def _execute_rhs_archive_only(
    context: DataGenerationContext,
    strategy: StrategySpec,
) -> Path:
    """Execute RHS archive using new build_dataset() with rhs_archive strategy."""
    rhs_glob = _resolve_rhs_archive_glob(strategy, context)
    if rhs_glob is None:
        raise ValueError(
            "RHS archive strategy requires either 'rhs_glob' in the strategy "
            f"or '{ConfigSections.SOURCE}.{ConfigKeys.RHS_PATH}'."
        )

    collection_kwargs = _merge_rhs_archive_options(strategy, context)

    seed_value = context.generation_cfg.get(ConfigKeys.SEED, DEFAULT_RANDOM_SEED)

    # Use new build_dataset with rhs_archive strategy
    from .data_types import NormalizeType
    from typing import cast

    dataset_path = build_dataset(
        matrix_path=context.matrix_path,
        dataset_dir=str(context.dataset_dir),
        counts={"rhs_archive": strategy.samples},
        normalize=cast(NormalizeType, context.normalize),
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
    plan: GenerationPlan,
) -> Path:
    rhs_archive_strategy = plan.rhs_archive
    solution_archive_strategy = plan.solution_archive
    synthetic_strategies = plan.synthetic

    has_solution_archive = solution_archive_strategy is not None
    has_rhs_archive = rhs_archive_strategy is not None
    has_synthetic_strategies = bool(synthetic_strategies)

    if has_solution_archive and not (has_rhs_archive or has_synthetic_strategies):
        return _execute_solution_archive(context, solution_archive_strategy)

    if has_synthetic_strategies or has_solution_archive:
        return _execute_synthetic_generation(
            context=context,
            synthetic_strategies=synthetic_strategies,
            rhs_archive_strategy=rhs_archive_strategy,
            solution_archive_strategy=solution_archive_strategy,
        )

    if not has_rhs_archive:
        raise ValueError("No generation strategies configured")

    return _execute_rhs_archive_only(context, rhs_archive_strategy)


def process_config(
    config: Mapping[str, Any], config_path: Path | str | None = None
) -> Path:
    """Process a data config and execute the declared generation plan.

    After generating the main dataset, this also checks for a [test] section
    and generates comparison.npz if test solutions are specified.
    """
    context, plan = _build_context(config=config, config_path=config_path)
    dataset_dir = _execute_plan(context, plan)

    # Generate comparison data if [test] section is present
    test_config = config.get("test", {})
    if test_config:
        solutions_glob = test_config.get("solutions_glob")
        if solutions_glob:
            print("\n=== Skipping comparison.npz generation (not yet migrated) ===")
            # persist_comparison_samples(
            #     dataset_dir=dataset_dir,
            #     test_solutions_glob=solutions_glob,
            #     max_samples=max_samples,
            # )

    return dataset_dir
