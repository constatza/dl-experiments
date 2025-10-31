#!/usr/bin/env python3
"""Unified data processing interface across RHS archives, synthetic generation, and solution archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import (
    ConfigKeys,
    ConfigSections,
    DEFAULT_KRYLOV_ITERATIONS,
    DEFAULT_NORMALIZE,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESIDUAL_TRACE_ITERS,
    DEFAULT_SHUFFLE,
)
from .generation_plan import GenerationPlan, StrategySpec, parse_generation_plan
from .paths import ProjectRoots, FlowPaths, DataPaths, parse_flow_keys
from .data_collection import collect_case, generate_case, solution_archive_case
from .sample_builders import build_solution_archive_samples


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

    flow_id, dataset_id = parse_flow_keys(config, config_path=config_path)
    roots = ProjectRoots.from_overrides(
        project_root=output_cfg.get("project_root"),
        processed_root=output_cfg.get(ConfigKeys.PROCESSED_DIR),
        output_root=output_cfg.get("output_root"),
        figures_root=output_cfg.get("figures_root"),
    )
    flow_paths = FlowPaths(flow_id=flow_id, roots=roots)
    data_paths = DataPaths(flow=flow_paths, dataset_id=dataset_id)

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
            "'spectral', 'matrix', 'rhs', or 'none'."
        )
    normalize = str(normalize_value)

    plan = parse_generation_plan(generation_cfg)

    return DataGenerationContext(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        dataset_dir=data_paths.base_dir,
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

    representative = candidates[0]
    raw_samples = build_solution_archive_samples(
        matrix_path=Path(matrix_path),
        solution_files=[representative],
        shuffle=False,
        seed=None,
    )
    rhs_vector = raw_samples.rhs[0]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "auto-mother-rhs.npy"
    np.save(output_path, rhs_vector.astype(np.float64, copy=False))
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
            raise FileNotFoundError(
                f"provide_rhs path does not exist: {path}"
            )
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
        ConfigKeys.SEED, context.generation_cfg.get(ConfigKeys.SEED, DEFAULT_RANDOM_SEED)
    )

    return solution_archive_case(
        matrix_path=context.matrix_path,
        solutions_path=resolved_solutions_path,
        dataset_dir=context.dataset_dir,
        normalize=context.normalize,
        shuffle=bool(shuffle_value),
        seed=_coerce_optional_int(seed_value),
    )


def _execute_synthetic_generation(
    context: DataGenerationContext,
    synthetic_strategies: Mapping[str, StrategySpec],
    rhs_archive_strategy: StrategySpec | None,
    solution_archive_strategy: StrategySpec | None,
) -> Path:
    generation_cfg = context.generation_cfg

    num_samples_raw = generation_cfg.get(ConfigKeys.NUM_SAMPLES, DEFAULT_NUM_SAMPLES)
    num_samples = int(num_samples_raw)
    krylov_iters = int(generation_cfg.get(ConfigKeys.KRYLOV_ITERS, DEFAULT_KRYLOV_ITERATIONS))
    residual_iters = int(
        generation_cfg.get(ConfigKeys.RESIDUAL_ITERS, DEFAULT_RESIDUAL_TRACE_ITERS)
    )

    seed_value = generation_cfg.get(ConfigKeys.SEED, DEFAULT_RANDOM_SEED)
    if seed_value is None:
        seed_value = DEFAULT_RANDOM_SEED
    seed = int(seed_value)
    shuffle_value = generation_cfg.get(ConfigKeys.SHUFFLE, DEFAULT_SHUFFLE)

    mix: dict[str, float] = {}
    strategy_overrides: dict[str, dict[str, Any]] = {}
    for name, spec in synthetic_strategies.items():
        mix[name] = spec.percentage
        if spec.options:
            strategy_overrides[name] = dict(spec.options)

    solution_archive_options: dict[str, Any] | None = None
    if solution_archive_strategy is not None:
        mix[ConfigKeys.TYPE_SOLUTION_ARCHIVE] = solution_archive_strategy.percentage
        solution_archive_options = dict(solution_archive_strategy.options)
        if ConfigKeys.SOLUTIONS_GLOB not in solution_archive_options:
            fallback = context.source_cfg.get(ConfigKeys.SOLUTIONS_PATH)
            if isinstance(fallback, str) and fallback:
                solution_archive_options[ConfigKeys.SOLUTIONS_GLOB] = fallback
        if ConfigKeys.SHUFFLE not in solution_archive_options:
            solution_archive_options[ConfigKeys.SHUFFLE] = context.generation_cfg.get(
                ConfigKeys.SHUFFLE, DEFAULT_SHUFFLE
            )
        if ConfigKeys.SEED not in solution_archive_options:
            solution_archive_options[ConfigKeys.SEED] = context.generation_cfg.get(
                ConfigKeys.SEED, DEFAULT_RANDOM_SEED
            )

    provide_rhs_option = None
    if solution_archive_options is not None:
        provide_rhs_option = solution_archive_options.pop(ConfigKeys.PROVIDE_RHS, None)

    rhs_archive_glob: str | None = None
    if rhs_archive_strategy is not None:
        mix[ConfigKeys.TYPE_RHS_ARCHIVE] = rhs_archive_strategy.percentage
        rhs_archive_glob = _resolve_rhs_archive_glob(rhs_archive_strategy, context)
        if rhs_archive_glob is None:
            raise ValueError(
                "RHS archive strategy requires 'rhs_glob' within the strategy, "
                f"'{ConfigSections.SOURCE}.{ConfigKeys.RHS_PATH}', "
                f"or 'generation.{ConfigKeys.RHS_ARCHIVE_GLOB}'."
            )

    rhs_source_path = _resolve_rhs_source(
        context=context,
        solution_archive_options=solution_archive_options,
        provide_rhs_option=provide_rhs_option,
    )
    if rhs_source_path is None:
        raise ValueError(
            "Missing 'source.rhs_path' for synthetic data generation. "
            "Provide one in the config or set provide_rhs on the solution_archive strategy."
        )

    return generate_case(
        matrix_path=context.matrix_path,
        rhs_path=rhs_source_path,
        num_samples=num_samples,
        dataset_dir=context.dataset_dir,
        mix=mix,
        krylov_iters=krylov_iters,
        residual_iters=residual_iters,
        seed=seed,
        normalize=context.normalize,
        shuffle=bool(shuffle_value),
        rhs_archive_glob=rhs_archive_glob,
        solution_archive_options=solution_archive_options,
        strategy_overrides=strategy_overrides,
    )


def _execute_rhs_archive_only(
    context: DataGenerationContext,
    strategy: StrategySpec,
) -> Path:
    rhs_glob = _resolve_rhs_archive_glob(strategy, context)
    if rhs_glob is None:
        raise ValueError(
            "RHS archive strategy requires either 'rhs_glob' in the strategy "
            f"or '{ConfigSections.SOURCE}.{ConfigKeys.RHS_PATH}'."
        )

    collection_kwargs = _merge_rhs_archive_options(strategy, context)

    return collect_case(
        matrix_path=context.matrix_path,
        rhs_path=rhs_glob,
        dataset_dir=context.dataset_dir,
        normalize=context.normalize,
        **collection_kwargs,
    )


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


def process_config(config: dict[str, Any], config_path: Path | str | None = None) -> Path:
    """Process a data config and execute the declared generation plan."""
    context, plan = _build_context(config=config, config_path=config_path)
    return _execute_plan(context, plan)
