"""Generation context and plan assembly for the generate workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from neuralls.domain.generation.data_types import NormalizeType
from neuralls.domain.generation.plan import (
    GenerationPlan,
    StrategySpec,
    canonicalize_strategy_name,
    plan_from_specs,
)
from neuralls.domain.generation.source_streams import EnumerateBy
from neuralls.platform.config.models.data_models import DataConfigFile, GenerationConfig


@dataclass(frozen=True)
class DataGenerationContext:
    """Immutable context assembled from config inputs.

    Attributes:
        matrix_path: Path to the system matrix file or glob.
        rhs_path: Optional path to the RHS vector file or glob.
        solutions_path: Optional path to a solutions archive glob.
        solution_path: Optional path to a single solution file.
        sample_id_regex: Optional regex for extracting sample IDs.
        enumerate_by: Optional enumeration strategy for archive sources.
        dataset_dir: Target directory for the generated dataset.
        normalize: Normalization strategy applied to each sample.
        seed: Random seed for reproducibility.
        shuffle: Whether to shuffle samples after generation.
        replacement: Whether to sample with replacement.
        parameters_paths: Tuple of additional parameter file paths.
        dataset_format: Storage format — currently only ``"zarr_dense"``.
    """

    matrix_path: str
    rhs_path: str | None
    solutions_path: str | None
    solution_path: str | None
    sample_id_regex: str | None
    enumerate_by: EnumerateBy | None
    dataset_dir: Path
    normalize: NormalizeType
    seed: int
    shuffle: bool
    replacement: bool
    parameters_paths: tuple[str, ...] = ()
    dataset_format: Literal["zarr_dense"] = "zarr_dense"


def _plan_from_generation_config(
    generation_cfg: GenerationConfig,
) -> GenerationPlan:
    """Translate a validated GenerationConfig into a GenerationPlan.

    Converts Pydantic StrategyConfig objects directly to StrategySpec objects,
    bypassing raw-dict re-parsing.

    Args:
        generation_cfg: Validated generation configuration.

    Returns:
        GenerationPlan with resolved strategy specs.

    Raises:
        ValueError: If no strategy entries are declared.
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


def _build_context(
    *,
    config: DataConfigFile,
) -> tuple[DataGenerationContext, GenerationPlan]:
    """Assemble generation context and plan from a fully resolved DataConfigFile.

    Args:
        config: Loaded and resolved data config (output.data_dir must be set).

    Returns:
        (DataGenerationContext, GenerationPlan) pair.

    Raises:
        ValueError: If output.data_dir is None, matrix_path is missing, or
            normalize is a bare bool.
    """
    if config.output.data_dir is None:
        raise ValueError("output.data_dir must be resolved before process_config().")
    dataset_dir = config.output.data_dir / config.id

    matrix_path = config.source.matrix_path
    if not matrix_path:
        raise ValueError("Missing 'source.matrix_path' in config")

    normalize_value = config.generation.normalize
    if isinstance(normalize_value, bool):
        raise ValueError(
            f"Invalid normalize value in config: {normalize_value} (bool). "
            "The 'normalize' parameter expects one of "
            "'spectral', 'matrix', 'rhs', 'diagonal', or 'none'."
        )
    normalize = cast(NormalizeType, str(normalize_value))

    plan = _plan_from_generation_config(config.generation)

    return DataGenerationContext(
        matrix_path=matrix_path,
        rhs_path=config.source.rhs_path,
        solutions_path=config.source.solutions_path,
        solution_path=config.source.solution_path,
        sample_id_regex=config.source.sample_id_regex,
        enumerate_by=config.source.enumerate_by,
        dataset_dir=dataset_dir,
        normalize=normalize,
        seed=config.generation.seed,
        shuffle=config.generation.shuffle,
        replacement=config.generation.replacement,
        parameters_paths=config.source.parameters_paths,
        dataset_format=config.output.dataset_format,
    ), plan
