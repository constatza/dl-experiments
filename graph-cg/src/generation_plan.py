"""Helpers for parsing data generation strategy plans from configuration files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .constants import ConfigKeys


def _canonicalize_strategy_name(raw_name: str) -> str:
    """Normalize strategy identifiers while preserving legacy aliases."""

    normalized = raw_name.strip().lower().replace("-", "_")
    if normalized in {"rhs_archive", "rhs_bank", "rhs_repository"}:
        return ConfigKeys.TYPE_RHS_ARCHIVE
    if normalized in {"solution_archive", "solution_bank", "solution_repository"}:
        return ConfigKeys.TYPE_SOLUTION_ARCHIVE
    return normalized


@dataclass(frozen=True)
class StrategySpec:
    """Immutable description of a single data-generation strategy."""

    raw_name: str
    canonical_name: str
    percentage: float
    options: dict[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(self.percentage):
            raise ValueError(
                f"Strategy '{self.raw_name}' has a non-finite percentage: {self.percentage}"
            )
        if self.percentage <= 0.0:
            raise ValueError(
                f"Strategy '{self.raw_name}' must have a positive percentage, got {self.percentage}"
            )


@dataclass(frozen=True)
class GenerationPlan:
    """Parsed representation of the generation strategies declared in config."""

    strategies: dict[str, StrategySpec]

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ValueError("At least one generation strategy must be configured")
        total = self.total_percentage
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"generation.strategy percentages must sum to 1.0 (found {total:.6f})"
            )

    @property
    def total_percentage(self) -> float:
        return sum(spec.percentage for spec in self.strategies.values())

    @property
    def rhs_archive(self) -> StrategySpec | None:
        return self.strategies.get(ConfigKeys.TYPE_RHS_ARCHIVE)

    @property
    def solution_archive(self) -> StrategySpec | None:
        return self.strategies.get(ConfigKeys.TYPE_SOLUTION_ARCHIVE)

    @property
    def synthetic(self) -> dict[str, StrategySpec]:
        """Return strategies that correspond to synthetic data generation."""

        excluded = {ConfigKeys.TYPE_RHS_ARCHIVE, ConfigKeys.TYPE_SOLUTION_ARCHIVE}
        return {name: spec for name, spec in self.strategies.items() if name not in excluded}


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ValueError("'generation.strategy' must be an array of tables")


def parse_generation_plan(generation_cfg: Mapping[str, Any]) -> GenerationPlan:
    """Parse the ``[[generation.strategy]]`` entries into a structured plan."""

    raw_entries = generation_cfg.get(ConfigKeys.STRATEGY)
    if raw_entries is None:
        raise ValueError("Missing '[[generation.strategy]]' entries in config")

    entries = _ensure_sequence(raw_entries)
    if not entries:
        raise ValueError("At least one '[[generation.strategy]]' block is required")

    aggregated: dict[str, StrategySpec] = {}

    for index, raw_spec in enumerate(entries):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"'generation.strategy' entry at index {index} must be a table"
            )

        raw_name = raw_spec.get(ConfigKeys.NAME)
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                f"'generation.strategy' entry at index {index} is missing a non-empty 'name'"
            )

        if ConfigKeys.PERCENTAGE not in raw_spec:
            raise ValueError(
                f"'generation.strategy' entry '{raw_name}' is missing 'percentage'"
            )

        try:
            percentage = float(raw_spec[ConfigKeys.PERCENTAGE])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'generation.strategy' entry '{raw_name}' has non-numeric 'percentage'"
            ) from exc

        canonical = _canonicalize_strategy_name(raw_name)
        options = {
            key: value
            for key, value in raw_spec.items()
            if key not in {ConfigKeys.PERCENTAGE, ConfigKeys.NAME}
        }

        existing = aggregated.get(canonical)
        if existing is None:
            aggregated[canonical] = StrategySpec(
                raw_name=raw_name,
                canonical_name=canonical,
                percentage=percentage,
                options=dict(options),
            )
            continue

        merged_options = {**existing.options, **options}
        aggregated[canonical] = StrategySpec(
            raw_name=raw_name,
            canonical_name=canonical,
            percentage=existing.percentage + percentage,
            options=merged_options,
        )

    return GenerationPlan(strategies=aggregated)
