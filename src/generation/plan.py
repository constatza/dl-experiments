"""Count-based generation strategy plan parsing.

This module replaces percentage-based mixing with explicit sample counts.
Each strategy specifies either:
  - samples > 0: Exact count
  - samples = -1: All available (for archives)
  - samples = 0: Skip this strategy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..constants import ConfigKeys


def _canonicalize_strategy_name(raw_name: str) -> str:
    """Normalize strategy identifiers while preserving legacy aliases.

    Args:
        raw_name: Strategy name from config (e.g., "rhs-archive", "RHS_ARCHIVE")

    Returns:
        Canonical name (e.g., "rhs_archive")

    Examples:
        >>> _canonicalize_strategy_name("rhs-archive")
        'rhs_archive'
        >>> _canonicalize_strategy_name("solution_bank")
        'solution_archive'
    """
    normalized = raw_name.strip().lower().replace("-", "_")

    # Legacy alias mapping
    if normalized in {"rhs_bank", "rhs_repository"}:
        return ConfigKeys.TYPE_RHS_ARCHIVE
    if normalized in {"solution_bank", "solution_repository"}:
        return ConfigKeys.TYPE_SOLUTION_ARCHIVE

    return normalized


@dataclass(frozen=True)
class StrategySpec:
    """Immutable specification for a single generation strategy.

    Attributes:
        raw_name: Original name from config
        canonical_name: Normalized name for registry lookup
        samples: Sample count (>0 exact, -1 all, 0 skip)
        options: Additional strategy-specific parameters
    """

    raw_name: str
    canonical_name: str
    samples: int
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate count constraints."""
        if self.samples < -1:
            raise ValueError(
                f"Strategy '{self.raw_name}' has invalid samples count: {self.samples}. "
                f"Must be -1 (all), 0 (skip), or positive integer."
            )


@dataclass(frozen=True)
class GenerationPlan:
    """Parsed count-based generation plan.

    Attributes:
        strategies: Mapping of canonical_name -> StrategySpec

    Properties:
        rhs_archive: RHS archive strategy (if present)
        solution_archive: Solution archive strategy (if present)
        synthetic: All non-archive strategies
    """

    strategies: Mapping[str, StrategySpec]

    def __post_init__(self) -> None:
        """Validate plan has at least one strategy with samples > 0."""
        if not self.strategies:
            raise ValueError("At least one generation strategy must be configured")

        active_strategies = [
            spec for spec in self.strategies.values() if spec.samples != 0
        ]
        if not active_strategies:
            raise ValueError(
                "At least one strategy must have samples > 0 or samples = -1"
            )

    @property
    def rhs_archive(self) -> StrategySpec | None:
        """Get RHS archive strategy if present."""
        return self.strategies.get(ConfigKeys.TYPE_RHS_ARCHIVE)

    @property
    def solution_archive(self) -> StrategySpec | None:
        """Get solution archive strategy if present."""
        return self.strategies.get(ConfigKeys.TYPE_SOLUTION_ARCHIVE)

    @property
    def synthetic(self) -> Mapping[str, StrategySpec]:
        """Get all synthetic (non-archive) strategies."""
        excluded = {ConfigKeys.TYPE_RHS_ARCHIVE, ConfigKeys.TYPE_SOLUTION_ARCHIVE}
        return {
            name: spec
            for name, spec in self.strategies.items()
            if name not in excluded
        }


def _ensure_sequence(value: Any) -> Sequence[Any]:
    """Validate that value is a sequence (not str/bytes).

    Args:
        value: Value to validate

    Returns:
        The value if it's a valid sequence

    Raises:
        ValueError: If value is not a sequence or is str/bytes
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ValueError(
        "'generation.strategy' must be an array of tables ([[generation.strategy]])"
    )


def _parse_strategy_entry(
    raw_spec: Mapping[str, Any],
    index: int,
) -> tuple[str, int, str, Mapping[str, Any]]:
    """Parse a single [[generation.strategy]] entry.

    Args:
        raw_spec: TOML table for this strategy
        index: Index in array (for error messages)

    Returns:
        Tuple of (raw_name, samples, canonical_name, options)

    Raises:
        ValueError: If required fields missing or invalid
    """
    # Validate name field
    raw_name = raw_spec.get(ConfigKeys.NAME)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(
            f"'generation.strategy' entry at index {index} is missing a non-empty 'name'"
        )

    # Validate samples field
    if ConfigKeys.SAMPLES not in raw_spec:
        raise ValueError(
            f"'generation.strategy' entry '{raw_name}' is missing 'samples' field. "
            f"Use samples=N (exact count), samples=-1 (all), or samples=0 (skip)."
        )

    try:
        samples = int(raw_spec[ConfigKeys.SAMPLES])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'generation.strategy' entry '{raw_name}' has non-integer 'samples'"
        ) from exc

    # Validate samples range
    if samples < -1:
        raise ValueError(
            f"'generation.strategy' entry '{raw_name}' has invalid samples={samples}. "
            f"Must be -1 (all), 0 (skip), or positive integer."
        )

    # Canonicalize name
    canonical = _canonicalize_strategy_name(raw_name)

    # Extract options (everything except 'name' and 'samples')
    options = {
        key: value
        for key, value in raw_spec.items()
        if key not in {ConfigKeys.NAME, ConfigKeys.SAMPLES}
    }

    return raw_name, samples, canonical, options


def parse_generation_plan(generation_cfg: Mapping[str, Any]) -> GenerationPlan:
    """Parse [[generation.strategy]] entries into count-based plan.

    Args:
        generation_cfg: The [generation] section from TOML config

    Returns:
        GenerationPlan with all strategies

    Raises:
        ValueError: If config invalid or strategies missing

    Examples:
        >>> config = {
        ...     "strategy": [
        ...         {"name": "random", "samples": 1000},
        ...         {"name": "krylov", "samples": 500, "krylov_iters": 15},
        ...         {"name": "solution_archive", "samples": -1,
        ...          "solutions_glob": "/data/*.txt"},
        ...     ]
        ... }
        >>> plan = parse_generation_plan(config)
        >>> plan.strategies.keys()
        dict_keys(['random', 'krylov', 'solution_archive'])
        >>> plan.solution_archive.samples
        -1
    """
    # Extract strategy array
    raw_entries = generation_cfg.get(ConfigKeys.STRATEGY)
    if raw_entries is None:
        raise ValueError(
            "Missing '[[generation.strategy]]' entries in config. "
            "Add at least one strategy with 'name' and 'samples' fields."
        )

    entries = _ensure_sequence(raw_entries)
    if not entries:
        raise ValueError("At least one '[[generation.strategy]]' block is required")

    # Parse and aggregate strategies
    aggregated: dict[str, StrategySpec] = {}

    for index, raw_spec in enumerate(entries):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"'generation.strategy' entry at index {index} must be a table"
            )

        raw_name, samples, canonical, options = _parse_strategy_entry(raw_spec, index)

        # Handle duplicate strategies: merge counts and options
        existing = aggregated.get(canonical)
        if existing is None:
            aggregated[canonical] = StrategySpec(
                raw_name=raw_name,
                canonical_name=canonical,
                samples=samples,
                options=dict(options),
            )
            continue

        # Merge duplicate entries
        # Counts add (unless one is -1, which means "all available")
        merged_samples = samples
        if existing.samples == -1 or samples == -1:
            # If either is "all", use -1
            merged_samples = -1
        elif existing.samples > 0 and samples > 0:
            # Both positive: add them
            merged_samples = existing.samples + samples
        else:
            # One or both are 0: use max
            merged_samples = max(existing.samples, samples)

        # Options merge (later entries override)
        merged_options = {**existing.options, **options}

        aggregated[canonical] = StrategySpec(
            raw_name=raw_name,
            canonical_name=canonical,
            samples=merged_samples,
            options=merged_options,
        )

    return GenerationPlan(strategies=aggregated)


__all__ = [
    "StrategySpec",
    "GenerationPlan",
    "parse_generation_plan",
]
