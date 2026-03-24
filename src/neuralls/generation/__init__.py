"""Data generation module for synthetic linear system datasets.

This module provides a complete framework for generating synthetic training data
for linear system solvers. It supports multiple generation strategies (normal,
krylov, residual traces, error traces, eigenvector-based) that can be mixed
according to user-specified proportions.

Public API:
    generate_mixture: Main orchestration function for mixed-strategy generation
    run_generation: Registry-based strategy dispatcher
    Data types: StrategyOutput, ArchiveData, GeneratedSamples
    Helpers: rng_from_seed, rounded_counts

Architecture:
    - types: Immutable data structures
    - helpers: Pure utility functions
    - trace_utils: Trace manipulation
    - orchestration: High-level workflow (generate_mixture)
    - strategies: Individual generation strategies (registry pattern)
    - runner: Strategy registry and dispatcher

Usage:
    >>> from neuralls.generation import generate_mixture
    >>> X, Y, res_traces, err_traces = generate_mixture(
    ...     A, b,
    ...     mix={"normal": 1.0, "krylov": 1.0},
    ...     total=100,
    ...     seed=42
    ... )
"""

from __future__ import annotations

from .orchestration import generate_mixture, build_dataset
from .types import StrategyOutput, ArchiveData, GeneratedSamples
from .data_types import NormalizeType
from .helpers import rng_from_seed, rounded_counts, select_archive_files
from .runner import run_generation
from .config_processing import process_config
from .plan import GenerationPlan, StrategySpec, parse_generation_plan

# SOLID Architecture (Phases 1-2 complete)
from . import providers
from . import transforms

# Import strategies to trigger registration
from . import strategies

__all__ = [
    # Main API
    "generate_mixture",
    "build_dataset",
    "run_generation",
    "process_config",
    "parse_generation_plan",
    # Data types
    "StrategyOutput",
    "ArchiveData",
    "GeneratedSamples",
    "GenerationPlan",
    "StrategySpec",
    "NormalizeType",
    # Helpers
    "rng_from_seed",
    "rounded_counts",
    "select_archive_files",
    # SOLID Components (Phase 1-2)
    "providers",
    "transforms",
    # Strategies (for registration)
    "strategies",
]

__version__ = "2.0.0"
__author__ = "neuralls Contributors"
__description__ = "SOLID-compliant data generation framework"

# TODO(repo-split): reduce cross-package dependencies before considering a
# standalone generation package. This layer still relies on solver tracing,
# normalization trace containers, shared constants, and dataset I/O helpers, so
# it is not as extraction-ready as `neuralls.solver`.
