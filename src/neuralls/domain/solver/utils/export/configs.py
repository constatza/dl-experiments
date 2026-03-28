"""Configuration export utilities for solver data.

This module provides functions for saving solver, data, and experiment
configurations to TOML files for reproducibility.

All functions are I/O actions with clear contracts.
"""

from pathlib import Path
from typing import Any

import tomli_w

from neuralls.domain.solver.models.config import SolverConfig

from .serialization import sanitize_for_toml


def save_solver_config(
    output_path: Path,
    config: SolverConfig,
) -> None:
    """Save solver configuration to TOML file.

    Creates TOML file with [solver] table containing all configuration fields.

    Args:
        output_path: Path to output TOML file.
        config: SolverConfig dataclass instance.

    Example:
        >>> from pathlib import Path
        >>> from neuralls.domain.solver.models.config import SolverConfig
        >>> from neuralls.domain.solver.monitoring.trace_mode import TraceMode
        >>> config = SolverConfig(
        ...     algorithm="preconditioned_cg",
        ...     rtol=1e-6,
        ...     atol=1e-10,
        ...     maxiter=1000,
        ...     trace_mode=TraceMode.MINIMAL,
        ... )
        >>> save_solver_config(Path("/tmp/solver.toml"), config)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {"solver": sanitize_for_toml(config)}

    with output_path.open("wb") as f:
        tomli_w.dump(config_dict, f)


def save_full_config(
    output_path: Path,
    solver_config: SolverConfig,
    data_config: dict[str, Any] | None = None,
    experiment_config: dict[str, Any] | None = None,
) -> None:
    """Save complete configuration (solver + data + experiment) to TOML.

    Creates TOML file with separate tables for each configuration type:
    - [solver]: Solver parameters (algorithm, tolerances, etc.)
    - [data]: Data parameters (matrix type, size, etc.) - optional
    - [experiment]: Experiment metadata (project, purpose, etc.) - optional

    Args:
        output_path: Path to output TOML file.
        solver_config: SolverConfig dataclass instance.
        data_config: Optional data parameters dict.
        experiment_config: Optional experiment metadata dict.

    Example:
        >>> from pathlib import Path
        >>> from neuralls.domain.solver.models.config import SolverConfig
        >>> from neuralls.domain.solver.monitoring.trace_mode import TraceMode
        >>> solver_cfg = SolverConfig(
        ...     algorithm="preconditioned_cg",
        ...     rtol=1e-6,
        ...     atol=1e-10,
        ...     maxiter=1000,
        ...     trace_mode=TraceMode.MINIMAL,
        ... )
        >>> data_cfg = {"matrix_type": "tridiagonal", "size": 100}
        >>> exp_cfg = {"project": "benchmarks", "purpose": "testing"}
        >>> save_full_config(
        ...     Path("/tmp/config.toml"),
        ...     solver_cfg,
        ...     data_cfg,
        ...     exp_cfg,
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "solver": sanitize_for_toml(solver_config),
    }

    if data_config is not None:
        config_dict["data"] = sanitize_for_toml(data_config)

    if experiment_config is not None:
        config_dict["experiment"] = sanitize_for_toml(experiment_config)

    with output_path.open("wb") as f:
        tomli_w.dump(config_dict, f)
