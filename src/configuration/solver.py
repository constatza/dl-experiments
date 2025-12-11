"""Solver parameters extraction and management.

This module provides dataclasses and functions for managing CG solver parameters
decoupled from model architecture. Parameters can be loaded from solver configs
or extracted from legacy settings objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING, Iterable

from ..constants import (
    DEFAULT_ATOL,
    DEFAULT_BETA_MAX,
    DEFAULT_CURVATURE_EPSILON,
    DEFAULT_DIVERGENCE_FACTOR,
    DEFAULT_M_MAX,
    DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
    DEFAULT_RTOL,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..solver.info import SolverResult


@dataclass(frozen=True)
class GeneralSolverParams:
    """Global solver defaults used across all solver variants."""

    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    max_iterations: int = 100
    stopping_criterion: str = "residual_norm"
    normalize_system: str | bool = "matrix"
    matrix_path: str | None = None
    rhs_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolverSpec:
    """Individual solver/preconditioner entry."""

    name: str
    type: str
    args: dict[str, Any]


@dataclass(frozen=True)
class SolverParams:
    """Legacy single-solver parameters retained for compatibility."""

    solver_type: str
    rtol: float
    atol: float
    max_iterations: int
    normalize_system: str | bool
    stopping_criterion: str
    m_max: int
    beta_max: float
    eps_curv: float
    eps_breakdown: float
    m_replacement: int
    gamma_div: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_solver_specs(entries: Iterable[dict[str, Any]] | None) -> list[SolverSpec]:
    specs: list[SolverSpec] = []
    for idx, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", f"solver_{idx}"))
        solver_type = str(entry.get("type", "flexible_cg"))
        args = {k: v for k, v in entry.items() if k not in {"name", "type"}}
        specs.append(SolverSpec(name=name, type=solver_type, args=args))
    return specs


def parse_solver_config(
    solver_config: dict[str, Any],
) -> tuple[GeneralSolverParams, list[SolverSpec]]:
    """Parse solver configuration with new schema; fall back to legacy [solver]."""
    # New schema
    if "general" in solver_config or "solvers" in solver_config:
        general_section = solver_config.get("general", {})
        data_gen_section = solver_config.get("data_generation", {})

        general = GeneralSolverParams(
            rtol=float(general_section.get("rtol", DEFAULT_RTOL)),
            atol=float(general_section.get("atol", DEFAULT_ATOL)),
            max_iterations=int(general_section.get("max_iterations", 100)),
            stopping_criterion=str(
                general_section.get("stopping_criterion", "residual_norm")
            ),
            normalize_system=data_gen_section.get("normalize", "matrix"),
            matrix_path=general_section.get("matrix"),
            rhs_path=general_section.get("rhs"),
        )
        solvers = _coerce_solver_specs(solver_config.get("solvers"))
        return general, solvers

    # Legacy [solver] schema support for backward compatibility
    solver_section = solver_config.get("solver", {})
    data_gen_section = solver_config.get("data_generation", {})
    general = GeneralSolverParams(
        rtol=float(solver_section.get("rtol", solver_section.get("tolerance", DEFAULT_RTOL))),
        atol=float(solver_section.get("atol", DEFAULT_ATOL)),
        max_iterations=int(solver_section.get("max_iterations", 100)),
        stopping_criterion=str(solver_section.get("stopping_criterion", "residual_norm")),
        normalize_system=data_gen_section.get("normalize", "matrix"),
        matrix_path=solver_section.get("matrix"),
        rhs_path=solver_section.get("rhs"),
    )
    solver_type = str(solver_section.get("solver_type", "flexible_cg"))
    legacy_args = {
        "m_max": solver_section.get("m_max", DEFAULT_M_MAX),
        "beta_max": solver_section.get("beta_max", DEFAULT_BETA_MAX),
        "eps_curv": solver_section.get("eps_curv", DEFAULT_CURVATURE_EPSILON),
        "eps_breakdown": solver_section.get("eps_breakdown", 1e-14),
        "m_replacement": solver_section.get("m_replacement", DEFAULT_RESIDUAL_REPLACEMENT_FREQ),
        "gamma_div": solver_section.get("gamma_div", DEFAULT_DIVERGENCE_FACTOR),
    }
    solvers = [SolverSpec(name=solver_type, type=solver_type, args=legacy_args)]
    return general, solvers


def extract_solver_params_from_config(
    solver_config: dict[str, Any],
) -> SolverParams:
    """Extract a single SolverParams for legacy callers from new config schema."""
    general, solvers = parse_solver_config(solver_config)
    first_solver = solvers[0] if solvers else None
    args = first_solver.args if first_solver else {}

    return SolverParams(
        solver_type=(first_solver.type if first_solver else "flexible_cg"),
        rtol=general.rtol,
        atol=general.atol,
        max_iterations=general.max_iterations,
        normalize_system=general.normalize_system,
        stopping_criterion=general.stopping_criterion,
        m_max=int(args.get("m_max", DEFAULT_M_MAX)),
        beta_max=float(args.get("beta_max", DEFAULT_BETA_MAX)),
        eps_curv=float(args.get("eps_curv", DEFAULT_CURVATURE_EPSILON)),
        eps_breakdown=float(args.get("eps_breakdown", 1e-14)),
        m_replacement=int(args.get("m_replacement", DEFAULT_RESIDUAL_REPLACEMENT_FREQ)),
        gamma_div=float(args.get("gamma_div", DEFAULT_DIVERGENCE_FACTOR)),
    )


def get_solver_params(settings: Any) -> SolverParams:
    """Get solver parameters from GeneralSettings object.

    Provides backwards compatibility by reading from settings.EXTRAS.solver
    or settings.EXTRAS.solver_config if available, otherwise returns
    sensible defaults.

    Args:
        settings: GeneralSettings object (from dlkit).

    Returns:
        SolverParams: Extracted or defaulted solver parameters.
    """
    extras = getattr(settings, "EXTRAS", None)
    if extras is None:
        return extract_solver_params_from_config({})

    # EXTRAS might be a Pydantic model, convert to dict
    if hasattr(extras, "model_dump"):
        extras_dict = extras.model_dump()
    elif isinstance(extras, dict):
        extras_dict = extras
    else:
        extras_dict = {}

    # Prefer solver_config if provided, otherwise fallback to legacy keys
    if "solver_config" in extras_dict:
        return extract_solver_params_from_config(extras_dict["solver_config"])
    return extract_solver_params_from_config(extras_dict)


def create_solver_from_params(
    params: SolverParams,
    preconditioner: Callable[[NDArray], NDArray] | None = None,
) -> Callable[[NDArray, NDArray, NDArray | None], tuple[NDArray, "SolverResult"]]:
    """Create a solver function from SolverParams configuration.

    Args:
        params: Solver parameters extracted from config.
        preconditioner: Optional preconditioner function (required for PCG).

    Returns:
        Solver function with signature (A, b, x0) -> (x, result).

    Raises:
        ValueError: If solver_type is unknown or required preconditioner is missing.

    Examples:
        >>> params = extract_solver_params_from_config(config)
        >>> solver_fn = create_solver_from_params(params)
        >>> x, result = solver_fn(A, b, x0=None)
    """
    from ..solver import flexible_cg, preconditioned_cg

    # Prepare common parameters
    common_kwargs = {
        "rtol": params.rtol,
        "atol": params.atol,
        "max_iterations": params.max_iterations,
        "eps_curv": params.eps_curv,
        "eps_breakdown": params.eps_breakdown,
        "m_replacement": params.m_replacement,
        "gamma_div": params.gamma_div,
        "stopping_criterion": params.stopping_criterion,
    }

    if params.solver_type == "flexible_cg":
        # Flexible CG with orthogonalization
        def solver(
            A: NDArray,
            b: NDArray,
            x0: NDArray | None = None,
        ) -> tuple[NDArray, "SolverResult"]:
            return flexible_cg(
                A,
                b,
                x0,
                preconditioner=preconditioner,
                m_max=params.m_max,
                **common_kwargs,
            )

        return solver

    elif params.solver_type == "preconditioned_cg":
        # Preconditioned CG (requires preconditioner)
        if preconditioner is None:
            raise ValueError(
                "preconditioned_cg requires a preconditioner, but none was provided"
            )

        def solver(
            A: NDArray,
            b: NDArray,
            x0: NDArray | None = None,
        ) -> tuple[NDArray, "SolverResult"]:
            return preconditioned_cg(
                A,
                b,
                x0,
                preconditioner=preconditioner,
                beta_max=params.beta_max,
                **common_kwargs,
            )

        return solver

    else:
        raise ValueError(
            f"Unknown solver_type: {params.solver_type}. "
            f"Must be one of: 'flexible_cg', 'preconditioned_cg'"
        )
