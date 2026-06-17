"""Preconditioner creation, scheduling, input binding, and stopping criterion registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from neuralls.composition.preconditioners.factory import (
    PreconditionerScheduleConfig,
    create_preconditioner,
    create_scheduled_preconditioner,
)
from neuralls.domain.solver.preconditioners.base import BindableInputs, Preconditioner
from neuralls.platform.config.models.preconditioner import PreconditionerConfig

# Arrays always available from LinearSystem — never need loading from disk.
_SYSTEM_ARRAYS_ALWAYS_AVAILABLE: frozenset[str] = frozenset({"matrix"})

StoppingCriterion = Literal["tolerance", "fixed_iterations"]

# Registry mapping config-level stopping criterion names to internal literals.
_STOPPING_CRITERION_REGISTRY: dict[str, StoppingCriterion] = {
    "tolerance": "tolerance",
    "residual_norm": "tolerance",
    "fixed": "fixed_iterations",
    "fixed_iterations": "fixed_iterations",
}


def register_stopping_criterion(name: str, criterion: StoppingCriterion) -> None:
    """Register a stopping criterion name mapping.

    Args:
        name: Config-level criterion name (case-insensitive).
        criterion: Internal typed literal — ``"tolerance"`` or ``"fixed_iterations"``.
    """
    _STOPPING_CRITERION_REGISTRY[name.lower()] = criterion


def _map_stopping_criterion(name: str) -> StoppingCriterion:
    """Map a string stopping criterion to the typed literal.

    Args:
        name: Stopping criterion name from config (e.g. ``"residual_norm"``).

    Returns:
        Typed literal ``"tolerance"`` or ``"fixed_iterations"``.

    Raises:
        ValueError: If the criterion name is not registered.
    """
    normalized = name.lower()
    criterion = _STOPPING_CRITERION_REGISTRY.get(normalized)
    if criterion is None:
        valid_names = ", ".join(sorted(_STOPPING_CRITERION_REGISTRY.keys()))
        raise ValueError(f"Unknown stopping criterion: '{name}'. Valid options: {valid_names}")
    return criterion


class PreconditionerService:
    """Service for creating and managing preconditioners via the composition factory.

    Args:
        adapter: Optional adapter for neural preconditioners (dependency injection for testing).
    """

    def __init__(self, adapter: Any = None) -> None:
        self._adapter = adapter

    def create_preconditioner(
        self,
        matrix: np.ndarray,
        config: PreconditionerConfig,
    ) -> Preconditioner:
        """Create a single preconditioner.

        Args:
            matrix: System matrix to precondition (shape: n x n).
            config: Preconditioner configuration.

        Returns:
            Preconditioner instance.
        """
        return create_preconditioner(matrix, config, adapter=self._adapter)

    def create_preconditioner_set(
        self,
        matrix: np.ndarray,
        configs: Sequence[PreconditionerConfig],
    ) -> dict[str, Preconditioner]:
        """Create multiple preconditioners for comparison.

        Args:
            matrix: System matrix to precondition (same for all).
            configs: Sequence of preconditioner configurations.

        Returns:
            Dictionary mapping preconditioner names to Preconditioner instances.
        """
        return {cfg.name: self.create_preconditioner(matrix, cfg) for cfg in configs}


def _create_scheduled_preconditioners(
    preconditioner_configs: Sequence[PreconditionerConfig],
    matrix: np.ndarray,
    base_preconditioners: dict[str, Any],
) -> dict[str, Preconditioner]:
    """Wrap base preconditioners with scheduling if configured.

    Args:
        preconditioner_configs: Config objects carrying scheduling parameters.
        matrix: System matrix (needed for fallback creation).
        base_preconditioners: Already-created preconditioner instances.

    Returns:
        Dict mapping names to scheduled (or unscheduled) preconditioner instances.
    """
    scheduled: dict[str, Preconditioner] = {}
    for cfg in preconditioner_configs:
        primary = base_preconditioners[cfg.name]
        schedule = PreconditionerScheduleConfig(
            limit_iters=cfg.limit_iters,
            fallback=cfg.fallback,
        )
        scheduled[cfg.name] = create_scheduled_preconditioner(
            primary=primary,
            schedule=schedule,
        )
    return scheduled


def _bind_system_inputs(
    preconditioners: dict[str, Preconditioner],
    system_data: dict[str, np.ndarray],
) -> None:
    """Bind dataset-sourced extra inputs to preconditioners that declare them.

    Reads extra_input_names from each preconditioner. For each name present in
    system_data, calls bind_inputs() so the preconditioner can forward them to
    the model without changing the CG-facing apply(residual) interface.

    Args:
        preconditioners: Map of name to preconditioner.
        system_data: Available named arrays (always includes ``"matrix"``).
    """
    for precond in preconditioners.values():
        if not isinstance(precond, BindableInputs):
            continue
        needed = {k: v for k, v in system_data.items() if k in precond.extra_input_names}
        if needed:
            precond.bind_inputs(**needed)
