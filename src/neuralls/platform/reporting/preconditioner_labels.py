"""Descriptive labels for constructed preconditioner objects.

Builds short, human-readable structural descriptions of live ``torchalg``
preconditioner instances (grid levels, cycle type, smoother damping,
coarsening strategy, fitted POD rank, ...) for use in comparison plot
legends and axis labels.

Detail is read from the constructed object's own attributes, never from the
TOML config that produced it: the object is the ground truth for what was
actually built (e.g. the POD basis's real fitted rank, which may differ from
a configured energy-threshold ``rank``).
"""

from __future__ import annotations

from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import IC0Preconditioner, ScheduledPreconditioner
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    AMGPreconditioner,
    JacobiSmoother,
    VCycle,
    WCycle,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy


def describe_preconditioner(precond: Preconditioner) -> str:
    """Build a short structural-detail string from a constructed preconditioner.

    Args:
        precond (Preconditioner): A constructed (already-instantiated)
            preconditioner object, e.g. as produced by
            ``composition.preconditioners.factory.create_preconditioner``.
            ``ScheduledPreconditioner`` wrappers are unwrapped to their
            primary preconditioner before inspection.

    Returns:
        str: A structural detail string, such as ``"2-lvl V-cycle, Jacobi
            ω=0.67, aggregation θ=0.25 ω=0.67"`` for AMG, or ``""`` for
            preconditioner types with no meaningful structural variants
            (e.g. ``Identity``, ``JacobiPreconditioner``, ``NeuralPreconditioner``).
    """
    if isinstance(precond, ScheduledPreconditioner):
        return describe_preconditioner(precond._primary)
    if isinstance(precond, AMGPreconditioner):
        return _describe_amg(precond)
    if isinstance(precond, IC0Preconditioner):
        return f"threshold={precond._threshold:.0e}"
    return ""


def preconditioner_label(name: str, precond: Preconditioner) -> str:
    """Combine a preconditioner's config name with its live structural detail.

    Args:
        name (str): Config-level preconditioner name (the dict key used
            throughout the comparison pipeline).
        precond (Preconditioner): The corresponding constructed
            preconditioner object.

    Returns:
        str: ``"{name} ({detail})"`` when structural detail is available
            (see :func:`describe_preconditioner`), else just ``name``.
    """
    detail = describe_preconditioner(precond)
    return f"{name} ({detail})" if detail else name


def build_preconditioner_labels(
    preconditioners: dict[str, Preconditioner],
) -> dict[str, str]:
    """Build a name -> descriptive-label mapping for a set of preconditioners.

    Args:
        preconditioners (dict[str, Preconditioner]): Constructed
            preconditioner instances keyed by config name, e.g. as produced
            by ``PreconditionerService.create_preconditioner_set``.

    Returns:
        dict[str, str]: Mapping from each name to its
            :func:`preconditioner_label` string.
    """
    return {name: preconditioner_label(name, precond) for name, precond in preconditioners.items()}


def _describe_amg(precond: AMGPreconditioner) -> str:
    """Describe an AMG preconditioner's grid levels, cycle, smoother, and coarsening.

    Args:
        precond (AMGPreconditioner): Constructed AMG preconditioner instance.

    Returns:
        str: Comma-separated structural detail string.
    """
    parts = [f"{precond._n_levels}-lvl {_cycle_name(precond._cycle)}"]
    smoother = getattr(precond._cycle, "_smoother", None)
    if isinstance(smoother, JacobiSmoother):
        parts.append(f"Jacobi ω={smoother._omega:.2g}")
    parts.append(_describe_coarsening(precond._coarsening))
    return ", ".join(parts)


def _cycle_name(cycle: object) -> str:
    """Return a short display name for a multigrid cycle instance.

    Args:
        cycle (object): The AMG preconditioner's cycle object (typically a
            ``VCycle`` or ``WCycle``).

    Returns:
        str: ``"V-cycle"``, ``"W-cycle"``, or the class name as a fallback
            for other ``MultigridCycle`` implementations.
    """
    if isinstance(cycle, WCycle):
        return "W-cycle"
    if isinstance(cycle, VCycle):
        return "V-cycle"
    return type(cycle).__name__


def _describe_coarsening(coarsening: object) -> str:
    """Describe an AMG coarsening strategy's key structural parameter.

    Args:
        coarsening (object): The AMG preconditioner's coarsening strategy
            object.

    Returns:
        str: ``"POD rank={n}"`` using the actual fitted basis width for
            POD-2G coarsening, ``"aggregation θ={theta} ω={omega}"`` for
            smoothed aggregation, or the class name as a fallback for other
            ``CoarseningStrategy`` implementations.
    """
    if isinstance(coarsening, PODCoarseningStrategy):
        return f"POD rank={coarsening._basis.shape[1]}"
    if isinstance(coarsening, AggregationCoarsening):
        return f"aggregation θ={coarsening._theta:.2g} ω={coarsening._omega:.2g}"
    return type(coarsening).__name__
