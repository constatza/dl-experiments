"""Family classification for preconditioner configs.

Groups preconditioner configs for consumers that need to treat related
variants alike (today: comparison-plot styling) without those consumers
reaching into config internals themselves.
"""

from __future__ import annotations

from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
    PreconditionerType,
)
from neuralls.shared.types import PreconditionerFamily

__all__ = ["PreconditionerFamilyKey", "preconditioner_family"]

type PreconditionerFamilyKey = PreconditionerFamily | PreconditionerType
"""A preconditioner's family: one of the merged `PreconditionerFamily` names
(`amg`/`pod2g`/`neural`), or, for every other preconditioner type, that
type's own `PreconditionerType` member directly — no separate family name
needed since each already denotes a distinct group."""


def preconditioner_family(cfg: PreconditionerConfig) -> PreconditionerFamilyKey:
    """Classify a preconditioner config into a family.

    AMG splits into ``PreconditionerFamily.POD2G`` (POD-2G coarsening,
    inline-fit or checkpoint-predicted) and ``PreconditionerFamily.AMG``
    (classical aggregation/target-dim coarsening): the two are routinely
    compared side by side but are architecturally distinct methods.
    ``NEURAL`` and ``NEURAL_AMG`` collapse to a single
    ``PreconditionerFamily.NEURAL``. Every other type is its own family —
    its own ``PreconditionerType`` member, unmodified — since each already
    appears at most once in a typical comparison, so no further grouping is
    needed.

    Args:
        cfg (PreconditionerConfig): Preconditioner config, as constructed by
            the case TOML loader.

    Returns:
        PreconditionerFamilyKey: Family key for this config.
    """
    if isinstance(cfg, AMGPreconditionerConfig):
        if isinstance(cfg.coarsening, (PODCoarseningConfig, NeuralPODCoarseningConfig)):
            return PreconditionerFamily.POD2G
        return PreconditionerFamily.AMG
    if cfg.type in (PreconditionerType.NEURAL, PreconditionerType.NEURAL_AMG):
        return PreconditionerFamily.NEURAL
    return cfg.type
