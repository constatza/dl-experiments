"""Random/normal strategy implementation using registry."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy
from ..runner import register_strategy


@register_strategy
class RandomNormalStrategy(IMatrixOnlyGenerationStrategy):
    name = "random"

    def requires_rhs(self) -> bool:
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict,
    ) -> GeneratedSamples:
        count = int(cfg.get("samples", 0))
        scale = float(cfg.get("target_rhs_scale", 1.0))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))

        n = matrix.shape[0]
        solutions = rng.normal(size=(count, n), scale=scale).astype(np.float64, copy=False)
        rhs_out = np.array([matrix @ x for x in solutions], dtype=np.float64)
        return GeneratedSamples(matrix=matrix, rhs=rhs_out, solutions=solutions)

# Alias for "normal"
@register_strategy
class NormalStrategy(RandomNormalStrategy):
    name = "normal"
