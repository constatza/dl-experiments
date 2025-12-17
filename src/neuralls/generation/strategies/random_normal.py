"""Random/normal strategy implementation using registry."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import RandomNormalConfig


@register_strategy
class RandomNormalStrategy(IMatrixOnlyGenerationStrategy):
    name = "random"
    ConfigType = RandomNormalConfig

    def requires_rhs(self) -> bool:
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate random normal solutions and compute RHS.

        Args:
            matrix: System matrix
            rhs: Mother RHS (ignored)
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with random normal solutions
        """
        # Validate and convert to typed config
        config = RandomNormalConfig(**cfg)

        count = config.samples
        scale = config.target_rhs_scale
        rng = np.random.default_rng(config.seed)

        n = matrix.shape[0]
        solutions = rng.normal(size=(count, n), scale=scale).astype(
            np.float64, copy=False
        )
        rhs_out = np.array([matrix @ x for x in solutions], dtype=np.float64)
        return GeneratedSamples(matrix=matrix, rhs=rhs_out, solutions=solutions)


# Alias for "normal"
@register_strategy
class NormalStrategy(RandomNormalStrategy):
    name = "normal"
