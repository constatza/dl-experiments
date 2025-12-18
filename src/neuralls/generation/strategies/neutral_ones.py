"""Neutral ones strategy: x=ones, b=A@x for unbiased baseline testing."""

from __future__ import annotations

import numpy as np

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy, ArchiveData
from ..runner import register_strategy
from ..strategy_configs import BaseStrategyConfig


class NeutralOnesConfig(BaseStrategyConfig):
    """Configuration for NeutralOnesStrategy.

    Generates neutral test cases where x = ones and b = A @ x.
    This provides an unbiased baseline for solver comparison across experiments.
    """
    pass


@register_strategy
class NeutralOnesStrategy(IMatrixOnlyGenerationStrategy):
    """Generate neutral test cases with x=ones, b=A@x.

    This strategy creates a deterministic, unbiased test case where the solution
    is simply a vector of ones. This is useful for:
    - Consistent baseline comparison across different experiments
    - Testing solver behavior on simple, known solutions
    - Validating that A @ ones produces expected RHS

    The normalization is applied by the data generation pipeline BEFORE this
    strategy is called, so this generates normalized data in normalized space.
    """
    name = "neutral_ones"
    ConfigType = NeutralOnesConfig

    def requires_rhs(self) -> bool:
        """Neutral ones doesn't require mother RHS."""
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate x=ones and compute b=A@x.

        Args:
            matrix: System matrix (already normalized)
            rhs: Mother RHS (ignored, not needed for neutral test)
            cfg: Configuration dictionary
            archive: Optional archive data (ignored)

        Returns:
            GeneratedSamples with x=ones solutions and b=A@x RHS
        """
        # Validate and convert to typed config
        config = NeutralOnesConfig(**cfg)

        count = config.samples
        n = matrix.shape[0]

        # Generate x = ones for all samples
        solutions = np.ones((count, n), dtype=np.float64)

        # Compute b = A @ x for each solution
        rhs_out = np.array([matrix @ x for x in solutions], dtype=np.float64)

        return GeneratedSamples(matrix=matrix, rhs=rhs_out, solutions=solutions)
