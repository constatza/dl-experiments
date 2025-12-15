"""Eigenvector-based generation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy
from ..runner import register_strategy
from ..strategy_configs import EigenvectorForwardConfig, EigenvectorInverseConfig
from ..helpers import (
    _compute_eigendecomposition,
    _select_eigenvectors,
    _generate_eigenvector_combinations,
)
from scipy.linalg import solve  # type: ignore[import-untyped]


def _validate_count(count: int, n: int) -> None:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")


@register_strategy
class EigenvectorForwardStrategy(IMatrixOnlyGenerationStrategy):
    name = "eigenvector_forward"
    ConfigType = EigenvectorForwardConfig

    def requires_rhs(self) -> bool:
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict[str, Any],
    ) -> GeneratedSamples:
        # Validate and convert to typed config
        config = EigenvectorForwardConfig(**cfg)

        count = config.samples
        which = config.which
        include_eigenvectors = config.include_eigenvectors
        num_eigenvectors_raw = config.num_eigenvectors
        rng = np.random.default_rng(config.seed)

        eigenvalues, eigenvectors = _compute_eigendecomposition(matrix)
        n = eigenvectors.shape[0]
        _validate_count(count, n)

        if num_eigenvectors_raw is None:
            num_eigenvectors = n
        else:
            num_eigenvectors = int(num_eigenvectors_raw)
            if num_eigenvectors == -1:
                num_eigenvectors = n
        if num_eigenvectors > n or num_eigenvectors <= 0:
            raise ValueError(
                f"num_eigenvectors ({num_eigenvectors}) must be positive and ≤ {n}"
            )

        selected_eigenvectors, _, _ = _select_eigenvectors(
            eigenvectors, eigenvalues, num_eigenvectors, which, rng
        )

        if include_eigenvectors:
            num_combinations = count - num_eigenvectors
            if num_combinations < 0:
                raise ValueError(
                    f"When include_eigenvectors=True, count ({count}) must be >= num_eigenvectors ({num_eigenvectors})."
                )
            if num_combinations > 0:
                X_combinations = _generate_eigenvector_combinations(
                    selected_eigenvectors, num_combinations, rng
                )
                X = np.vstack([selected_eigenvectors.T, X_combinations])
            else:
                X = selected_eigenvectors.T
        else:
            X = _generate_eigenvector_combinations(selected_eigenvectors, count, rng)

        rhs_out = np.array([matrix @ x for x in X], dtype=np.float64)

        return GeneratedSamples(matrix=matrix, rhs=rhs_out, solutions=X)


@register_strategy
class EigenvectorInverseStrategy(IMatrixOnlyGenerationStrategy):
    name = "eigenvector_inverse"
    ConfigType = EigenvectorInverseConfig

    def requires_rhs(self) -> bool:
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict[str, Any],
    ) -> GeneratedSamples:
        # Validate and convert to typed config
        config = EigenvectorInverseConfig(**cfg)

        count = config.samples
        which = config.which
        include_eigenvectors = config.include_eigenvectors
        num_eigenvectors_raw = config.num_eigenvectors
        rng = np.random.default_rng(config.seed)

        eigenvalues, eigenvectors = _compute_eigendecomposition(matrix)
        n = eigenvectors.shape[0]
        _validate_count(count, n)

        if num_eigenvectors_raw is None:
            num_eigenvectors = n
        else:
            num_eigenvectors = int(num_eigenvectors_raw)
            if num_eigenvectors == -1:
                num_eigenvectors = n
        if num_eigenvectors > n or num_eigenvectors <= 0:
            raise ValueError(
                f"num_eigenvectors ({num_eigenvectors}) must be positive and ≤ {n}"
            )

        selected_eigenvectors, _, _ = _select_eigenvectors(
            eigenvectors, eigenvalues, num_eigenvectors, which, rng
        )

        if include_eigenvectors:
            num_combinations = count - num_eigenvectors
            if num_combinations < 0:
                raise ValueError(
                    f"When include_eigenvectors=True, count ({count}) must be >= num_eigenvectors ({num_eigenvectors})."
                )
            if num_combinations > 0:
                rhs_comb = _generate_eigenvector_combinations(
                    selected_eigenvectors, num_combinations, rng
                )
                R = np.vstack([selected_eigenvectors.T, rhs_comb])
            else:
                R = selected_eigenvectors.T
        else:
            R = _generate_eigenvector_combinations(selected_eigenvectors, count, rng)

        # For inverse mode, RHS vectors are the eigenvectors/combos, solutions solved directly
        rhs_out = R
        solutions_out = np.zeros_like(rhs_out)
        for idx, rhs_vec in enumerate(rhs_out):
            solutions_out[idx] = solve(matrix, rhs_vec, assume_a="pos")

        return GeneratedSamples(matrix=matrix, rhs=rhs_out, solutions=solutions_out)
