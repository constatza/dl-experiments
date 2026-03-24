"""Functional composition utilities for workflow pipelines.

This module provides pure functional utilities for composing transformations
in a clear, maintainable way.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")
Transform = Callable[[T], T]


def pipe(*transforms: Transform[T]) -> Transform[T]:
    """Compose multiple transforms into a sequential pipeline.

    Applies transforms left-to-right (functional composition).
    Each transform receives the output of the previous transform.

    Args:
        *transforms: Variable number of transformation functions

    Returns:
        A new function that applies all transforms sequentially

    Examples:
        >>> def add_one(x: int) -> int:
        ...     return x + 1
        >>> def double(x: int) -> int:
        ...     return x * 2
        >>> def square(x: int) -> int:
        ...     return x ** 2
        >>> pipeline = pipe(add_one, double, square)
        >>> pipeline(3)  # (3 + 1) * 2 ** 2 = 64
        64

        >>> # With settings transformations
        >>> from dlkit.tools.config.core.settings import GeneralSettings
        >>> def configure_dataset(settings: GeneralSettings) -> GeneralSettings:
        ...     return patch_model(settings, {"DATASET": new_dataset})
        >>> def configure_mlflow(settings: GeneralSettings) -> GeneralSettings:
        ...     return patch_model(settings, {"MLFLOW": mlflow_config})
        >>> pipeline = pipe(configure_dataset, configure_mlflow)
        >>> updated_settings = pipeline(original_settings)

    Notes:
        - All transforms must have the same input/output type
        - Transforms should be pure functions (no side effects)
        - Empty transforms list returns identity function
    """
    def pipeline(value: T) -> T:
        result = value
        for transform in transforms:
            result = transform(result)
        return result

    return pipeline
