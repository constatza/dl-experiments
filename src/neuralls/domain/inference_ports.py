"""Domain-owned ports for batch inference prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InferencePredictorPort(ABC):
    """Abstract port for batch inference predictors.

    Application-level inference logic depends on this port instead of any
    framework-specific predictor object. Implementations may wrap DLKit or any
    other runtime, but callers always exchange numpy feature batches and numpy
    predictions.
    """

    @abstractmethod
    def predict_batch(self, feature_batch: dict[str, NDArray]) -> NDArray:
        """Predict one batch of samples from named feature arrays."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release any runtime resources held by the predictor."""
        ...

    def __enter__(self) -> InferencePredictorPort:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()
