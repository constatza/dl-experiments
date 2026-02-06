"""Predictor ports - framework-agnostic abstractions.

This module defines the port interfaces for the neural preconditioner domain,
following Ports & Adapters (Hexagonal Architecture) pattern.

Design Principles:
    - Framework-agnostic: No PyTorch, DLKit, or any ML framework imports
    - Pure numpy interface: (NDArray) -> NDArray
    - Lifecycle management: Context manager support for resource cleanup
    - SOLID: ABC enforces contracts at runtime (unlike Protocol)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PredictorPort(ABC):
    """Abstract port for neural predictors with lifecycle management.

    This port defines the interface for neural network-based preconditioners
    without coupling to specific ML frameworks. Adapters implement this port
    for different frameworks (DLKit, PyTorch, etc.).

    Lifecycle:
        1. Create predictor (loads model)
        2. Apply predictions (multiple calls)
        3. Cleanup (free GPU memory)

    Example:
        >>> with adapter.create_predictor(checkpoint) as predictor:
        ...     result = predictor.apply(residual)
        ...     # Automatic cleanup on exit
    """

    @abstractmethod
    def apply(self, residual: NDArray) -> NDArray:
        """Apply neural network to residual vector.

        Args:
            residual: Residual vector (numpy array, any dtype)

        Returns:
            Predicted correction (float64 numpy array)

        Raises:
            RuntimeError: If predictor not loaded or GPU OOM
            ValueError: If residual shape incompatible
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (GPU memory, model weights).

        Should be idempotent - safe to call multiple times.
        """
        ...

    def __enter__(self) -> PredictorPort:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatic cleanup."""
        self.cleanup()


class PredictorAdapter(ABC):
    """Abstract adapter for creating predictors from checkpoints.

    Adapters implement framework-specific loading and provide
    framework-agnostic predictors through the PredictorPort interface.

    Design:
        - Adapter pattern: Converts framework-specific API to port interface
        - Factory method: create_predictor() returns PredictorPort
        - Dependency inversion: Domain depends on port, not adapter
    """

    @abstractmethod
    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> PredictorPort:
        """Create predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional model config (unused by some adapters)
            data_config_path: Optional data config (unused by some adapters)

        Returns:
            PredictorPort instance (use as context manager for cleanup)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If model loading fails
            ImportError: If framework not installed

        Example:
            >>> adapter = DLKitAdapter()
            >>> with adapter.create_predictor(Path("model.ckpt")) as pred:
            ...     result = pred.apply(residual)
        """
        ...
