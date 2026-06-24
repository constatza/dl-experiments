"""Predictor ports - framework-agnostic abstractions.

This module defines the port interfaces for the neural preconditioner domain,
following Ports & Adapters (Hexagonal Architecture) pattern.

Design Principles:
    - Framework-agnostic: No PyTorch, DLKit, or any ML framework imports
    - Pure numpy interface: (NDArray) -> NDArray
    - Lifecycle management: Context manager support for resource cleanup
    - SOLID: ABC enforces contracts at runtime (unlike Protocol)
    - OCP: base port is minimal; subtype ExtraInputPredictorPort extends it
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PredictorPort(ABC):
    """Abstract port for neural predictors with lifecycle management.

    This port defines the minimal interface for neural network-based
    preconditioners without coupling to specific ML frameworks. Adapters
    implement this port for different frameworks (DLKit, PyTorch, etc.).

    Implementors that do not need named extra inputs beyond the residual
    should inherit from this class directly. For models that require
    additional named arrays (e.g. stiffness matrix, coordinates), use
    :class:`ExtraInputPredictorPort` instead.

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
    def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        """Apply neural network to residual vector.

        Args:
            residual: Residual vector (numpy array, any dtype)
            **extra_inputs: Optional named arrays forwarded to the model
                (e.g., matrix=A). Implementations that do not need extra
                inputs should accept and ignore them.

        Returns:
            Predicted correction (float64 numpy array)

        Raises:
            RuntimeError: If predictor not loaded or GPU OOM
            ValueError: If residual shape incompatible
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (GPU memory, model weights). Idempotent."""
        ...

    def __enter__(self) -> PredictorPort:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - automatic cleanup."""
        self.cleanup()


class ExtraInputPredictorPort(PredictorPort):
    """Port for predictors that actively use named extra inputs.

    Inherit from this (instead of PredictorPort directly) when the model
    needs more than the residual — e.g., stiffness matrix, node coordinates,
    PDE parameters.

    Implementors must declare ``required_inputs``, which names the extra arrays
    the model expects. This eliminates the need to re-declare input names in the
    comparison TOML: the model config is the single source of truth.

    Example:
        >>> class MyPredictor(ExtraInputPredictorPort):
        ...     @property
        ...     def required_inputs(self) -> tuple[str, ...]:
        ...         return ("matrix", "positions")
        ...
        ...     def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        ...         return self._model(residual, extra_inputs["matrix"])
        ...
        ...     def cleanup(self) -> None: ...
    """

    @property
    @abstractmethod
    def required_inputs(self) -> tuple[str, ...]:
        """Names of extra arrays this model expects beyond the residual.

        Derived from the model config; eliminates need to re-declare in comparison TOML.
        The composition layer calls ``bind_inputs(**{k: v for k, v in system_data.items()
        if k in required_inputs})`` before the CG loop.

        Returns:
            Tuple of string keys expected in ``**extra_inputs`` on ``apply()``.
        """
        ...


class PredictorAdapter(ABC):
    """Abstract adapter for creating predictors from checkpoints.

    Adapters implement framework-specific loading and provide
    framework-agnostic predictors through the PredictorPort interface.

    Design:
        - Adapter pattern: Converts framework-specific API to port interface
        - Factory method: create_predictor() returns ExtraInputPredictorPort
        - Dependency inversion: Domain depends on port, not adapter
    """

    @abstractmethod
    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional model config (unused by some adapters)
            data_config_path: Optional data config (unused by some adapters)

        Returns:
            ExtraInputPredictorPort instance (use as context manager for cleanup)

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
