"""Predictor factory protocol for neural preconditioner dependency injection.

This module defines the PredictorFactory abstraction that decouples neural
preconditioners from specific ML frameworks (e.g., DLKit).

Design Principles:
    - Dependency Inversion: NeuralBuilder depends on PredictorFactory protocol,
      not concrete DLKit implementation
    - Open/Closed: New ML frameworks can be supported by implementing the protocol
    - Testability: Mock factories enable testing without trained models

The protocol defines a single method `create()` that loads a trained model
and returns a callable predictor function.

Example:
    >>> # Production: Use DLKit
    >>> factory = DLKitPredictorFactory()
    >>> predictor = factory.create(checkpoint_path=Path("model.ckpt"))
    >>> solution = predictor(matrix, residual)
    >>>
    >>> # Testing: Use mock
    >>> class MockFactory:
    ...     def create(self, **kwargs):
    ...         return lambda A, r: r * 0.5
    >>> factory = MockFactory()
    >>> predictor = factory.create(checkpoint_path=Path("unused"))
    >>> # No real model needed!
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from pathlib import Path
from collections.abc import Callable

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PredictorFactory(Protocol):
    """Factory protocol for creating neural predictors.

    This protocol defines the interface for loading trained neural networks
    that can be used as preconditioners. By depending on this abstraction
    (Dependency Inversion Principle), the NeuralBuilder decouples from
    specific ML frameworks.

    Benefits:
        - **Testability**: Mock factories allow testing without trained models
        - **Flexibility**: Swap ML frameworks (DLKit, PyTorch, TensorFlow) without
          changing preconditioner code
        - **Decoupling**: Neural preconditioner logic independent of training code

    The predictor is a function taking (matrix, residual) → solution, enabling
    problem-specific learned approximations of A^{-1}.
    """

    def create(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> Callable[[NDArray, NDArray], NDArray]:
        """Create predictor from checkpoint.

        Loads a trained model from the checkpoint and returns a callable
        that can be used to predict solutions or corrections given a
        matrix and residual.

        Args:
            checkpoint_path: Path to model checkpoint file
            config_path: Optional path to model configuration file
            data_config_path: Optional path to data configuration file

        Returns:
            Callable taking (matrix, residual) → solution

        Example:
            >>> predictor = factory.create(
            ...     checkpoint_path=Path("model.ckpt"),
            ...     config_path=Path("config.toml"),
            ... )
            >>> solution = predictor(A, b)

        Notes:
            The predictor should be stateless and thread-safe if possible.
            For stateful models, document thread-safety requirements.
        """
        ...


class DLKitPredictorFactory:
    """Default predictor factory using DLKit.

    This is the production implementation that loads neural networks trained
    with DLKit. It wraps the `create_neural_preconditioner()` function from
    the neural_inference module.

    Example:
        >>> factory = DLKitPredictorFactory()
        >>> predictor = factory.create(checkpoint_path=Path("model.ckpt"))
        >>> solution = predictor(matrix, residual)

    Notes:
        This factory is lazy-loaded to avoid importing DLKit at module load time,
        which is important for:
        - Faster imports when DLKit not needed
        - Testing without DLKit installed
        - Reducing circular dependencies
    """

    def create(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> Callable[[NDArray, NDArray], NDArray]:
        """Create DLKit predictor from checkpoint.

        Args:
            checkpoint_path: Path to DLKit model checkpoint
            config_path: Optional model config path
            data_config_path: Optional data config path

        Returns:
            Callable taking (matrix, residual) → solution

        Raises:
            ImportError: If DLKit or dependencies not installed
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model loading fails

        Notes:
            Imports DLKit lazily to avoid hard dependency at module load time.
        """
        from neuralls.neural_inference import create_neural_preconditioner

        return create_neural_preconditioner(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_config_path=data_config_path,
        )
