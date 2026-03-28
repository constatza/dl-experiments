"""Neural network preconditioner."""

from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray

from ..base import NonLinearPreconditioner, PreconditionerContext


class NeuralPreconditioner(NonLinearPreconditioner):
    """Neural network preconditioner with automatic cleanup.

    Loads trained model from checkpoint and applies it to residuals.
    GPU resources are automatically freed when the preconditioner is garbage collected.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Optional model configuration
        data_config_path: Optional data configuration
        adapter: Predictor adapter supplied by the composition layer

    Example:
        >>> # Simple! Direct instantiation
        >>> precond = NeuralPreconditioner(Path("model.ckpt"))
        >>> z = precond.apply(residual)
        >>> # GPU resources automatically freed when precond is garbage collected
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
        adapter=None,  # Type: PredictorAdapter | None (avoid circular import)
    ):
        """Initialize neural preconditioner from checkpoint.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional model configuration
            data_config_path: Optional data configuration
            adapter: Predictor adapter supplied by the composition layer
        """
        if adapter is None:
            raise ValueError("NeuralPreconditioner requires an explicit predictor adapter.")

        # Load predictor (GPU model)
        self._predictor = adapter.create_predictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_config_path=data_config_path,
        )

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply neural network to residual.

        Args:
            residual: Current residual vector r_k
            context: Ignored (neural preconditioner doesn't use context)

        Returns:
            Preconditioned residual z_k = network(r_k)
        """
        return self._predictor.apply(residual)

    def cleanup(self) -> None:
        """Free GPU resources manually.

        Called automatically by __del__, but can be called explicitly
        if you want to free resources before garbage collection.
        """
        # Check if _predictor exists (might not if __init__ failed)
        if hasattr(self, "_predictor") and hasattr(self._predictor, "cleanup"):
            self._predictor.cleanup()

    def __del__(self):
        """Automatic cleanup on garbage collection.

        Ensures GPU resources are freed when the preconditioner
        is no longer referenced.
        """
        self.cleanup()
