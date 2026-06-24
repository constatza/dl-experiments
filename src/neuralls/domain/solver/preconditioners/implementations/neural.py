"""Neural network preconditioner."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from numpy.typing import NDArray

from ..base import NonLinearPreconditioner, PreconditionerContext
from ..ports import ExtraInputPredictorPort, PredictorAdapter


class NeuralPreconditioner(NonLinearPreconditioner):
    """Neural network preconditioner with automatic cleanup.

    Loads trained model from checkpoint and applies it to residuals.
    Extra named inputs (matrix, coordinates, parameters) declared in
    extra_input_names are pre-bound via bind_inputs() before the CG loop
    and forwarded to the predictor on each apply() call.

    GPU resources are automatically freed when the preconditioner is garbage
    collected.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Optional model configuration
        data_config_path: Optional data configuration
        adapter (PredictorAdapter | None): Predictor adapter supplied by the
            composition layer. Must not be None at runtime.
        extra_input_names: Names of extra inputs the model expects beyond the
            residual (e.g., ``("matrix",)``). Matched against bind_inputs() keys.

    Example:
        >>> precond = NeuralPreconditioner(
        ...     Path("model.ckpt"), adapter=adapter, extra_input_names=("matrix",)
        ... )
        >>> precond.bind_inputs(matrix=A)  # done once before CG loop
        >>> z = precond.apply(residual)  # called each CG iteration
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
        adapter: PredictorAdapter | None = None,
        extra_input_names: tuple[str, ...] = (),
    ):
        """Initialize neural preconditioner from checkpoint.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional model configuration
            data_config_path: Optional data configuration
            adapter (PredictorAdapter | None): Predictor adapter supplied by the
                composition layer. Must not be None at runtime.
            extra_input_names: Names of extra inputs the model expects beyond the
                residual (e.g., ``("matrix",)``). Matched against bind_inputs() keys.
        """
        if adapter is None:
            raise ValueError("NeuralPreconditioner requires an explicit predictor adapter.")

        self._extra_input_names: tuple[str, ...] = extra_input_names
        self._extra_inputs: dict[str, NDArray] = {}

        # Load predictor (GPU model); adapter contract guarantees ExtraInputPredictorPort
        self._predictor: ExtraInputPredictorPort = adapter.create_predictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_config_path=data_config_path,
        )

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Names of extra inputs this preconditioner expects beyond the residual.

        When the config field is empty, falls back to the predictor's own declaration
        so that comparison TOMLs do not need to duplicate what the model config states.
        """
        if self._extra_input_names:
            return self._extra_input_names
        return self._predictor.required_inputs

    def bind_inputs(self, **inputs: NDArray) -> None:
        """Store extra named inputs for forwarding on each apply() call.

        Only keys declared in extra_input_names are retained; others are ignored
        with a warning so misconfigured extra_input_names is detectable.

        Args:
            **inputs: Named arrays matching extra_input_names entries.
        """
        unexpected = set(inputs) - set(self._extra_input_names)
        if unexpected:
            logger.warning(
                "bind_inputs: keys {} not in extra_input_names {} — ignored",
                unexpected,
                self._extra_input_names,
            )
        self._extra_inputs = {k: v for k, v in inputs.items() if k in self._extra_input_names}

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply neural network to residual.

        Args:
            residual: Current residual vector r_k
            context: Ignored (neural preconditioner doesn't use context)

        Returns:
            Preconditioned residual z_k = network(r_k, **extra_inputs)
        """
        return self._predictor.apply(residual, **self._extra_inputs)

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
