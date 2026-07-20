"""Preconditioner factory for TOML workflow.

This module provides a simple factory function for creating preconditioners
from configuration objects. It lives in the assembly layer because it depends
on both the configuration layer (PreconditionerType, config models) and the
solver layer (concrete preconditioner classes).

For direct usage, just instantiate preconditioners directly:
    >>> precond = JacobiPreconditioner(matrix)

For TOML workflow:
    >>> config = load_comparison_config("comparison.toml")
    >>> precond = create_preconditioner(matrix, config.preconditioner)

Design:
    - Simple factory function with isinstance checks and explicit mapping
    - No builder classes needed - just call preconditioner constructors
    - Supports dependency injection for neural preconditioners (testing)
    - Better type safety via isinstance checks (understood by mypy)
    - Explicit enum-to-class mapping for clarity and extensibility
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import (
    Identity,
    JacobiPreconditioner,
    ILUPreconditioner,
    IC0Preconditioner,
    ICholeskyPreconditioner,
    NeuralPreconditioner,
)
from neuralls.platform.config.models.preconditioner import PreconditionerType

if TYPE_CHECKING:
    from neuralls.platform.config.models.preconditioner import (
        ConcretePreconditionerConfig,
    )
    from neuralls.domain.inference_ports import InferencePredictorPort
    from torchalg.preconditioners.ports import PredictorAdapter


@dataclass(frozen=True)
class PreconditionerScheduleConfig:
    """Scheduling parameters for preconditioner switching.

    Extracted from BasePreconditionerConfig for internal use.
    Separates scheduling concerns from preconditioner configuration.

    Attributes:
        start_iter: Iteration at which the primary preconditioner becomes active.
        limit_iters: Number of iterations to apply primary preconditioner.
                     -1 means unlimited (use primary for entire solve).
        fallback: Preconditioner type to switch to after limit is reached.
    """

    start_iter: int = 0
    limit_iters: int = -1
    fallback: PreconditionerType = PreconditionerType.IDENTITY


def create_preconditioner(
    matrix: torch.Tensor,
    config: ConcretePreconditionerConfig,
    adapter: PredictorAdapter | None = None,
    inference_predictor_factory: Callable[[Path, Any], InferencePredictorPort] | None = None,
) -> Preconditioner:
    """Create preconditioner from configuration.

    Uses isinstance checks for type narrowing and explicit mapping
    for better type safety and IDE support.

    Args:
        matrix: System matrix A
        config: Preconditioner configuration from TOML
        adapter: Optional adapter for neural preconditioner (DI for testing)
        inference_predictor_factory: Optional batch-inference predictor
            factory for neural POD-2G coarsening (DI for testing); defaults
            to `create_inference_predictor` from `platform.dlkit.inference_adapter`.

    Returns:
        Preconditioner instance

    Example:
        >>> # Load from TOML
        >>> config = load_comparison_config("comparison.toml")
        >>> precond = create_preconditioner(A, config.preconditioner)
        >>>
        >>> # Use it
        >>> z = precond.apply(residual)

    Raises:
        ValueError: If preconditioner type is not supported
    """
    from neuralls.platform.config.models.preconditioner import (
        AMGPreconditionerConfig,
        IC0PreconditionerConfig,
        NeuralAMGPreconditionerConfig,
        NeuralPODCoarseningConfig,
        NeuralPreconditionerConfig,
        PODCoarseningConfig,
    )

    # AMG family: config.coarsening selects the strategy that builds the
    # prolongation/restriction operator (aggregation vs. POD-2G); everything
    # else (cycle, smoothing, n_levels) is shared.
    if config.type == PreconditionerType.AMG:
        if not isinstance(config, AMGPreconditionerConfig):
            raise TypeError(f"AMG type requires AMGPreconditionerConfig, got {type(config)}")
        from torchalg.preconditioners.implementations.amg import (
            AggregationCoarsening,
            AMGPreconditioner,
            JacobiSmoother,
            VCycle,
        )

        if isinstance(config.coarsening, PODCoarseningConfig):
            from neuralls.platform.storage.dataset_readers import load_dense_training_arrays
            from torchalg.preconditioners.implementations.pod import (
                PODCoarseningStrategy,
            )

            _, solutions = load_dense_training_arrays(config.coarsening.dataset_dir)
            if config.coarsening.n_snapshots != -1:
                solutions = solutions[: config.coarsening.n_snapshots]
            coarsening = PODCoarseningStrategy(
                snapshots=torch.as_tensor(solutions, dtype=matrix.dtype, device=matrix.device),
                rank=config.coarsening.rank,
            )
        elif isinstance(config.coarsening, NeuralPODCoarseningConfig):
            from neuralls.application.inference.prediction import (
                collect_predictions,
                stack_predictions,
            )
            from neuralls.platform.storage.dataset_readers import load_parameter_arrays
            from torchalg.preconditioners.implementations.pod import (
                PODCoarseningStrategy,
            )

            neural_pod_cfg = config.coarsening
            ckpt = neural_pod_cfg.resolved_checkpoint_path or neural_pod_cfg.checkpoint_path
            if ckpt is None:
                raise ValueError(
                    "NeuralPODCoarseningConfig requires checkpoint_path or resolved_checkpoint_path"
                )
            if inference_predictor_factory is None:
                from neuralls.platform.dlkit.inference_adapter import (
                    create_inference_predictor,
                )

                inference_predictor_factory = create_inference_predictor

            param_arrays = load_parameter_arrays(neural_pod_cfg.dataset_dir)
            input_names = neural_pod_cfg.input_names
            if len(input_names) != len(param_arrays):
                raise ValueError(
                    f"NeuralPODCoarseningConfig.input_names has {len(input_names)} name(s) but "
                    f"dataset_dir={neural_pod_cfg.dataset_dir!r} has {len(param_arrays)} `params` "
                    "array(s) — one name per array, in matching order, is required."
                )
            feature_batch = dict(zip(input_names, param_arrays))
            with inference_predictor_factory(ckpt, None) as predictor:
                raw_predictions, _ = collect_predictions(predictor, feature_batch, batch_size=256)
            predicted = stack_predictions(raw_predictions)
            if neural_pod_cfg.n_snapshots != -1:
                predicted = predicted[: neural_pod_cfg.n_snapshots]
            coarsening = PODCoarseningStrategy(
                snapshots=torch.as_tensor(predicted, dtype=matrix.dtype, device=matrix.device),
                rank=neural_pod_cfg.rank,
            )
        else:
            coarsening = AggregationCoarsening(omega=config.coarsening.omega)

        smoother = JacobiSmoother(omega=config.smoother_omega)
        cycle = VCycle(
            smoother=smoother,
            n_pre=config.pre_smoothing_steps,
            n_post=config.post_smoothing_steps,
        )
        return AMGPreconditioner(
            matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=True
        )

    # Neural AMG (neural prolongation/restriction, stub)
    if config.type == PreconditionerType.NEURAL_AMG:
        if not isinstance(config, NeuralAMGPreconditionerConfig):
            raise TypeError(
                f"NEURAL_AMG type requires NeuralAMGPreconditionerConfig, got {type(config)}"
            )
        if adapter is None:
            from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter

            adapter = DLKitAdapter()
        p_cfg = config.prolongation
        ckpt_p = p_cfg.resolved_checkpoint_path or p_cfg.checkpoint_path
        if ckpt_p is None:
            raise ValueError("NeuralAMGPreconditionerConfig.prolongation requires a checkpoint")
        prolongator = adapter.create_predictor(ckpt_p, p_cfg.config_path, p_cfg.data_config_path)
        restrictor = None
        if config.restriction is not None:
            r_cfg = config.restriction
            ckpt_r = r_cfg.resolved_checkpoint_path or r_cfg.checkpoint_path
            if ckpt_r is None:
                raise ValueError("NeuralAMGPreconditionerConfig.restriction requires a checkpoint")
            restrictor = adapter.create_predictor(ckpt_r, r_cfg.config_path, r_cfg.data_config_path)
        from torchalg.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            NeuralCoarseningStrategy,
            VCycle,
        )

        coarsening = NeuralCoarseningStrategy(prolongator=prolongator, restrictor=restrictor)
        smoother = JacobiSmoother(omega=config.smoother_omega)
        cycle = VCycle(
            smoother=smoother,
            n_pre=config.pre_smoothing_steps,
            n_post=config.post_smoothing_steps,
        )
        return AMGPreconditioner(
            matrix, coarsening=coarsening, cycle=cycle, n_levels=config.n_levels, linear=False
        )

    # Check if type is NEURAL but config is not NeuralPreconditionerConfig
    if config.type == PreconditionerType.NEURAL:
        if not isinstance(config, NeuralPreconditionerConfig):
            raise TypeError(f"Neural type requires NeuralPreconditionerConfig, got {type(config)}")
        ckpt = config.resolved_checkpoint_path or config.checkpoint_path
        if ckpt is None:
            raise ValueError(
                "NeuralPreconditionerConfig requires checkpoint_path or resolved_checkpoint_path"
            )
        if adapter is None:
            from neuralls.platform.dlkit.predictor_adapter import DLKitAdapter

            adapter = DLKitAdapter()
        return NeuralPreconditioner(
            checkpoint_path=ckpt,
            config_path=config.config_path,
            data_config_path=config.data_config_path,
            adapter=adapter,
            extra_input_names=tuple(config.extra_input_names),
        )

    # IC(0) with threshold parameter
    if config.type == PreconditionerType.IC0:
        if not isinstance(config, IC0PreconditionerConfig):
            raise TypeError(f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}")
        return IC0Preconditioner(matrix, threshold=config.threshold)

    # Standard cases with explicit dispatch for type safety
    if config.type in (PreconditionerType.IDENTITY, PreconditionerType.NONE):
        return Identity()
    if config.type == PreconditionerType.JACOBI:
        return JacobiPreconditioner(matrix)
    if config.type == PreconditionerType.ILU:
        return ILUPreconditioner(matrix)
    if config.type == PreconditionerType.ICHOLESKY:
        return ICholeskyPreconditioner(matrix)

    raise ValueError(f"Unsupported preconditioner type: {config.type}")


def _extract_schedule(cfg: ConcretePreconditionerConfig) -> PreconditionerScheduleConfig:
    """Extract scheduling parameters from preconditioner config.

    Pure function to extract scheduling concerns from mixed config.

    Args:
        cfg: Preconditioner configuration from TOML

    Returns:
        Extracted schedule configuration
    """
    return PreconditionerScheduleConfig(
        start_iter=cfg.start_iter,
        limit_iters=cfg.limit_iters,
        fallback=cfg.fallback,
    )


def create_scheduled_preconditioner(
    primary: Preconditioner,
    schedule: PreconditionerScheduleConfig,
) -> Preconditioner:
    """Create a scheduled preconditioner based on schedule config.

    Args:
        primary: Main preconditioner to apply
        schedule: Schedule configuration with activation, limit, and fallback type

    Returns:
        ScheduledPreconditioner if delayed or limited, otherwise primary unchanged

    Example:
        >>> # Limit neural preconditioner to first 10 iterations
        >>> schedule = PreconditionerScheduleConfig(limit_iters=10)
        >>> scheduled = create_scheduled_preconditioner(neural_precond, schedule)
    """
    if schedule.start_iter == 0 and schedule.limit_iters < 0:
        return primary

    from torchalg.preconditioners.implementations.scheduled import (
        ScheduledPreconditioner,
    )

    # Create fallback preconditioner based on type
    if schedule.fallback == PreconditionerType.IDENTITY:
        fallback_precond = Identity()
    else:
        raise ValueError(f"Unsupported fallback type: {schedule.fallback}")

    return ScheduledPreconditioner(
        primary=primary,
        fallback=fallback_precond,
        limit_iters=None if schedule.limit_iters < 0 else schedule.limit_iters,
        start_iter=schedule.start_iter,
    )
