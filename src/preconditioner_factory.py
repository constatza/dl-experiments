"""Factory functions for preconditioners, warm starts, and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Literal

import numpy as np
from loguru import logger
from scipy.sparse.linalg import LinearOperator, spilu

from .math_utils import _to_csc
from .neural_inference import create_neural_preconditioner, create_neural_step_helper
from .pca_training import load_pca_model, get_pca_components_matrix
from .metadata_repository import extract_residual_iters


@dataclass(frozen=True)
class NeuralPreconditionerMetadata:
    """Metadata for neural preconditioner.

    Attributes:
        residual_iters: Number of CG iterations the model was trained on (None if unknown)
        applied_iters: Number of iterations to apply during inference (None = use residual_iters)
    """

    residual_iters: int | None
    applied_iters: int | None = None


def make_identity_preconditioner() -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: x


def _to_linear_operator(
    matvec: Callable[[np.ndarray], np.ndarray],
    A: np.ndarray,
) -> LinearOperator:
    dtype = np.asarray(A).dtype
    return LinearOperator(
        shape=A.shape,
        dtype=dtype,
        matvec=lambda x: matvec(np.asarray(x, dtype=dtype, copy=False)),
    )


def make_jacobi_preconditioner(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    diag = np.diag(A)
    # Keep the preconditioner positive and avoid identity fallback on tiny entries.
    eps = 1e-8
    diag_safe = np.where(np.abs(diag) > eps, np.abs(diag), eps)
    diag_inv = 1.0 / diag_safe
    return lambda x: diag_inv * x


def make_ilu_preconditioner(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    A_csc = _to_csc(A)
    ilu = spilu(A_csc)
    ilu_dtype = A_csc.dtype
    return lambda x: ilu.solve(x.astype(ilu_dtype, copy=False))


def make_identity_preconditioner_operator(A: np.ndarray) -> LinearOperator:
    return _to_linear_operator(make_identity_preconditioner(), A)


def make_jacobi_preconditioner_operator(A: np.ndarray) -> LinearOperator:
    return _to_linear_operator(make_jacobi_preconditioner(A), A)


def make_ilu_preconditioner_operator(A: np.ndarray) -> LinearOperator:
    return _to_linear_operator(make_ilu_preconditioner(A), A)


def make_neural_preconditioner(
    A: np.ndarray,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], NeuralPreconditionerMetadata]:
    """Create neural preconditioner with training metadata.

    Args:
        A: System matrix
        checkpoint_path: Path to trained model checkpoint
        config_path: Optional model config path
        data_config_path: Optional data config path

    Returns:
        Tuple of (preconditioner_function, metadata)
    """
    predictor = create_neural_preconditioner(
        checkpoint_path,
        config_path,
        data_config_path,
    )
    matrix = np.asarray(A, dtype=np.float64, copy=False)

    # Extract training metadata
    try:
        residual_iters = extract_residual_iters(
            checkpoint_path=checkpoint_path,
            data_config_path=data_config_path,
        )
    except ValueError:
        logger.warning(
            f"Could not extract residual_iters from checkpoint={checkpoint_path} "
            f"or data_config={data_config_path}. Neural preconditioner will be applied for all iterations."
        )
        residual_iters = None

    metadata = NeuralPreconditionerMetadata(
        residual_iters=residual_iters,
        applied_iters=None,  # Set by comparison layer
    )

    def _precondition(residual: np.ndarray) -> np.ndarray:
        residual_vec = np.asarray(residual, dtype=np.float64, copy=False)
        solution = predictor(matrix, residual_vec)
        return np.asarray(solution, dtype=np.float64, copy=False)

    return _precondition, metadata


def make_hybrid_preconditioner(
    A: np.ndarray,
    neural_checkpoint_path: str | Path,
    neural_iters: int = 5,
    fallback: Literal["identity", "jacobi", "ilu"] = "jacobi",
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    NeuralPreconditionerMetadata,
]:
    """Create neural preconditioner with cheaper fallback.

    Returns a pair of preconditioners for use with flexible_pcg's
    precond_iters parameter, plus metadata.

    Args:
        A: System matrix
        neural_checkpoint_path: Path to neural network checkpoint
        neural_iters: Number of iterations to use neural preconditioner
        fallback: Type of fallback preconditioner ("identity", "jacobi", "ilu")
        config_path: Optional model config path
        data_config_path: Optional data config path

    Returns:
        Tuple of (neural_preconditioner, fallback_preconditioner, metadata)
    """
    neural_precond, metadata = make_neural_preconditioner(
        A, neural_checkpoint_path, config_path, data_config_path
    )

    if fallback == "identity":
        fallback_precond = make_identity_preconditioner()
    elif fallback == "jacobi":
        fallback_precond = make_jacobi_preconditioner(A)
    elif fallback == "ilu":
        fallback_precond = make_ilu_preconditioner(A)
    else:
        raise ValueError(f"Unknown fallback preconditioner: {fallback}")

    return neural_precond, fallback_precond, metadata


def make_neural_step_helper(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> Callable[..., np.ndarray | None]:
    return create_neural_step_helper(
        checkpoint_path,
        config_path,
        data_config_path,
    )


def make_pca_preconditioner(
    A: np.ndarray, pca_path: str | Path
) -> Callable[[np.ndarray], np.ndarray]:
    pca, stats = load_pca_model(pca_path)
    V = get_pca_components_matrix(pca)
    A_reduced = V.T @ A @ V
    try:
        A_reduced_inv = np.linalg.inv(A_reduced)
        use_precomputed_inv = True
    except np.linalg.LinAlgError:
        use_precomputed_inv = False

    def preconditioner(r: np.ndarray) -> np.ndarray:
        r_reduced = V.T @ r
        if use_precomputed_inv:
            z_reduced = A_reduced_inv @ r_reduced
        else:
            z_reduced = np.linalg.solve(A_reduced, r_reduced)
        return V @ z_reduced

    return preconditioner


def make_pca_preconditioner_operator(
    A: np.ndarray, pca_path: str | Path
) -> LinearOperator:
    return _to_linear_operator(make_pca_preconditioner(A, pca_path), A)


def make_pca_warm_start(
    A: np.ndarray, pca_path: str | Path
) -> Callable[[np.ndarray], np.ndarray]:
    pca, stats = load_pca_model(pca_path)
    V = get_pca_components_matrix(pca)
    A_reduced = V.T @ A @ V
    try:
        A_reduced_inv = np.linalg.inv(A_reduced)
        use_precomputed_inv = True
    except np.linalg.LinAlgError:
        use_precomputed_inv = False

    def warm_start(b: np.ndarray) -> np.ndarray:
        b_reduced = V.T @ b
        if use_precomputed_inv:
            x_reduced = A_reduced_inv @ b_reduced
        else:
            x_reduced = np.linalg.solve(A_reduced, b_reduced)
        return V @ x_reduced

    return warm_start


def create_preconditioner(
    name: Literal["none", "jacobi", "ilu", "pca", "neural"],
    A: np.ndarray,
    **kwargs,
) -> (
    Callable[[np.ndarray], np.ndarray]
    | tuple[Callable[[np.ndarray], np.ndarray], NeuralPreconditionerMetadata]
):
    """Create a preconditioner by name.

    Args:
        name: Preconditioner type
        A: System matrix
        **kwargs: Additional arguments for specific preconditioners

    Returns:
        For classical preconditioners: just the preconditioner function
        For neural preconditioner: tuple of (function, metadata)
    """
    if name == "none":
        return make_identity_preconditioner()
    if name == "jacobi":
        return make_jacobi_preconditioner(A)
    if name == "ilu":
        return make_ilu_preconditioner(A)
    if name == "pca":
        pca_path = kwargs.get("pca_path")
        if pca_path is None:
            raise ValueError("PCA preconditioner requires 'pca_path'")
        return make_pca_preconditioner(A, pca_path)
    if name == "neural":
        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("Neural preconditioner requires 'checkpoint_path'")
        return make_neural_preconditioner(
            A,
            checkpoint_path,
            kwargs.get("config_path"),
            kwargs.get("data_config_path"),
        )
    raise ValueError(f"Unknown preconditioner type: {name}")


def create_preconditioners(
    A: np.ndarray,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "none": create_preconditioner("none", A),
        "jacobi": create_preconditioner("jacobi", A),
        "ilu": create_preconditioner("ilu", A),
    }


def create_warm_starts(
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    pca_path: str | Path | None = None,
    A: np.ndarray | None = None,
) -> dict[str, Callable[[np.ndarray], np.ndarray | None]]:
    warm_starts: dict[str, Callable[[np.ndarray], np.ndarray | None]] = {
        "none": lambda _: None,
    }

    checkpoint_to_use = checkpoint_path
    config_to_use = config_path
    data_config_to_use = data_config_path

    if checkpoint_to_use is not None:
        try:
            predictor = create_neural_preconditioner(
                checkpoint_to_use,
                config_to_use,
                data_config_to_use,
            )

            def _predict(rhs: np.ndarray, *, _pred=predictor, _A=A) -> np.ndarray:
                return _pred(_A, rhs)

            warm_starts["neural_warm_start"] = _predict
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to create neural warm start: {exc}")

            def _raise(_: np.ndarray, *, _err=exc) -> np.ndarray:  # type: ignore[return-type]
                raise RuntimeError(
                    f"Neural warm start unavailable: {_err}. Checkpoint: {checkpoint_to_use}"
                )

            warm_starts["neural_warm_start"] = _raise

    if pca_path is not None:
        if A is None:
            logger.error("PCA warm start requires system matrix A")

            def _raise_no_A(_: np.ndarray) -> np.ndarray:  # type: ignore[return-type]
                raise RuntimeError("PCA warm start requires system matrix A")

            warm_starts["pca_warm_start"] = _raise_no_A
        else:
            try:
                warm_starts["pca_warm_start"] = make_pca_warm_start(A, pca_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to create PCA warm start: {exc}")

                def _raise(_: np.ndarray, *, _err=exc) -> np.ndarray:  # type: ignore[return-type]
                    raise RuntimeError(
                        f"PCA warm start unavailable: {_err}. Path: {pca_path}"
                    )

                warm_starts["pca_warm_start"] = _raise

    return warm_starts


def create_step_helpers(
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> dict[str, Callable[..., np.ndarray | None]]:
    helpers: dict[str, Callable[..., np.ndarray | None]] = {
        "none": lambda *_args, **_kwargs: None,
    }

    checkpoint_to_use = checkpoint_path
    config_to_use = config_path
    data_config_to_use = data_config_path

    if checkpoint_to_use is not None:
        try:
            helpers["neural_step"] = make_neural_step_helper(
                checkpoint_to_use,
                config_to_use,
                data_config_to_use,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to create neural step helper: {exc}")

            def _raise(_: object, *, _err=exc) -> np.ndarray:  # type: ignore[return-type]
                raise RuntimeError(
                    f"Neural step helper unavailable: {_err}. Checkpoint: {checkpoint_to_use}"
                )

            helpers["neural_step"] = _raise

    return helpers


def create_all_preconditioners(
    A: np.ndarray,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    *,
    pca_path: str | Path | None = None,
) -> tuple[
    dict[str, Callable[[np.ndarray], np.ndarray]],
    dict[str, NeuralPreconditionerMetadata | None],
]:
    """Create all preconditioners with optional neural preconditioner metadata.

    Args:
        A: System matrix
        checkpoint_path: Path to neural preconditioner checkpoint
        config_path: Optional model config
        data_config_path: Optional data config
        pca_path: Optional PCA model path

    Returns:
        Tuple of (preconditioners_dict, metadata_dict)
        - preconditioners_dict: Maps name -> preconditioner function
        - metadata_dict: Maps name -> metadata (only for neural preconditioner)
    """
    preconditioners = create_preconditioners(A)
    metadata_dict: dict[str, NeuralPreconditionerMetadata | None] = {}

    # Initialize metadata for classical preconditioners (no metadata)
    for name in preconditioners:
        metadata_dict[name] = None

    if checkpoint_path is not None:
        try:
            precond_fn, neural_metadata = make_neural_preconditioner(
                A,
                checkpoint_path,
                config_path,
                data_config_path,
            )
            preconditioners["neural"] = precond_fn
            metadata_dict["neural"] = neural_metadata
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to create neural preconditioner: {exc}")

    if pca_path is not None:
        try:
            preconditioners["pca"] = make_pca_preconditioner(A, pca_path)
            metadata_dict["pca"] = None
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to create PCA preconditioner: {exc}")

    return preconditioners, metadata_dict
