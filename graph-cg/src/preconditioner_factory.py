"""Factory for creating different types of preconditioners."""

from __future__ import annotations
from pathlib import Path
from collections.abc import Callable
from typing import Literal
import numpy as np
from scipy.sparse.linalg import spilu

from .math_utils import _to_csc
from .neural_inference import create_neural_preconditioner


def make_identity_preconditioner() -> Callable[[np.ndarray], np.ndarray]:
    """Create identity (no-op) preconditioner.

    Returns:
        Identity preconditioner function
    """
    return lambda x: x


def make_jacobi_preconditioner(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create Jacobi (diagonal) preconditioner.

    Args:
        A: System matrix

    Returns:
        Jacobi preconditioner function
    """
    diag = np.diag(A)
    diag_inv = np.where(np.abs(diag) > 1e-15, 1.0 / diag, 1.0)
    return lambda x: diag_inv * x


def make_ilu_preconditioner(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create ILU preconditioner.

    Args:
        A: System matrix

    Returns:
        ILU preconditioner function
    """
    ilu = spilu(_to_csc(A))
    return lambda x: ilu.solve(x)


def create_preconditioner(
    name: Literal["none", "jacobi", "ilu"],
    A: np.ndarray,
    **kwargs
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a preconditioner by name.

    Args:
        name: Preconditioner type
        A: System matrix
        **kwargs: Additional arguments for neural preconditioner

    Returns:
        Preconditioner function
    """
    if name == "none":
        return make_identity_preconditioner()
    elif name == "jacobi":
        return make_jacobi_preconditioner(A)
    elif name == "ilu":
        return make_ilu_preconditioner(A)
    else:
        raise ValueError(f"Unknown preconditioner type: {name}")


def create_preconditioners(A: np.ndarray) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Create classical preconditioners."""

    return {
        "none": create_preconditioner("none", A),
        "jacobi": create_preconditioner("jacobi", A),
        "ilu": create_preconditioner("ilu", A),
    }


def create_warm_starts(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None
) -> dict[str, Callable[[np.ndarray], np.ndarray | None]]:
    """Create warm-start providers."""

    warm_starts: dict[str, Callable[[np.ndarray], np.ndarray | None]] = {
        "none": lambda _: None,
    }

    if checkpoint_path is None or config_path is None:
        return warm_starts

    try:
        predictor = create_neural_preconditioner(checkpoint_path, config_path)
    except Exception as exc:  # noqa: BLE001
        def _raise(_: np.ndarray, *, _err=exc) -> np.ndarray:  # type: ignore[return-type]
            raise RuntimeError(f"Neural warm start unavailable: {_err}. Checkpoint: {checkpoint_path}")

        warm_starts["neural_warm_start"] = _raise
    else:
        def _predict(rhs: np.ndarray, *, _pred=predictor) -> np.ndarray:
            return _pred(rhs)

        warm_starts["neural_warm_start"] = _predict

    return warm_starts


def create_all_preconditioners(
    A: np.ndarray,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None
) -> tuple[
    dict[str, Callable[[np.ndarray], np.ndarray]],
    dict[str, Callable[[np.ndarray], np.ndarray | None]],
]:
    """Create preconditioners and warm-start providers."""

    preconditioners = create_preconditioners(A)
    warm_starts = create_warm_starts(checkpoint_path, config_path)
    return preconditioners, warm_starts
