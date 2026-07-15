"""Array export utilities for solver data.

This module provides functions for saving numerical data (matrices, vectors,
iteration histories) in multiple formats (NPZ binary, TXT human-readable).

All functions are I/O actions with clear contracts.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torchalg.monitoring import IterationHistory, TraceMode


from .formats import ArrayFormat


def _history_array(value: object) -> np.ndarray:
    """Convert torchalg vector histories to NumPy arrays for export."""
    to_tensor = getattr(value, "to_tensor", None)
    if callable(to_tensor):
        value = to_tensor()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_system_arrays(
    output_dir: Path,
    A: np.ndarray,
    b: np.ndarray,
    x_sol: np.ndarray,
    x0: np.ndarray | None = None,
    format: ArrayFormat = ArrayFormat.NPZ,
) -> None:
    """Save system matrices and vectors.

    Saves system data in specified format(s):
    - NPZ mode: Single 'system.npz' file with A, b, x0, x_sol
    - TXT mode: Separate A.txt, b.txt, x0.txt, x_sol.txt files
    - BOTH mode: Both NPZ and TXT files

    Args:
        output_dir: Directory to save arrays.
        A: System matrix (n × n).
        b: Right-hand side vector (n,).
        x_sol: Solution vector (n,).
        x0: Initial guess (n,). If None, zeros are used.
        format: Export format (NPZ, TXT, or BOTH).

    Example:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> output_dir = Path("/tmp/solver_output")
        >>> output_dir.mkdir(exist_ok=True)
        >>> A = np.eye(3)
        >>> b = np.ones(3)
        >>> x_sol = np.ones(3)
        >>> save_system_arrays(output_dir, A, b, x_sol, format=ArrayFormat.NPZ)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle x0 (use zeros if not provided)
    x0_array = x0 if x0 is not None else np.zeros_like(b)

    # Save NPZ format
    if format in (ArrayFormat.NPZ, ArrayFormat.BOTH):
        np.savez(
            output_dir / "system.npz",
            A=A,
            b=b,
            x0=x0_array,
            x_sol=x_sol,
        )

    # Save TXT format
    if format in (ArrayFormat.TXT, ArrayFormat.BOTH):
        np.savetxt(output_dir / "A.txt", A)
        np.savetxt(output_dir / "b.txt", b)
        np.savetxt(output_dir / "x0.txt", x0_array)
        np.savetxt(output_dir / "x_sol.txt", x_sol)


def save_iteration_history(
    output_dir: Path,
    history: IterationHistory,
    format: ArrayFormat = ArrayFormat.NPZ,
) -> None:
    """Save iteration history data.

    Saves history based on TraceMode:
    - MINIMAL: Only residual_norms
    - FULL: residual_norms + residuals + solutions + directions (if available)

    Format options:
    - NPZ mode: Single 'history.npz' file with all arrays
    - TXT mode: Separate files (residual_norms.txt, residuals.txt, etc.)
    - BOTH mode: Both NPZ and TXT files

    Args:
        output_dir: Directory to save arrays.
        history: IterationHistory with traced data.
        format: Export format (NPZ, TXT, or BOTH).

    Example:
        >>> from torchalg.monitoring import TraceMode
        >>> history = IterationHistory(mode=TraceMode.MINIMAL)
        >>> history.log_iteration(residual_norm=1.0)
        >>> history.log_iteration(residual_norm=0.5)
        >>> save_iteration_history(Path("/tmp/solver_output"), history)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract arrays from history
    norms_array = np.array(history.residual_norms.to_list())

    # Prepare arrays based on trace mode
    arrays_dict = {"residual_norms": norms_array}

    if history.mode == TraceMode.FULL:
        if history.residuals is not None:
            arrays_dict["residuals"] = _history_array(history.residuals)
        if history.solutions is not None:
            arrays_dict["solutions"] = _history_array(history.solutions)
        if history.directions is not None:
            arrays_dict["directions"] = _history_array(history.directions)

    # Save NPZ format
    if format in (ArrayFormat.NPZ, ArrayFormat.BOTH):
        cast(Any, np.savez)(str(output_dir / "history.npz"), **arrays_dict)

    # Save TXT format
    if format in (ArrayFormat.TXT, ArrayFormat.BOTH):
        np.savetxt(output_dir / "residual_norms.txt", norms_array)

        if history.mode == TraceMode.FULL:
            if history.residuals is not None:
                np.savetxt(output_dir / "residuals.txt", _history_array(history.residuals))
            if history.solutions is not None:
                np.savetxt(output_dir / "solutions.txt", _history_array(history.solutions))
            if history.directions is not None:
                np.savetxt(output_dir / "directions.txt", _history_array(history.directions))
