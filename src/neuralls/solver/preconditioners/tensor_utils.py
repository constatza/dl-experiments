"""Pure tensor conversion utilities - framework-agnostic.

This module provides pure functions for converting between numpy arrays
and PyTorch tensors. These are separated from business logic to enable
testing without GPU and maintain functional programming principles.

Design Principles:
    - Pure functions: No side effects, same input → same output
    - Functional composition: Small, composable utilities
    - Separation: Actions (I/O, GPU) vs pure functions (conversions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import torch


def numpy_to_tensor(
    array: NDArray,
    device: str,
    dtype: type = np.float64,
) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor on specified device.

    Pure function: No side effects, deterministic output.

    Args:
        array: Numpy array (any dtype)
        device: Device string ("cpu", "cuda", "cuda:0", "mps")
        dtype: Target numpy dtype for conversion

    Returns:
        PyTorch tensor on specified device

    Example:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> tensor = numpy_to_tensor(arr, "cuda")
        >>> assert tensor.device.type == "cuda"
    """
    import torch

    arr_typed = array.astype(dtype, copy=False)
    tensor = torch.from_numpy(arr_typed)
    return tensor.to(device)


def tensor_to_numpy(
    tensor: torch.Tensor,
    dtype: type = np.float64,
) -> NDArray:
    """Convert PyTorch tensor to numpy array.

    Pure function: No side effects, deterministic output.

    Args:
        tensor: PyTorch tensor (any device)
        dtype: Target numpy dtype

    Returns:
        Numpy array (CPU)

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        >>> arr = tensor_to_numpy(tensor)
        >>> assert isinstance(arr, np.ndarray)
    """
    arr = tensor.cpu().numpy()
    return arr.astype(dtype, copy=False)


def add_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """Add batch dimension to tensor.

    Pure function: tensor → batched_tensor

    Args:
        tensor: Input tensor of shape (N,) or (N, M, ...)

    Returns:
        Tensor with batch dimension: (1, N) or (1, N, M, ...)

    Example:
        >>> import torch
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>> batched = add_batch_dimension(t)
        >>> assert batched.shape == (1, 3)
    """
    return tensor.unsqueeze(0)


def remove_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    """Remove batch dimension from tensor.

    Pure function: batched_tensor → tensor

    Args:
        tensor: Batched tensor of shape (1, N) or (1, N, M, ...)

    Returns:
        Tensor without batch dimension: (N,) or (N, M, ...)

    Example:
        >>> import torch
        >>> t = torch.tensor([[1.0, 2.0, 3.0]])
        >>> unbatched = remove_batch_dimension(t)
        >>> assert unbatched.shape == (3,)
    """
    return tensor.squeeze(0)


def prepare_model_input(
    residual: NDArray,
    device: str,
) -> torch.Tensor:
    """Prepare numpy residual for model input.

    Composition of pure functions: numpy → tensor → batched

    Args:
        residual: Numpy residual vector
        device: Target device

    Returns:
        Batched tensor ready for model.forward()

    Example:
        >>> res = np.array([1.0, 2.0, 3.0])
        >>> inp = prepare_model_input(res, "cuda")
        >>> assert inp.shape == (1, 3)
        >>> assert inp.device.type == "cuda"
    """
    tensor = numpy_to_tensor(residual, device, dtype=np.float64)
    return add_batch_dimension(tensor)


def extract_model_output(
    output: torch.Tensor,
) -> NDArray:
    """Extract numpy array from model output.

    Composition of pure functions: batched → unbatched → numpy

    Args:
        output: Model output tensor (batched)

    Returns:
        Numpy array (float64)

    Example:
        >>> import torch
        >>> out = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        >>> arr = extract_model_output(out)
        >>> assert arr.shape == (3,)
        >>> assert arr.dtype == np.float64
    """
    unbatched = remove_batch_dimension(output)
    return tensor_to_numpy(unbatched, dtype=np.float64)
