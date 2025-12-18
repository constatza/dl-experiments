"""Tests for scale equivariance in NormScaledConstantWidthFFNN."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from dlkit.core.shape_specs import IShapeSpec, ModelFamily, ShapeSource, create_shape_spec
from dlkit.nn import NormScaledConstantWidthFFNN

INPUT_DIM: Final = 16
HIDDEN_SIZE: Final = 32
NUM_LAYERS: Final = 2
BATCH_SIZE: Final = 4
SCALE: Final = 2.5


def _build_shape_spec(input_dim: int) -> IShapeSpec:
    return create_shape_spec(
        shapes={"x": (input_dim,), "y": (input_dim,)},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.CONFIGURATION,
        default_input="x",
        default_output="y",
    )


def _build_model(input_dim: int) -> NormScaledConstantWidthFFNN:
    return NormScaledConstantWidthFFNN(
        unified_shape=_build_shape_spec(input_dim),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.0,
    )


def test_norm_scaled_constant_width_ffnn_scale_equivariance() -> None:
    torch.manual_seed(0)
    model = _build_model(INPUT_DIM)
    model.eval()

    base = torch.randn(BATCH_SIZE, INPUT_DIM)
    scale = torch.tensor(SCALE, dtype=base.dtype)

    with torch.no_grad():
        output = model(base)
        scaled_output = model(scale * base)

    torch.testing.assert_close(
        scaled_output,
        scale * output,
        rtol=1e-6,
        atol=1e-6,
    )
