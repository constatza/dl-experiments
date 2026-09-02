"""Tests for the dlkit `Fittable` adapter over `PODCoarseningStrategy`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.composition.preconditioners.pod_fittable import PODCoarseningFittable


@pytest.fixture
def snapshot_batches() -> list[dict[str, dict[str, torch.Tensor]]]:
    """Two dataloader-shaped batches whose `targets.y` entries are POD-2G snapshots.

    Mirrors the `TensorDict`-nested-under-"targets" batch shape
    `FlexibleDataset`/`ArrayDataModule` actually yield (see
    `RuntimeDatasetContract.loss_target_key == "targets.y"`), split across two
    batches to exercise concatenation across the whole dataloader, not just
    one batch.
    """
    rng = np.random.default_rng(0)
    first = torch.as_tensor(rng.standard_normal((3, 5)), dtype=torch.float64)
    second = torch.as_tensor(rng.standard_normal((2, 5)), dtype=torch.float64)
    return [{"targets": {"y": first}}, {"targets": {"y": second}}]


def test_fit_concatenates_batches_and_delegates_to_tensor_fit(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """fit(dataloader) materializes all batches' targets.y and fits the basis."""
    model = PODCoarseningFittable(rank=2)

    model.fit(snapshot_batches)

    assert model.is_fitted()
    assert isinstance(model, PODCoarseningStrategy)
    assert model._basis.shape == (5, 2)


def test_fit_matches_direct_tensor_fit(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """Fitting via the dataloader adapter yields the same basis as a direct tensor fit."""
    via_dataloader = PODCoarseningFittable(rank=2)
    via_dataloader.fit(snapshot_batches)

    all_snapshots = torch.cat([batch["targets"]["y"] for batch in snapshot_batches], dim=0)
    via_tensor = PODCoarseningStrategy(rank=2)
    via_tensor.fit(all_snapshots)

    torch.testing.assert_close(via_dataloader._basis, via_tensor._basis)


def test_fit_respects_custom_target_name(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """A non-default target_name reads that key instead of 'y'."""
    renamed_batches = [
        {"targets": {"solutions": batch["targets"]["y"]}} for batch in snapshot_batches
    ]
    model = PODCoarseningFittable(rank=2, target_name="solutions")

    model.fit(renamed_batches)

    assert model.is_fitted()


def test_fit_rejects_empty_dataloader() -> None:
    """An empty dataloader raises a clear ValueError, not an opaque torch error."""
    model = PODCoarseningFittable(rank=2)

    with pytest.raises(ValueError, match="empty dataloader"):
        model.fit([])


def test_hparams_exposes_configured_rank_before_fit() -> None:
    """`.hparams` is readable before fit() — dlkit's tracking logger reads it first."""
    model = PODCoarseningFittable(rank=0.99)

    assert dict(model.hparams) == {"rank": 0.99}
