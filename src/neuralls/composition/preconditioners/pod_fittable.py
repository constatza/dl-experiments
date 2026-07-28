"""dlkit `Fittable` adapter over torchalg's `PODCoarseningStrategy`.

Lives in `composition/preconditioners/` (not `domain/solver/`) per
`domain/solver/README.md`'s boundary rule: all preconditioner algorithms live
in `torchalg`, and `neuralls.domain.solver` production code must not hold
preconditioner algorithm classes. This adapter holds no new math at all — it
inherits `build_transfer`/`compute_pod_basis` from `PODCoarseningStrategy`
unchanged — it exists solely so dlkit's job loader (`[model] module_path`,
resolved via `importlib`) has a concrete, importable class to construct for a
`run.type = "fit"` (POD-2G) assignment, mirroring
`composition/preconditioners/factory.py`'s existing role translating between
neuralls' config layer and `torchalg`'s runtime objects.

`PODCoarseningStrategy.fit(snapshots: torch.Tensor)` takes a raw tensor — its
natural call site is `factory.py`'s inline fit, which already has the
snapshot tensor in hand. dlkit's `FitJobConfig`/`OneShotFitExecutor` pipeline
instead calls `model.fit(dataloader)` (see
`dlkit.engine.training.fittable.Fittable`), passing an iterable of batches,
not a tensor — verified directly against the installed `PODCoarseningStrategy`:
`PODCoarseningStrategy(rank=2).fit(a_real_dataloader)` raises
`AttributeError: 'DataLoader' object has no attribute 'shape'`.

This adapter bridges that gap the same way dlkit's own
`TransformChain.fit_from_dataloader()` bridges `IFittableTransformer
.fit(dataloader)` to each wrapped `Transform.fit(data: Tensor)`
(`dlkit.domain.transforms.chain`) — materialize the target array across the
dataloader's batches, then delegate to the tensor-based `fit()`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)

_DEFAULT_TARGET_NAME = default_training_dataset_contract().target_name


class PODCoarseningFittable(PODCoarseningStrategy):
    """`PODCoarseningStrategy`, usable directly as a `FitJobConfig` job's `[model]`.

    Subclasses `PODCoarseningStrategy` (inheritance, not composition/wrapping)
    so the fitted checkpoint's buffer key stays exactly `_basis` — no wrapper
    attribute prefix in `state_dict()`. `register_buffer("_basis", ...)`
    happens in the parent's `fit()`, called on `self`, so a reconstructed
    instance round-trips through
    `composition/preconditioners/factory.py::_load_fitted_pod_coarsening`
    (`isinstance(model, PODCoarseningStrategy)`, `state_dict` shape) exactly
    like a bare `PODCoarseningStrategy` would — `isinstance` follows
    inheritance, and the checkpoint format is identical either way.
    """

    def __init__(self, rank: float, target_name: str = _DEFAULT_TARGET_NAME) -> None:
        """Store the target rank and dataloader batch key; unfitted until `fit()`.

        Args:
            rank: Fixed mode count (int) or minimum cumulative captured
                energy (float in (0, 1]) — forwarded unchanged to
                `PODCoarseningStrategy.__init__`.
            target_name: Batch key (under `batch["targets"]`) supplying the
                snapshot ensemble to fit from. Defaults to
                `RuntimeDatasetContract.target_name`
                (`composition/assignments/runtime_dataset_contract.py`) —
                the same target key every other assignment in this repo binds
                its supervised target array to.
        """
        super().__init__(rank=rank)
        self._target_name = target_name

    @property
    def hparams(self) -> dict[str, Any]:
        """Minimal Lightning-shaped `hparams` for dlkit's tracking param logger.

        `dlkit.engine.tracking.settings_logger.SettingsLogger.log_model_parameters`
        unconditionally reads `model.hparams` — populated, for the ordinary
        trainer-backed path, by `save_hyperparameters()` inside
        `CoreLightningWrapper`. `OneShotFitExecutor` runs this model directly
        with no Lightning wrapper at all (no `Trainer`, no wrapping step), so
        without this property the tracking decorator's param-logging step
        raises `AttributeError` before `fit()` ever runs — not
        POD-specific, but a gap in any `Fittable` model used through
        `FitJobConfig` today.
        """
        return {"rank": self._rank}

    def fit(self, snapshots: torch.Tensor | Iterable[Any]) -> None:
        """Fit from a raw tensor, or by materializing a dataloader's batches first.

        A strict widening of `PODCoarseningStrategy.fit(snapshots: torch.Tensor)`
        (LSP-compatible override — a bare tensor still fits exactly like the
        parent class): a `torch.Tensor` is forwarded unchanged, so this class
        is substitutable everywhere a plain `PODCoarseningStrategy` is
        (`factory.py`'s inline-fit call site). Anything else is treated as a
        dlkit training dataloader — each batch's `["targets"][target_name]`
        entry is concatenated into one snapshot ensemble before delegating to
        the tensor path, satisfying
        `dlkit.engine.training.fittable.Fittable.fit(dataloader)`.

        Args:
            snapshots: Either the snapshot ensemble directly (shape
                `(n_samples, n_dofs)`), or a dataloader yielding batches
                whose `["targets"][target_name]` entry is this run's
                snapshot array (e.g. a `solutions-cgN` dataset's `y`).

        Raises:
            ValueError: If `snapshots` is a dataloader that yields no batches.
        """
        if isinstance(snapshots, torch.Tensor):
            super().fit(snapshots)
            return
        snapshot_batches = [batch["targets"][self._target_name] for batch in snapshots]
        if not snapshot_batches:
            raise ValueError("PODCoarseningFittable.fit() received an empty dataloader.")
        concatenated = torch.cat([torch.as_tensor(batch) for batch in snapshot_batches], dim=0)
        super().fit(concatenated)
