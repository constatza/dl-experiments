"""Inference helpers shared by CLI scripts and orchestration flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dlkit.interfaces.api import predict_with_config
from dlkit.core.postprocessing import to_plot_data

from ..common import (
    load_config_with_context,
    get_paths_from_config,
    derive_model_identifier,
    sanitize_identifier,
)
from ..experiment_manifest import update_manifest
from ..plotting import plot_parity_and_residuals


def _merge_pred_target_batches(batches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in batches:
        if not isinstance(item, dict):
            continue
        preds = item.get("predictions", {}) or {}
        tgts = item.get("targets", {}) or {}
        flat = {f"pred/{k}": v for k, v in preds.items()}
        flat.update({f"tgt/{k}": v for k, v in tgts.items()})
        merged.append(flat)
    return merged


def _pick_pred_target_arrays(plot_ready: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(plot_ready, list) and plot_ready and isinstance(plot_ready[0], dict):
        preds, tgts = [], []
        for d in plot_ready:
            p = d.get("preds")
            t = d.get("targets")
            if p is None or t is None:
                continue
            preds.append(np.asarray(p).ravel())
            tgts.append(np.asarray(t).ravel())
        if preds and tgts:
            return np.concatenate(preds), np.concatenate(tgts)

    if isinstance(plot_ready, dict):
        keys = list(plot_ready.keys())
        pred_keys = [k for k in keys if k.startswith("pred/")]
        tgt_keys = [k for k in keys if k.startswith("tgt/")]
        if len(pred_keys) == 1 and len(tgt_keys) == 1:
            return (
                np.asarray(plot_ready[pred_keys[0]]).ravel(),
                np.asarray(plot_ready[tgt_keys[0]]).ravel(),
            )
        pred_suffix = {k.split("/", 1)[1]: k for k in pred_keys}
        tgt_suffix = {k.split("/", 1)[1]: k for k in tgt_keys}
        for suffix in pred_suffix:
            if suffix in tgt_suffix:
                return (
                    np.asarray(plot_ready[pred_suffix[suffix]]).ravel(),
                    np.asarray(plot_ready[tgt_suffix[suffix]]).ravel(),
                )
        fallback_pairs = [
            ("pred/y_hat", "tgt/y"),
            ("pred/y", "tgt/y"),
            ("pred/preds", "tgt/y"),
            ("pred/out", "tgt/y"),
            ("pred/logits", "tgt/y"),
        ]
        for pk, tk in fallback_pairs:
            if pk in plot_ready and tk in plot_ready:
                return (
                    np.asarray(plot_ready[pk]).ravel(),
                    np.asarray(plot_ready[tk]).ravel(),
                )

    return None, None


def _derive_run_identifier(
    settings: Any,
    context: Any,
    checkpoint_path: str | Path | None,
    config_path: str | Path,
) -> str:
    if checkpoint_path:
        cp = Path(checkpoint_path)
        parent = cp.parent
        if parent.name.lower() == "checkpoints" and parent.parent.name:
            return parent.parent.name
        if cp.stem:
            return cp.stem

    run_id = getattr(context, "run_id", None)
    if isinstance(run_id, str) and run_id and not run_id.lower().startswith("dlkit-session"):
        return run_id

    return derive_model_identifier(settings, context, config_path)


def run_inference(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    features_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    save_plots: bool = True,
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run inference for parity plot generation using DLKit."""

    settings, context = load_config_with_context(config_path, data_config_path)
    paths = get_paths_from_config(settings, context)

    if features_path is not None:
        features_file = Path(features_path)
    else:
        resolved = paths.get("features_path")
        if resolved is None:
            raise ValueError(
                "No features path specified. Provide features_path or include [DATASET] in config."
            )
        features_file = Path(resolved)

    if targets_path is not None:
        targets_file = Path(targets_path)
    else:
        resolved = paths.get("targets_path")
        if resolved is None:
            raise ValueError(
                "No targets path specified. Provide targets_path or include [DATASET] in config."
            )
        targets_file = Path(resolved)

    checkpoint_to_use = checkpoint_path or paths.get("checkpoint_path")
    if checkpoint_to_use is None:
        raise ValueError("No checkpoint path specified")

    if settings.DATASET is None or not settings.DATASET.features:
        from dlkit.tools.config.data_entries import Feature, Target
        from dlkit.tools.config.dataset_settings import DatasetSettings

        dataset = DatasetSettings(
            name="FlexibleDataset",
            features=(Feature(name="x", path=str(features_file)),),
            targets=(Target(name="y", path=str(targets_file)),),
        )
        settings = settings.model_copy(update={"DATASET": dataset})

    result = predict_with_config(settings, str(checkpoint_to_use))
    predictions = result.predictions

    if (
        isinstance(predictions, list)
        and predictions
        and isinstance(predictions[0], dict)
        and "predictions" in predictions[0]
    ):
        merged = _merge_pred_target_batches(predictions)
        plot_ready = to_plot_data(merged)
    else:
        plot_ready = to_plot_data(predictions)

    y_hat_arr, y_arr = _pick_pred_target_arrays(plot_ready)

    plot_path = None
    if save_plots and y_hat_arr is not None and y_arr is not None:
        from ..common import DEFAULT_FIGURES_DIR

        figures_root = Path(figures_dir) if figures_dir is not None else Path(
            paths.get("figures_dir", DEFAULT_FIGURES_DIR)
        )
        figures_root.mkdir(parents=True, exist_ok=True)

        dataset_id = getattr(getattr(context, "data", None), "dataset_id", None)
        if dataset_id is None and data_config_path is not None:
            dataset_id = Path(data_config_path).stem
        dataset_id = dataset_id or "dataset"

        run_identifier = _derive_run_identifier(
            settings, context, checkpoint_to_use, config_path
        )
        dataset_slug = sanitize_identifier(str(dataset_id))
        run_identifier = sanitize_identifier(run_identifier)
        suffix = f"{dataset_slug}-{run_identifier}"

        plot_path = figures_root / f"parity_residuals_{suffix}.png"
        plot_parity_and_residuals(y_hat_arr, y_arr, sample=0, save_path=plot_path, show=False)

    checkpoint_path_obj = Path(checkpoint_to_use)
    if checkpoint_path_obj.parent.name == "checkpoints":
        experiment_dir = checkpoint_path_obj.parent.parent
        update_manifest(
            experiment_dir,
            "inference",
            {"checkpoint_path": str(checkpoint_path_obj.relative_to(experiment_dir))},
        )

    return {
        "predictions": predictions,
        "y_true": y_arr,
        "y_pred": y_hat_arr,
        "duration_seconds": result.duration_seconds,
        "plot_path": plot_path,
    }
