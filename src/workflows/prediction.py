"""Inference helpers shared by CLI scripts and orchestration flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator

import numpy as np
from loguru import logger

from dlkit import load_predictor
from dlkit.core.postprocessing import stack_batches
from dlkit.tools.config.precision.strategy import PrecisionStrategy
from dlkit.tools.io import load_array

from ..configuration import load_experiment
from ..diagnostics import save_prediction_samples_to_csv
from ..file_operations import derive_model_identifier, sanitize_identifier
from ..plotting import plot_parity_and_residuals, plot_prediction_diagnostics
from ..mlflow_utils import build_run_config, finalize_run, open_run


PREDICTION_ARTIFACTS: tuple[str, ...] = ("figures", "predictions")


def _ensure_dataset_settings(
    settings: Any,
    *,
    feature_values: np.ndarray,
    target_values: np.ndarray,
) -> Any:
    """Guarantee that DATASET entries exist and reflect explicit overrides.

    Note: Transforms should NOT be defined here - they are loaded from the checkpoint.
    This function loads arrays eagerly and injects Value-based entries so the dataset
    never relies on file paths at runtime.
    """
    from dlkit.tools.config.data_entries import Feature, FeatureType, Target, TargetType
    from dlkit.tools.config.dataset_settings import DatasetSettings

    dataset = settings.DATASET
    if dataset is None or not dataset.features:
        # Minimal dataset for loading data - transforms come from checkpoint
        dataset = DatasetSettings(name="FlexibleDataset")
        dataset = dataset.model_copy(
            update={
                "features": [Feature(name="x", value=feature_values, path=None)],
                "targets": [Target(name="y", value=target_values, path=None)],
            }
        )
        return settings.model_copy(update={"DATASET": dataset})

    features: list[FeatureType] = list(dataset.features)
    targets: list[TargetType] = list(dataset.targets or ())

    if features:
        features[0] = features[0].model_copy(
            update={"value": feature_values, "path": None}
        )
    else:
        features = [Feature(name="x", value=feature_values, path=None)]

    if targets:
        targets[0] = targets[0].model_copy(
            update={"value": target_values, "path": None}
        )
    else:
        targets = [Target(name="y", value=target_values, path=None)]

    updated_dataset = dataset.model_copy(
        update={
            "features": list(features) if features else dataset.features,
            "targets": list(targets) if targets else dataset.targets,
        }
    )
    return settings.model_copy(update={"DATASET": updated_dataset})


def _load_feature_arrays(entries: Iterable[Any]) -> dict[str, np.ndarray]:
    """Load feature tensors into numpy arrays keyed by entry name."""
    feature_arrays: dict[str, np.ndarray] = {}
    for entry in entries:
        value = getattr(entry, "value", None)
        if value is None and hasattr(entry, "get_value"):
            value = entry.get_value()

        if value is not None:
            array = np.asarray(value)
        else:
            tensor = load_array(entry.path)
            array = (
                tensor.cpu().numpy()
                if hasattr(tensor, "cpu") and hasattr(tensor, "numpy")
                else np.asarray(tensor)
            )

        feature_arrays[entry.name] = array
    if not feature_arrays:
        raise ValueError("No feature entries defined in DATASET configuration.")
    lengths = {array.shape[0] for array in feature_arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Feature entries must share the same leading dimension.")
    return feature_arrays


def _iterate_feature_batches(
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield contiguous feature batches to stream through the predictor."""
    total = next(iter(feature_arrays.values())).shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {name: array[start:end] for name, array in feature_arrays.items()}


def _resolve_batch_size(settings: Any) -> int:
    """Reuse dataloader batch size when available, else default to 256."""
    datamodule = getattr(settings, "DATAMODULE", None)
    dataloader = getattr(datamodule, "dataloader", None) if datamodule else None
    configured = getattr(dataloader, "batch_size", None)
    try:
        return int(configured) if configured is not None else 256
    except (TypeError, ValueError):
        return 256


def _collect_predictions(
    predictor: Any,
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[list[Any], float]:
    """Run inference for every feature batch while keeping cumulative duration."""
    predictions: list[Any] = []
    total_duration = 0.0
    for batch in _iterate_feature_batches(feature_arrays, batch_size):
        result = predictor.predict(batch)
        predictions.append(result.predictions)
        total_duration += getattr(result, "duration_seconds", 0.0)
    if not predictions:
        raise ValueError("Predictor returned no predictions.")
    return predictions, total_duration


def _merge_pred_target_batches(
    batches: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in batches:
        if not isinstance(item, dict):
            continue
        preds = item.get("predictions")
        tgts = item.get("targets")

        # Handle case where predictions/targets might be tensors/arrays directly
        if preds is None or (isinstance(preds, dict) and not preds):
            preds = {}
        if tgts is None or (isinstance(tgts, dict) and not tgts):
            tgts = {}

        if isinstance(preds, dict):
            flat = {f"pred/{k}": v for k, v in preds.items()}
        else:
            # If predictions is a tensor/array, use as-is with default key
            flat = {"pred/y_hat": preds}

        if isinstance(tgts, dict):
            flat.update({f"tgt/{k}": v for k, v in tgts.items()})
        else:
            # If targets is a tensor/array, use as-is with default key
            flat.update({"tgt/y": tgts})

        merged.append(flat)
    return merged


def _pick_pred_target_arrays(
    plot_ready: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def _summarize_vector(values: np.ndarray) -> dict[str, float]:
    """Compute quick numeric summaries for debug logging."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "l2": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "l2": float(np.linalg.norm(arr)),
    }


def _log_prediction_diagnostics(
    y_true: np.ndarray | None, y_pred: np.ndarray | None
) -> None:
    """Emit debug diagnostics comparing prediction and target magnitudes."""
    if y_true is None or y_pred is None:
        return
    if y_true.size != y_pred.size:
        logger.debug(
            f"Skipping prediction diagnostics due to mismatched sizes: "
            f"y_true={y_true.size}, y_pred={y_pred.size}"
        )
        return

    stats_true = _summarize_vector(y_true)
    stats_pred = _summarize_vector(y_pred)
    denom = max(stats_true["l2"], 1e-12)
    norm_ratio = stats_pred["l2"] / denom
    max_abs = float(np.max(np.abs(y_pred - y_true)))
    mean_abs = float(np.mean(np.abs(y_pred - y_true)))

    logger.debug(
        (
            "Prediction diagnostics | ||y_hat||={pred_l2:.3e}, ||y||={true_l2:.3e}, "
            "ratio={ratio:.3e}, mean_abs={mean_abs:.3e}, max_abs={max_abs:.3e}"
        ).format(
            pred_l2=stats_pred["l2"],
            true_l2=stats_true["l2"],
            ratio=norm_ratio,
            mean_abs=mean_abs,
            max_abs=max_abs,
        )
    )
    logger.debug(
        (
            "Prediction summary | "
            "y_hat[min={pmin:.3e}, max={pmax:.3e}, mean={pmean:.3e}, std={pstd:.3e}] | "
            "y[min={tmin:.3e}, max={tmax:.3e}, mean={tmean:.3e}, std={tstd:.3e}]"
        ).format(
            pmin=stats_pred["min"],
            pmax=stats_pred["max"],
            pmean=stats_pred["mean"],
            pstd=stats_pred["std"],
            tmin=stats_true["min"],
            tmax=stats_true["max"],
            tmean=stats_true["mean"],
            tstd=stats_true["std"],
        )
    )

    if norm_ratio > 10 or norm_ratio < 0.1:
        logger.warning(
            f"Prediction/target norms differ by more than 10x (ratio={norm_ratio:.3e}). "
            "Check normalization settings and dataset transforms."
        )


def _session_name(settings: Any) -> str | None:
    session = getattr(settings, "SESSION", None)
    name = getattr(session, "name", None) if session else None
    return name if isinstance(name, str) and name else None


def _start_prediction_run(
    settings: Any,
    workspace: Any,
    dataset_id: str,
    enable_mlflow: bool,
):
    config = build_run_config(
        settings=settings,
        workspace_root=workspace.root_dir,
        dataset_id=dataset_id,
        model_name=workspace.run_id,
        session_name=_session_name(settings),
        enabled=enable_mlflow,
    )
    if config is None:
        return None
    try:
        return open_run(config)
    except ModuleNotFoundError:
        logger.info("MLflow not installed; skipping MLflow logging.")
        return None


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
    if (
        isinstance(run_id, str)
        and run_id
        and not run_id.lower().startswith("dlkit-session")
    ):
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
    enable_mlflow: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run inference for parity plot generation using DLKit."""
    experiment = load_experiment(
        config_path,
        data_config_path,
        output_root=output_root,
    )
    settings = experiment.settings
    workspace = experiment.workspace
    dataset_id = experiment.spec.data_config_path.stem
    mlflow_state = _start_prediction_run(
        settings,
        workspace,
        dataset_id,
        enable_mlflow,
    )
    total_duration = 0.0
    metrics: dict[str, float] | None = None
    error: Exception | None = None

    try:
        if features_path is not None:
            features_file = Path(features_path)
        else:
            features_file = workspace.data_dir / "normalized.npz"
            if not features_file.exists():
                raise ValueError(
                    "No features path specified. Provide features_path or include [DATASET] in config."
                )

        if targets_path is not None:
            targets_file = Path(targets_path)
        else:
            targets_file = workspace.data_dir / "normalized.npz"
            if not targets_file.exists():
                raise ValueError(
                    "No targets path specified. Provide targets_path or include [DATASET] in config."
                )

        checkpoint_to_use = checkpoint_path
        if checkpoint_to_use is None:
            raise ValueError("No checkpoint path specified")

        logger.debug(f"Loading features from: {features_file}")
        feature_array_key = "rhs" if features_file.name == "normalized.npz" else None
        feature_values = np.asarray(
            load_array(features_file, array_key=feature_array_key)  # type: ignore[arg-type]
        )

        logger.debug(f"Loading targets from: {targets_file}")
        target_array_key = "solutions" if targets_file.name == "normalized.npz" else None
        target_values = np.asarray(
            load_array(targets_file, array_key=target_array_key)  # type: ignore[arg-type]
        )
        if target_values.ndim == 2 and target_values.shape[1] == 1:
            target_values = target_values.ravel()

        settings = _ensure_dataset_settings(
            settings,
            feature_values=feature_values,
            target_values=target_values,
        )

        data_dir = targets_file.parent
        normalized_path = data_dir / "normalized.npz"
        comparison_path = data_dir / "comparison.npz"
        if normalized_path.exists():
            logger.debug(f"Found normalized.npz at: {normalized_path}")
            norm_data = np.load(normalized_path)
            if "normalize_type" in norm_data:
                logger.debug(f"  Normalization type: {norm_data['normalize_type']}")
            if "spectral_radius_bound" in norm_data:
                logger.debug(
                    f"  Spectral radius bound: {norm_data['spectral_radius_bound']:.6e}"
                )
            if "dimension_scale" in norm_data:
                logger.debug(f"  Dimension scale: {norm_data['dimension_scale']:.6e}")
        if comparison_path.exists():
            logger.debug(f"Found comparison.npz at: {comparison_path}")
            comp_data = np.load(comparison_path)
            if "normalize_type" in comp_data:
                logger.debug(f"  Normalization type: {comp_data['normalize_type']}")
            if "spectral_radius_bound" in comp_data:
                logger.debug(
                    f"  Spectral radius bound: {comp_data['spectral_radius_bound']:.6e}"
                )
            if "dimension_scale" in comp_data:
                logger.debug(f"  Dimension scale: {comp_data['dimension_scale']:.6e}")

        y_arr = target_values.astype(np.float64, copy=False)
        logger.debug(
            f"Targets shape: {y_arr.shape}, range: [{y_arr.min():.3e}, {y_arr.max():.3e}], L2 norm: {np.linalg.norm(y_arr):.3e}"
        )

        feature_arrays = _load_feature_arrays(settings.DATASET.features)
        for name, arr in feature_arrays.items():
            logger.debug(
                f"Feature '{name}' shape: {arr.shape}, range: [{arr.min():.3e}, {arr.max():.3e}], L2 norm: {np.linalg.norm(arr):.3e}"
            )
        batch_size = _resolve_batch_size(settings)

        logger.debug(f"Loading checkpoint from: {checkpoint_to_use}")
        logger.debug("Using apply_transforms=False with precision=FULL_64")
        with load_predictor(
            str(checkpoint_to_use),
            apply_transforms=False,
            precision=PrecisionStrategy.FULL_64,
        ) as predictor:
            raw_predictions, total_duration = _collect_predictions(
                predictor,
                feature_arrays,
                batch_size,
            )
            logger.debug(f"Prediction completed in {total_duration:.3f}s")

        stacked = stack_batches(raw_predictions, mode="stack")
        predictions = raw_predictions

        if isinstance(stacked, dict):
            y_hat_arr = next(iter(stacked.values()))
        else:
            y_hat_arr = stacked

        logger.debug(
            f"Raw predictions shape: {y_hat_arr.shape}, range: [{y_hat_arr.min():.3e}, {y_hat_arr.max():.3e}], L2 norm: {np.linalg.norm(y_hat_arr):.3e}"
        )

        y_hat_arr = y_hat_arr.ravel()
        y_true_diag = y_arr.reshape(-1) if y_arr is not None else None

        if y_true_diag is None:
            logger.debug("Targets unavailable; skipping target-based diagnostics.")
        else:
            true_norm = float(np.linalg.norm(y_true_diag))
            pred_norm = float(np.linalg.norm(y_hat_arr))
            error = y_hat_arr - y_true_diag
            error_norm = float(np.linalg.norm(error))
            rel_denom = max(true_norm, 1e-12)

            logger.debug("=" * 80)
            logger.debug("PREDICTION VS TARGET COMPARISON:")
            logger.debug(f"  Predictions L2 norm: {pred_norm:.6e}")
            logger.debug(f"  Targets L2 norm:     {true_norm:.6e}")
            logger.debug(f"  L2 norm ratio (pred/target): {pred_norm / rel_denom:.6e}")
            logger.debug(f"  Error L2 norm:       {error_norm:.6e}")
            logger.debug(f"  Relative error:      {error_norm / rel_denom:.6e}")
            logger.debug("=" * 80)

            _log_prediction_diagnostics(y_true_diag, y_hat_arr)

        plot_path = None
        diagnostic_plot_path = None
        dataset_slug = sanitize_identifier(str(dataset_id))
        run_identifier = sanitize_identifier(
            _derive_run_identifier(
                settings, workspace, checkpoint_to_use, config_path
            )
        )
        suffix = f"{dataset_slug}-{run_identifier}"

        if y_arr is not None and y_hat_arr is not None:
            diagnostics_root = workspace.predictions_dir
            sample_csv_paths = save_prediction_samples_to_csv(
                y_true=y_arr,
                y_pred=y_hat_arr,
                output_dir=diagnostics_root,
                filename_prefix=suffix,
            )
            if sample_csv_paths:
                logger.info(
                    f"Saved prediction samples to CSV: {[str(p) for p in sample_csv_paths]}"
                )

        if (
            save_plots
            and y_hat_arr is not None
            and y_true_diag is not None
            and y_arr is not None
        ):
            figures_root = (
                Path(figures_dir) if figures_dir is not None else workspace.figures_dir
            )
            figures_root.mkdir(parents=True, exist_ok=True)

            plot_path = figures_root / f"parity_residuals_{suffix}.png"
            plot_parity_and_residuals(
                y_hat_arr, y_arr, sample=0, save_path=plot_path, show=False
            )

            diagnostic_plot_path = figures_root / f"diagnostics_{suffix}.png"
            plot_prediction_diagnostics(
                y_hat_arr, y_arr, sample=0, save_path=diagnostic_plot_path, show=False
            )

        metrics = {"duration_seconds": float(total_duration)}
        return {
            "predictions": predictions,
            "y_true": y_arr,
            "y_pred": y_hat_arr,
            "duration_seconds": total_duration,
            "plot_path": plot_path,
            "diagnostic_plot_path": diagnostic_plot_path,
        }
    except Exception as exc:  # noqa: BLE001
        error = exc
        raise
    finally:
        finalize_run(
            mlflow_state,
            metrics=metrics,
            workspace_root=workspace.root_dir,
            allowlist=PREDICTION_ARTIFACTS,
            failed=error is not None,
        )
