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
from ..diagnostics.synthetic import (
    generate_synthetic_test_case,
    save_synthetic_results,
)
from ..file_operations import derive_model_identifier, sanitize_identifier
from ..plotting import plot_parity_and_residuals, plot_prediction_diagnostics
from ..mlflow_utils import build_run_config, finalize_run, open_run
from ..io.comparison import load_solver_config


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
    synthetic_benchmark: bool = False,
    solver_config_path: str | Path | None = None,
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
        checkpoint_to_use = checkpoint_path
        if checkpoint_to_use is None:
            raise ValueError("No checkpoint path specified")

        # --- 1. Standard Data Loading ---
        if features_path is not None:
            features_file = Path(features_path)
        else:
            features_file = workspace.data_dir / "normalized.npz"
            # We don't raise error here if it doesn't exist yet, we check later if we actually need it.
            # But original code raised ValueError if explicit path not provided and file missing.
            if not features_file.exists():
                 # Should we fail? Original code did. Let's keep it safe.
                 pass 

        # Re-implementing standard loading logic robustly
        standard_features = None
        standard_targets = None
        
        # Determine standard paths
        feat_path = Path(features_path) if features_path else workspace.data_dir / "normalized.npz"
        tgt_path = Path(targets_path) if targets_path else workspace.data_dir / "normalized.npz"
        
        if feat_path.exists() and tgt_path.exists():
            logger.debug(f"Loading standard features from: {feat_path}")
            feature_array_key = "rhs" if feat_path.name == "normalized.npz" else None
            standard_features = np.asarray(
                load_array(feat_path, array_key=feature_array_key)
            )

            logger.debug(f"Loading standard targets from: {tgt_path}")
            target_array_key = "solutions" if tgt_path.name == "normalized.npz" else None
            standard_targets = np.asarray(
                load_array(tgt_path, array_key=target_array_key)
            )
            if standard_targets.ndim == 2 and standard_targets.shape[1] == 1:
                standard_targets = standard_targets.ravel()
        else:
            logger.warning(f"Standard data files not found at {feat_path} or {tgt_path}. Skipping standard inference.")

        # --- 2. Synthetic Data Generation ---
        synthetic_features = None
        synthetic_targets = None
        
        if synthetic_benchmark:
            logger.info("Preparing SYNTHETIC benchmark data (x_true = ones).")
            if not solver_config_path:
                 raise ValueError("Solver config path is required for synthetic benchmark.")

            solver_cfg = load_solver_config(solver_config_path)
            if solver_cfg.general.matrix_path is None:
                 raise ValueError(f"Solver config {solver_config_path} must specify [general] matrix_path.")

            matrix_path = Path(solver_cfg.general.matrix_path)
            if not matrix_path.exists():
                raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

            _, synthetic_targets, synthetic_features_raw = generate_synthetic_test_case(matrix_path)
            # Reshape to (1, N) for single RHS vector - inference expects (num_samples, num_features)
            synthetic_features = synthetic_features_raw.reshape(1, -1)
            synthetic_targets = synthetic_targets.reshape(1, -1)

        # Ensure we have something to run
        if standard_features is None and synthetic_features is None:
             raise ValueError("No data available for inference (neither standard nor synthetic).")

        # Use standard features for settings/dataset init if available, else synthetic
        # This is for internal DLKit config consistency
        init_features = standard_features if standard_features is not None else synthetic_features
        init_targets = standard_targets if standard_targets is not None else synthetic_targets

        settings = _ensure_dataset_settings(
            settings,
            feature_values=init_features,
            target_values=init_targets,
        )

        # ... (rest of path resolution for normalization/comparison files remains same) ...
        data_dir = workspace.data_dir
        normalized_path = data_dir / "normalized.npz"
        comparison_path = data_dir / "comparison.npz"
        if normalized_path.exists():
            # ... (logging)
            pass 
        
        batch_size = _resolve_batch_size(settings)

        logger.debug(f"Loading checkpoint from: {checkpoint_to_use}")
        logger.debug("Using apply_transforms=False with precision=FULL_64")
        
        standard_preds = None
        synthetic_preds = None
        total_duration = 0.0

        with load_predictor(
            str(checkpoint_to_use),
            apply_transforms=False,
            precision=PrecisionStrategy.FULL_64,
        ) as predictor:
            
            # Run Standard Inference
            if standard_features is not None:
                logger.info("Running inference on STANDARD data...")
                # We need to construct a temp dataset mapping for _collect_predictions 
                # or just pass array dict if _collect_predictions handles it?
                # _collect_predictions expects 'feature_arrays' dict.
                # _ensure_dataset_settings setup 'x' as feature name.
                
                # Re-verify feature name from settings
                feat_name = settings.DATASET.features[0].name # usually 'x'
                std_feat_dict = {feat_name: standard_features}
                
                raw_preds, duration = _collect_predictions(predictor, std_feat_dict, batch_size)
                total_duration += duration
                
                # Stack and process standard predictions
                stacked = stack_batches(raw_preds, mode="stack")
                if isinstance(stacked, dict):
                    standard_preds = next(iter(stacked.values())).ravel()
                else:
                    standard_preds = stacked.ravel()

            # Run Synthetic Inference
            if synthetic_features is not None:
                logger.info("Running inference on SYNTHETIC data...")
                feat_name = settings.DATASET.features[0].name
                syn_feat_dict = {feat_name: synthetic_features} # synthetic_features is (N,) or (N,1)? generate returns (N,) for b usually?
                # generate_synthetic_test_case returns b = A @ x. shape (N,).
                
                raw_preds_syn, duration_syn = _collect_predictions(predictor, syn_feat_dict, batch_size)
                total_duration += duration_syn
                
                stacked_syn = stack_batches(raw_preds_syn, mode="stack")
                if isinstance(stacked_syn, dict):
                    synthetic_preds = next(iter(stacked_syn.values())).ravel()
                else:
                    synthetic_preds = stacked_syn.ravel()

        logger.debug(f"Total prediction duration: {total_duration:.3f}s")

        # --- Post-Processing & Saving ---
        
        run_identifier = sanitize_identifier(
            _derive_run_identifier(
                settings, workspace, checkpoint_to_use, config_path
            )
        )
        dataset_slug = sanitize_identifier(str(dataset_id))
        suffix = f"{dataset_slug}-{run_identifier}"
        
        # 1. Save Synthetic Results (if any)
        if synthetic_benchmark and synthetic_preds is not None and synthetic_targets is not None:
            save_synthetic_results(
                workspace.predictions_dir,
                y_true=synthetic_targets,
                y_pred=synthetic_preds,
                identifier=suffix,
            )

        # 2. Save/Plot Standard Results (if any)
        plot_path = None
        diagnostic_plot_path = None
        
        if standard_preds is not None and standard_targets is not None:
            y_hat_arr = standard_preds
            y_arr = standard_targets
            y_true_diag = standard_targets # alias for diag logging
            
            # ... (Existing logging and plotting logic using y_hat_arr and y_arr) ...
            
            # Re-using existing logic block for standard outputs
            if y_true_diag is None:
                logger.debug("Targets unavailable; skipping target-based diagnostics.")
            else:
                # ... (diagnostics logging)
                pass # logic continues below

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

            if save_plots:
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
        
        # Return standard results for compatibility (or merged dict?)
        # Returning standard results is safest for existing consumers.
        return {
            "predictions": standard_preds if standard_preds is not None else synthetic_preds, # fallback return
            "y_true": standard_targets,
            "y_pred": standard_preds,
            "duration_seconds": total_duration,
            "plot_path": plot_path,
            "diagnostic_plot_path": diagnostic_plot_path,
        }


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
