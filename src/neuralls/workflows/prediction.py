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

from ..configuration.loader import load_experiment
from ..diagnostics import save_prediction_samples_to_csv
from ..diagnostics.synthetic import (
    generate_synthetic_test_case,
    save_synthetic_results,
)
from ..file_operations import derive_model_identifier, sanitize_identifier
from ..plotting import plot_parity_and_residuals, plot_prediction_diagnostics
from ..mlflow_utils import build_run_config, finalize_run, open_run
from ..io.toml_loader import load_solver_config


PREDICTION_ARTIFACTS: tuple[str, ...] = ("figures", "predictions")


# NOTE: Minimal dataset functions removed - no longer needed with InferenceWorkflowConfig
# Transforms are loaded from checkpoint metadata (apply_transforms=True)
# DATASET section is optional for inference mode


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


def _flatten_predictions(preds: Any) -> dict[str, Any]:
    """Flatten predictions to keyed dict.

    Args:
        preds: Predictions (dict or array/tensor)

    Returns:
        Dict with pred/* keys
    """
    match preds:
        case dict():
            return {f"pred/{k}": v for k, v in preds.items()}
        case _:
            return {"pred/y_hat": preds}


def _flatten_targets(tgts: Any) -> dict[str, Any]:
    """Flatten targets to keyed dict.

    Args:
        tgts: Targets (dict or array/tensor)

    Returns:
        Dict with tgt/* keys
    """
    match tgts:
        case dict():
            return {f"tgt/{k}": v for k, v in tgts.items()}
        case _:
            return {"tgt/y": tgts}


def _merge_pred_target_batches(
    batches: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge prediction/target batches into flat keyed dicts.

    Args:
        batches: Iterable of batch dictionaries

    Returns:
        List of flattened dictionaries with pred/* and tgt/* keys
    """
    merged = []

    for item in batches:
        # Guard: Skip non-dict items
        if not isinstance(item, dict):
            continue

        preds = item.get("predictions") or {}
        tgts = item.get("targets") or {}

        # Flatten predictions
        flat = _flatten_predictions(preds)

        # Flatten targets
        flat.update(_flatten_targets(tgts))

        merged.append(flat)

    return merged


# Expected keys for prediction/target extraction (fail-fast validation)
EXPECTED_PRED_KEY = "predictions"
EXPECTED_TARGET_KEY = "targets"


def _extract_from_dict(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract arrays from dict with strict key validation.

    Args:
        data: Dict with 'predictions' and 'targets' keys

    Returns:
        Tuple of (predictions, targets) as flattened arrays

    Raises:
        KeyError: If required keys missing
        ValueError: If arrays have incompatible shapes
    """
    # Strict validation - fail fast if keys missing
    if EXPECTED_PRED_KEY not in data:
        raise KeyError(
            f"Missing required key '{EXPECTED_PRED_KEY}'. "
            f"Available keys: {list(data.keys())}"
        )

    if EXPECTED_TARGET_KEY not in data:
        raise KeyError(
            f"Missing required key '{EXPECTED_TARGET_KEY}'. "
            f"Available keys: {list(data.keys())}"
        )

    preds = np.asarray(data[EXPECTED_PRED_KEY]).ravel()
    targets = np.asarray(data[EXPECTED_TARGET_KEY]).ravel()

    # Validate shapes match
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {preds.shape} != targets {targets.shape}"
        )

    return preds, targets


def _extract_from_list(data_list: list) -> tuple[np.ndarray, np.ndarray]:
    """Extract and concatenate arrays from list of dicts.

    Args:
        data_list: List of dicts, each with 'preds' and 'targets' keys

    Returns:
        Tuple of concatenated (predictions, targets)

    Raises:
        ValueError: If list is empty or contains invalid data
        KeyError: If required keys missing from any dict
    """
    # Guard: Validate list not empty
    if not data_list:
        raise ValueError("Cannot extract from empty list")

    # Guard: Validate first element is dict
    if not isinstance(data_list[0], dict):
        raise ValueError(
            f"Invalid list element type: {type(data_list[0]).__name__}. "
            "Expected list of dicts."
        )

    preds_list = []
    targets_list = []

    for idx, item in enumerate(data_list):
        # Strict validation per item
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid element at index {idx}: {type(item).__name__}. "
                "Expected dict."
            )

        if "preds" not in item:
            raise KeyError(
                f"Missing 'preds' key in element {idx}. "
                f"Available keys: {list(item.keys())}"
            )

        if "targets" not in item:
            raise KeyError(
                f"Missing 'targets' key in element {idx}. "
                f"Available keys: {list(item.keys())}"
            )

        preds_list.append(np.asarray(item["preds"]).ravel())
        targets_list.append(np.asarray(item["targets"]).ravel())

    return np.concatenate(preds_list), np.concatenate(targets_list)


def _pick_pred_target_arrays(plot_ready: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract prediction and target arrays with strict validation.

    Enforces expected format and fails fast with clear errors.
    Reduces error surface by not supporting multiple naming variations.

    Expected formats:
    1. Dict with 'predictions' and 'targets' keys
    2. List of dicts, each with 'preds' and 'targets' keys

    Args:
        plot_ready: Data structure containing predictions and targets

    Returns:
        Tuple of (predictions, targets) as flattened numpy arrays

    Raises:
        ValueError: If data doesn't match expected format
        KeyError: If required keys are missing
    """
    match plot_ready:
        case dict():
            return _extract_from_dict(plot_ready)
        case list():
            return _extract_from_list(plot_ready)
        case _:
            raise ValueError(
                f"Invalid data type: {type(plot_ready).__name__}. "
                "Expected dict or list."
            )


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
    """Run inference for parity plot generation.

    This is a pure orchestration function that composes single-responsibility
    functions from the inference module. No business logic here - only composition.

    Args:
        config_path: Path to model configuration file
        data_config_path: Path to data configuration file (optional)
        checkpoint_path: Path to model checkpoint (required)
        features_path: Path to features file (for standard inference)
        targets_path: Path to targets file (for standard inference)
        save_plots: Whether to generate diagnostic plots
        figures_dir: Custom figures directory (optional)
        enable_mlflow: Whether to log to MLflow
        output_root: Custom output root directory (optional)
        synthetic_benchmark: Use synthetic benchmark data
        solver_config_path: Path to solver config (for synthetic)

    Returns:
        Dictionary with prediction results for backward compatibility:
            - predictions: Predicted values
            - y_true: True target values
            - y_pred: Predicted values (duplicate for compatibility)
            - duration_seconds: Inference duration
            - plot_path: Path to parity/residuals plot
            - diagnostic_plot_path: Path to diagnostics plot

    Raises:
        ValueError: If no checkpoint path or no data available
        FileNotFoundError: If required files don't exist
    """
    from neuralls.workflows.inference import (
        load_inference_data,
        create_predictor,
        run_prediction,
        save_inference_outputs,
        start_mlflow_run,
        finalize_mlflow_run,
        save_synthetic_predictions,
    )

    # 1. Validate inputs
    if checkpoint_path is None:
        raise ValueError("No checkpoint path specified")

    checkpoint = Path(checkpoint_path)
    config = Path(config_path)

    # 2. Load experiment configuration in INFERENCE MODE
    # Inference mode uses InferenceWorkflowConfig (DATASET/DATAMODULE optional)
    # Transforms are loaded from checkpoint metadata
    experiment = load_experiment(
        config,
        data_config_path,
        output_root=output_root,
        mode="inference",  # Use InferenceWorkflowConfig
    )
    settings = experiment.settings
    workspace = experiment.workspace
    dataset_id = experiment.spec.data_config_path.stem

    # 3. Start MLflow run (if enabled)
    mlflow_state = start_mlflow_run(
        settings,
        workspace,
        dataset_id,
        enable_mlflow,
    )

    metrics: dict[str, float] | None = None
    error: Exception | None = None

    try:
        # 4. Load data (strategy pattern: standard vs synthetic)
        data = load_inference_data(
            workspace=workspace,
            features_path=Path(features_path) if features_path else None,
            targets_path=Path(targets_path) if targets_path else None,
            synthetic_benchmark=synthetic_benchmark,
            solver_config_path=Path(solver_config_path) if solver_config_path else None,
        )

        # 5. Run prediction (transforms applied automatically from checkpoint)
        # No DATASET configuration needed - InferenceWorkflowConfig handles this
        with create_predictor(checkpoint, settings) as predictor:
            predictions = run_prediction(predictor, data, settings)

        # 7. Save synthetic results separately (if applicable)
        if synthetic_benchmark and data.metadata.get("source") == "synthetic":
            from neuralls.file_operations import sanitize_identifier
            run_identifier = sanitize_identifier(
                str(workspace.run_id or checkpoint.stem)
            )
            dataset_slug = sanitize_identifier(str(dataset_id))
            identifier = f"{dataset_slug}-{run_identifier}"
            save_synthetic_predictions(predictions, workspace.predictions_dir, identifier)

        # 8. Save outputs (CSV + plots)
        outputs = save_inference_outputs(
            predictions=predictions,
            workspace=workspace,
            settings=settings,
            checkpoint_path=checkpoint,
            config_path=config,
            dataset_id=dataset_id,
            save_plots=save_plots,
            figures_dir=Path(figures_dir) if figures_dir else None,
        )

        # 9. Store metrics for MLflow
        metrics = outputs.metrics

        # 10. Build backward-compatible result
        plot_paths = outputs.plot_paths
        return {
            "predictions": predictions.predictions.get("y_pred"),
            "y_true": predictions.targets.get("y_true"),
            "y_pred": predictions.predictions.get("y_pred"),
            "duration_seconds": metrics.get("duration_seconds", 0.0),
            "plot_path": plot_paths[0] if len(plot_paths) > 0 else None,
            "diagnostic_plot_path": plot_paths[1] if len(plot_paths) > 1 else None,
        }

    except Exception as exc:  # noqa: BLE001
        error = exc
        raise
    finally:
        # 11. Finalize MLflow run
        if mlflow_state is not None and metrics is not None:
            finalize_mlflow_run(mlflow_state, metrics, workspace)
        elif mlflow_state is not None:
            # Cleanup even if metrics weren't set
            from neuralls.mlflow_utils import finalize_run
            finalize_run(
                mlflow_state,
                metrics={},
                workspace_root=workspace.root_dir,
                allowlist=PREDICTION_ARTIFACTS,
                failed=error is not None,
            )
