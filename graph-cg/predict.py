"""Prediction script for graph-cg using new dlkit API.

Revamped to use dlkit.core.postprocessing to prepare plot‑ready arrays.
Generically handles both array and graph-style predictions produced by DLKit
wrappers (list of {predictions, targets, latents} per batch).
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from dlkit.interfaces.api import predict_with_config
from dlkit import GeneralSettings
from dlkit.core.postprocessing import to_plot_data, summarize

from src.common import get_paths_from_config
from src.plotting import plot_parity_and_residuals


def _to_numpy(x):
    """Best-effort conversion to 1D numpy array."""
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(x)
    return x.reshape(-1)


def _merge_pred_target_batches(batches):
    """Flatten list of {'predictions': {...}, 'targets': {...}} to simple dicts.

    Keys are prefixed with 'pred/' and 'tgt/' so they can be stacked by
    dlkit.core.postprocessing.to_plot_data for array-like outputs.
    """
    merged = []
    for item in batches or []:
        if not isinstance(item, dict):
            continue
        preds = item.get("predictions", {}) or {}
        tgts = item.get("targets", {}) or {}
        flat = {f"pred/{k}": v for k, v in preds.items()}
        flat.update({f"tgt/{k}": v for k, v in tgts.items()})
        merged.append(flat)
    return merged


def _pick_pred_target_arrays(plot_ready):
    """Select matching pred/target arrays from plot_ready structure.

    Returns (y_hat, y) as 1D numpy arrays, or (None, None) if not found.
    Handles both graph-style (list of dicts with 'preds'/'targets') and
    array-style dicts produced by _merge_pred_target_batches().
    """
    # Graph/list case with 'preds' and 'targets' per item
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

    # Array/dict case with 'pred/<name>' and 'tgt/<name>' keys
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
        for s in pred_suffix.keys():
            if s in tgt_suffix:
                return (
                    np.asarray(plot_ready[pred_suffix[s]]).ravel(),
                    np.asarray(plot_ready[tgt_suffix[s]]).ravel(),
                )
        # Fallback heuristics
        candidates = [
            ("pred/y_hat", "tgt/y"),
            ("pred/y", "tgt/y"),
            ("pred/preds", "tgt/y"),
            ("pred/out", "tgt/y"),
            ("pred/logits", "tgt/y"),
        ]
        for pk, tk in candidates:
            if pk in plot_ready and tk in plot_ready:
                return (
                    np.asarray(plot_ready[pk]).ravel(),
                    np.asarray(plot_ready[tk]).ravel(),
                )
    return None, None


def run_inference(
    config_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """Run prediction for parity plot generation using training datasets.

    Uses predict_with_config which loads the training config and prediction dataloader
    to generate predictions vs targets for parity plots.

    Args:
        config_path: Path to config file
        checkpoint_path: Override checkpoint path
        save_plots: Whether to save plots

    Returns:
        Dictionary with prediction results
    """
    # Load config
    settings = GeneralSettings.from_toml_file(str(config_path))
    paths = get_paths_from_config(settings)

    # Use provided checkpoint or get from config
    if checkpoint_path is None:
        checkpoint_path = paths.get('checkpoint_path')
        if checkpoint_path is None:
            raise ValueError("No checkpoint path specified")

    # Run prediction using config and checkpoint
    result = predict_with_config(settings, str(checkpoint_path))
    predictions = result.predictions

    # Process predictions for plotting
    y_hat_arr = None
    y_arr = None
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict) and "predictions" in predictions[0]:
        merged = _merge_pred_target_batches(predictions)
        plot_ready = to_plot_data(merged)
        y_hat_arr, y_arr = _pick_pred_target_arrays(plot_ready)
    else:
        plot_ready = to_plot_data(predictions)
        y_hat_arr, y_arr = _pick_pred_target_arrays(plot_ready)

    # Save plots if requested
    plot_path = None
    if save_plots and y_hat_arr is not None and y_arr is not None:
        output_dir = Path(paths.get('output_dir', './output'))
        plot_path = output_dir / "parity_residuals.png"
        plot_parity_and_residuals(y_hat_arr, y_arr, sample=0, save_path=plot_path, show=False)

    return {
        'predictions': predictions,
        'y_true': y_arr,
        'y_pred': y_hat_arr,
        'duration_seconds': result.duration_seconds,
        'plot_path': plot_path
    }


if __name__ == "__main__":
    import typer

    def main(
        config: Path = typer.Option(Path(__file__).parent / "config-ffnn.toml", help="Path to config file"),
        checkpoint: Optional[Path] = typer.Option(None, help="Override checkpoint path"),
        no_plots: bool = typer.Option(False, help="Skip saving plots")
    ):
        """Run inference using config file."""
        print(f"Loading configuration from: {config}")

        try:
            results = run_inference(
                config_path=config,
                checkpoint_path=checkpoint,
                save_plots=not no_plots
            )

            # Print summary
            try:
                print(f"Prediction summary: {summarize(results['predictions'])}")
            except Exception:
                pass

            if results['y_true'] is not None and results['y_pred'] is not None:
                print(f"Generated predictions for {len(results['y_true'])} samples")
                if results['plot_path']:
                    print(f"Saved plots to: {results['plot_path']}")
            else:
                print("Could not extract matching prediction/target arrays for plotting.")

            print(f"Inference completed in {results['duration_seconds']:.2f}s")

        except Exception as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)

    typer.run(main)
