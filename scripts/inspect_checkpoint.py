#!/usr/bin/env python3
"""Inspect checkpoint to verify weights are trained (not random).

This script loads a checkpoint and analyzes the weight distributions to determine
if the model has been trained or if weights are still at random initialization.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def compute_weight_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Compute statistics for all weights in the model.

    Args:
        state_dict: Model state dictionary containing weights

    Returns:
        Dictionary with statistics for each weight tensor
    """
    stats = {}

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue

        param_np = param.detach().cpu().numpy()

        stats[name] = {
            "shape": param_np.shape,
            "mean": float(np.mean(param_np)),
            "std": float(np.std(param_np)),
            "min": float(np.min(param_np)),
            "max": float(np.max(param_np)),
            "abs_mean": float(np.mean(np.abs(param_np))),
            "abs_max": float(np.max(np.abs(param_np))),
            "num_zeros": int(np.sum(param_np == 0)),
            "total_params": int(np.prod(param_np.shape)),
        }

    return stats


def is_likely_trained(stats: dict[str, Any]) -> bool:
    """Determine if weights look trained vs random initialization.

    Heuristics:
    - Trained models typically have diverse weight magnitudes
    - Random init has consistent std across layers
    - Trained models often have specific patterns (e.g., output layer different from hidden)

    Args:
        stats: Weight statistics from compute_weight_stats

    Returns:
        True if weights appear trained, False if likely random
    """
    if not stats:
        return False

    # Get all std deviations
    stds = [s["std"] for s in stats.values() if s["std"] > 0]
    abs_means = [s["abs_mean"] for s in stats.values()]

    if not stds:
        return False

    # Check for diversity in weight statistics
    std_of_stds = float(np.std(stds))
    mean_of_stds = float(np.mean(stds))

    # Trained models typically have more diverse weight distributions
    # Random init is very uniform
    diversity_ratio = std_of_stds / mean_of_stds if mean_of_stds > 0 else 0

    # Check if any layer has significantly different stats (sign of training)
    max_abs_mean = max(abs_means) if abs_means else 0
    min_abs_mean = min(abs_means) if abs_means else 0

    print("\n=== Training Detection Heuristics ===")
    print(f"Diversity ratio (std of stds / mean of stds): {diversity_ratio:.4f}")
    print("  - Random init typically < 0.3")
    print("  - Trained models typically > 0.5")
    print(f"Max abs mean: {max_abs_mean:.4f}")
    print(f"Min abs mean: {min_abs_mean:.4f}")
    print(f"Range: {max_abs_mean - min_abs_mean:.4f}")

    # If diversity ratio is high, likely trained
    is_trained = diversity_ratio > 0.4

    return is_trained


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Print checkpoint structure
    print("\n=== Checkpoint Contents ===")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # Get model state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint

    # Compute statistics
    print("\n=== Weight Statistics ===")
    stats = compute_weight_stats(state_dict)

    print(f"Total layers: {len(stats)}")
    print(f"Total parameters: {sum(s['total_params'] for s in stats.values())}")

    # Print per-layer stats
    print("\n=== Per-Layer Analysis ===")
    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Shape: {s['shape']}")
        print(f"  Mean: {s['mean']:.6f}, Std: {s['std']:.6f}")
        print(f"  Min: {s['min']:.6f}, Max: {s['max']:.6f}")
        print(f"  Abs Mean: {s['abs_mean']:.6f}, Abs Max: {s['abs_max']:.6f}")
        print(
            f"  Zeros: {s['num_zeros']} / {s['total_params']} ({s['num_zeros'] / s['total_params'] * 100:.2f}%)"
        )

    # Determine if trained
    is_trained = is_likely_trained(stats)

    print("\n=== VERDICT ===")
    if is_trained:
        print("✓ Weights appear TRAINED (not random initialization)")
        print("  The model has learned patterns from data.")
    else:
        print("✗ Weights appear RANDOM (not trained)")
        print("  The model may be at initialization or training failed.")

    # Check for transforms in checkpoint
    print("\n=== Transform Metadata ===")
    if "transforms" in checkpoint:
        print(f"Transforms found: {checkpoint['transforms']}")
    elif "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
        if "transforms" in hp:
            print(f"Transforms in hyperparameters: {hp['transforms']}")
        else:
            print("No transforms found in hyperparameters")
    else:
        print("No transform metadata found in checkpoint")

    # Additional useful info
    if "epoch" in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    if "global_step" in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")


if __name__ == "__main__":
    main()
