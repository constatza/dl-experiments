#!/usr/bin/env python3
"""Quick test to verify prediction quality after normalization fix.

This script loads a trained model, makes predictions, and checks if they
match the target scale (not "rubbish").
"""

import sys
from pathlib import Path

import numpy as np
from dlkit.interfaces.api import load_predictor

def main() -> None:
    """Test prediction quality."""
    # Find checkpoint
    checkpoint_path = Path("/data/projects/graph-cg/data/output/collect-504-solutions/linear/checkpoints/linear.ckpt")

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load test data from normalized.npz
    data_dir = Path("/data/projects/graph-cg/data/processed/collect-504-solutions")
    normalized_data = np.load(data_dir / "normalized.npz")
    rhs_test = normalized_data["rhs"][:20]  # First 20 samples
    sol_test = normalized_data["solutions"][:20]

    print("=" * 70)
    print("PREDICTION QUALITY TEST (After Normalization Fix)")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Test samples: {len(rhs_test)}")

    # Test with apply_transforms=True (current behavior)
    print("\n--- Loading predictor with apply_transforms=True ---")
    with load_predictor(str(checkpoint_path), apply_transforms=True, batch_size=20) as predictor:
        predictions = predictor.predict(rhs_test)

    # Extract predictions from InferenceResult
    if hasattr(predictions, 'predictions'):
        # InferenceResult object
        pred_dict = predictions.predictions
        pred_arr = next(iter(pred_dict.values()))
    elif isinstance(predictions, dict):
        pred_arr = next(iter(predictions.values()))
    else:
        pred_arr = predictions

    # Convert to numpy if needed
    if hasattr(pred_arr, 'numpy'):
        pred_arr = pred_arr.numpy()

    # Compute statistics
    pred_norms = np.linalg.norm(pred_arr, axis=-1, ord=2)
    target_norms = np.linalg.norm(sol_test, axis=-1, ord=2)
    input_norms = np.linalg.norm(rhs_test, axis=-1, ord=2)

    # Compute errors
    abs_errors = np.linalg.norm(pred_arr - sol_test, axis=-1, ord=2)
    rel_errors = abs_errors / (target_norms + 1e-10)

    print("\n=== Input Statistics ===")
    print(f"Input (RHS) norms:")
    print(f"  Mean: {np.mean(input_norms):.6e}")
    print(f"  Std:  {np.std(input_norms):.6e}")
    print(f"  Min:  {np.min(input_norms):.6e}")
    print(f"  Max:  {np.max(input_norms):.6e}")

    print("\n=== Prediction Statistics ===")
    print(f"Prediction norms:")
    print(f"  Mean: {np.mean(pred_norms):.6e}")
    print(f"  Std:  {np.std(pred_norms):.6e}")
    print(f"  Min:  {np.min(pred_norms):.6e}")
    print(f"  Max:  {np.max(pred_norms):.6e}")

    print("\n=== Target Statistics ===")
    print(f"Target norms:")
    print(f"  Mean: {np.mean(target_norms):.6e}")
    print(f"  Std:  {np.std(target_norms):.6e}")
    print(f"  Min:  {np.min(target_norms):.6e}")
    print(f"  Max:  {np.max(target_norms):.6e}")

    print("\n=== Error Analysis ===")
    print(f"Absolute errors (L2 norm):")
    print(f"  Mean: {np.mean(abs_errors):.6e}")
    print(f"  Std:  {np.std(abs_errors):.6e}")
    print(f"  Min:  {np.min(abs_errors):.6e}")
    print(f"  Max:  {np.max(abs_errors):.6e}")

    print(f"\nRelative errors (abs_error / target_norm):")
    print(f"  Mean: {np.mean(rel_errors):.6e}")
    print(f"  Std:  {np.std(rel_errors):.6e}")
    print(f"  Min:  {np.min(rel_errors):.6e}")
    print(f"  Max:  {np.max(rel_errors):.6e}")

    print("\n=== Scale Consistency Check ===")
    scale_ratio = np.mean(pred_norms) / np.mean(target_norms)
    print(f"Scale ratio (pred/target): {scale_ratio:.4f}")

    # Check if predictions are reasonable
    print("\n=== VERDICT ===")

    if 0.1 < scale_ratio < 10.0:
        print(f"✓ Scale is REASONABLE (ratio {scale_ratio:.2f}x is within 0.1-10x)")
    else:
        print(f"✗ Scale is OFF (ratio {scale_ratio:.2f}x is outside 0.1-10x range)")

    if np.mean(rel_errors) < 2.0:
        print(f"✓ Relative error is GOOD (mean {np.mean(rel_errors):.2%} < 200%)")
    else:
        print(f"✗ Relative error is HIGH (mean {np.mean(rel_errors):.2%} > 200%)")

    if 0.5 < scale_ratio < 2.0 and np.mean(rel_errors) < 0.5:
        print("\n🎉 PREDICTIONS ARE GOOD! The normalization fix worked!")
        print("   Model outputs match target scale and have low error.")
    elif 0.1 < scale_ratio < 10.0:
        print("\n⚠️  PREDICTIONS ARE ACCEPTABLE but could be better")
        print("   Scale is reasonable but error might be high.")
    else:
        print("\n❌ PREDICTIONS ARE STILL RUBBISH")
        print("   Scale mismatch or high error persists.")

    print("=" * 70)

if __name__ == '__main__':
    main()
