#!/usr/bin/env python3
"""Debug script to understand dlkit inference output structure."""

import torch
import numpy as np
from dlkit import infer

def test_inference():
    """Test the dlkit inference directly to see what it returns."""
    checkpoint_path = "/data/projects/graph-cg/output/checkpoints/ffnn.ckpt"

    # Create test input
    test_vector = np.random.randn(24).astype(np.float64)
    input_tensor = torch.from_numpy(test_vector).float()
    # Match expected shape (24,) from checkpoint metadata
    if input_tensor.shape != (24,):
        input_tensor = input_tensor.reshape(24)

    # Try different input formats
    test_formats = [
        ("dict with x key", {"x": input_tensor}),
        ("tensor directly", input_tensor),
        ("numpy array", test_vector),
    ]

    for format_name, inputs in test_formats:
        print(f"\n=== Testing {format_name} ===")
        print(f"Input type: {type(inputs)}")
        if hasattr(inputs, 'shape'):
            print(f"Input shape: {inputs.shape}")
        elif isinstance(inputs, dict):
            print(f"Dict keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'no shape')}")

            # Call dlkit inference directly
            result = infer(
                checkpoint_path=checkpoint_path,
                inputs=inputs,
                batch_size=1
            )

            print(f"SUCCESS! Result type: {type(result)}")
            if hasattr(result, 'predictions'):
                pred = result.predictions
                print(f"Predictions type: {type(pred)}")
                if isinstance(pred, dict):
                    print(f"Predictions keys: {list(pred.keys())}")
                elif hasattr(pred, 'shape'):
                    print(f"Predictions shape: {pred.shape}")
            else:
                print("No predictions attribute found")
            break  # Exit loop on first success

        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    test_inference()