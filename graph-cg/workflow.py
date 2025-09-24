#!/usr/bin/env python3
"""Simple workflow script to chain graph-cg processing steps."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer

from generate_data import generate_training_data
from train_model import train_model
from predict import run_inference
from compare_methods import compare_preconditioners


def run_full_workflow(
    config_path: str | Path = "./config-ffnn.toml",
    skip_generation: bool = False,
    skip_training: bool = False,
    skip_inference: bool = False,
    skip_comparison: bool = False
) -> dict:
    """Run the complete workflow programmatically.

    Args:
        config_path: Path to config file
        skip_generation: Skip data generation step
        skip_training: Skip training step
        skip_inference: Skip inference step
        skip_comparison: Skip comparison step

    Returns:
        Dictionary with all results
    """
    results = {}

    print(f"🚀 Starting graph-cg workflow with config: {config_path}")
    print("=" * 60)

    # Step 1: Generate training data
    if not skip_generation:
        print("📊 Step 1: Generating training data...")
        try:
            features_path, targets_path = generate_training_data(config_path)
            results['data_generation'] = {
                'features_path': features_path,
                'targets_path': targets_path
            }
            print(f"✅ Generated training data: {features_path}, {targets_path}")
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            raise
    else:
        print("⏭️  Step 1: Skipping data generation")

    # Step 2: Train model
    if not skip_training:
        print("\n🎯 Step 2: Training model...")
        try:
            checkpoint_path = train_model(config_path)
            results['training'] = {
                'checkpoint_path': checkpoint_path
            }
            print(f"✅ Model trained: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    else:
        print("⏭️  Step 2: Skipping training")

    # Step 3: Run inference
    if not skip_inference:
        print("\n🔮 Step 3: Running inference...")
        try:
            inference_results = run_inference(config_path)
            results['inference'] = inference_results
            print(f"✅ Inference completed in {inference_results['duration_seconds']:.2f}s")
            if inference_results['plot_path']:
                print(f"📊 Saved prediction plots: {inference_results['plot_path']}")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            raise
    else:
        print("⏭️  Step 3: Skipping inference")

    # Step 4: Compare preconditioners
    if not skip_comparison:
        print("\n⚖️  Step 4: Comparing preconditioners...")
        try:
            comparison_results = compare_preconditioners(config_path)
            results['comparison'] = comparison_results
            print(f"✅ Compared {len(comparison_results['preconditioners'])} preconditioners")
            if comparison_results['plot_paths']:
                print("📊 Generated comparison plots:")
                for plot_type, path in comparison_results['plot_paths'].items():
                    print(f"   • {plot_type}: {path}")
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            raise
    else:
        print("⏭️  Step 4: Skipping comparison")

    print("\n" + "=" * 60)
    print("🎉 Workflow completed successfully!")
    return results


def main(
    mode: str = typer.Argument(help="Mode: full, generate, train, predict, or compare"),
    config: Path = typer.Option(Path(__file__).parent / "config-ffnn.toml", help="Path to config file"),
    checkpoint: Optional[Path] = typer.Option(None, help="Override checkpoint path (for predict/compare modes)"),
    total_samples: int = typer.Option(6000, help="Number of samples to generate (generate mode only)"),
    skip_generation: bool = typer.Option(False, help="Skip data generation step (full mode only)"),
    skip_training: bool = typer.Option(False, help="Skip training step (full mode only)"),
    skip_inference: bool = typer.Option(False, help="Skip inference step (full mode only)"),
    skip_comparison: bool = typer.Option(False, help="Skip comparison step (full mode only)"),
):
    """Run graph-cg workflow in various modes."""
    try:
        if mode == "full":
            results = run_full_workflow(
                config_path=config,
                skip_generation=skip_generation,
                skip_training=skip_training,
                skip_inference=skip_inference,
                skip_comparison=skip_comparison
            )
            print(f"\n📋 Workflow Summary:")
            print(f"   • Steps completed: {len(results)}")
            for step, result in results.items():
                print(f"   • {step}: ✅")
        elif mode == "generate":
            features_path, targets_path = generate_training_data(
                config_path=config,
                total_samples=total_samples
            )
            print(f"✅ Generated training data:")
            print(f"   • Features: {features_path}")
            print(f"   • Targets: {targets_path}")
        elif mode == "train":
            checkpoint_path = train_model(config_path=config)
            print(f"✅ Model trained: {checkpoint_path}")
        elif mode == "predict":
            results = run_inference(config_path=config, checkpoint_path=checkpoint)
            print(f"✅ Inference completed in {results['duration_seconds']:.2f}s")
            if results['plot_path']:
                print(f"📊 Saved plots: {results['plot_path']}")
        elif mode == "compare":
            results = compare_preconditioners(config_path=config, checkpoint_path=checkpoint)
            print(f"✅ Compared {len(results['preconditioners'])} preconditioners")
            print("\n" + "=" * 40)
            print(results['summary'])
            print("=" * 40)
        else:
            print(f"❌ Invalid mode: {mode}. Valid modes: full, generate, train, predict, compare")
            raise typer.Exit(code=1)
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(130)
