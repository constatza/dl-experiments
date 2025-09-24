#!/usr/bin/env python3
"""Compare different preconditioner methods using src/ library modules."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import typer

from src.common import load_config, get_paths_from_config, get_solver_params, load_system_data, ensure_dir
from src.math_utils import normalize_system
from src.cg_algorithms import run_cg_comparison, format_results_summary
from src.preconditioner_factory import create_all_preconditioners
from src.noise_generators import create_noise_strategy
from src.plotting import plot_residual_history, plot_convergence_comparison
from src.validation import validate_matrix, validate_rhs, validate_solver_params


def compare_preconditioners(
    config_path: str | Path,
    matrix_path: Optional[str | Path] = None,
    rhs_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    noise_strategy: str = "none",
    noise_level: float = 0.05,
    noise_seed: Optional[int] = None,
    save_plots: bool = True
) -> Dict[str, Any]:
    """Compare preconditioners programmatically.

    Args:
        config_path: Path to config file
        matrix_path: Override matrix path
        rhs_path: Override RHS path
        checkpoint_path: Override checkpoint path
        output_dir: Override output directory
        noise_strategy: Noise strategy
        noise_level: Noise level parameter
        noise_seed: Random seed for noise
        save_plots: Whether to save plots

    Returns:
        Dictionary with comparison results
    """
    # Load config
    settings = load_config(config_path)
    paths = get_paths_from_config(settings)
    solver_params = get_solver_params(settings)

    # Use provided paths or get from config
    matrix_file = matrix_path or paths['matrix_path']
    rhs_file = rhs_path or paths['rhs_path']
    checkpoint_file = checkpoint_path or paths.get('checkpoint_path')
    output_path = Path(output_dir or paths.get('output_dir', './output'))

    if matrix_file is None or rhs_file is None:
        raise ValueError("Matrix and RHS paths must be specified")

    ensure_dir(output_path)

    # Load and validate system
    A, b = load_system_data(matrix_file, rhs_file)
    validate_matrix(A)
    validate_rhs(b, A)

    # Normalize system
    normalize = solver_params.get("normalize_system", True)
    A, b = normalize_system(A, b, enabled=normalize)

    # Apply noise if requested
    if noise_strategy != "none":
        noise_fn = create_noise_strategy(noise_strategy, noise_level, seed=noise_seed, A=A)
        b = noise_fn(b)

    # Extract solver parameters
    tol = solver_params["tolerance"]
    max_iter = solver_params["max_iterations"]
    stopping_criterion = solver_params["stopping_criterion"]

    validate_solver_params(tol, max_iter, stopping_criterion)

    # Create preconditioners
    preconditioners, warm_starts = create_all_preconditioners(
        A,
        checkpoint_path=checkpoint_file,
        config_path=config_path if checkpoint_file else None
    )

    # Run comparison
    results = run_cg_comparison(
        A, b,
        preconditioners=preconditioners,
        warm_starts=warm_starts,
        tol=tol,
        max_iter=max_iter,
        stopping_criterion=stopping_criterion
    )

    # Generate plots if requested
    plot_paths = {}
    if save_plots:
        try:
            convergence_path = output_path / "preconditioner_comparison.png"
            plot_convergence_comparison(results, save_path=convergence_path)
            plot_paths['convergence'] = convergence_path

            residual_path = output_path / "residual_history.png"
            plot_residual_history(results, save_path=residual_path)
            plot_paths['residual'] = residual_path

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    return {
        'results': results,
        'summary': format_results_summary(results),
        'plot_paths': plot_paths,
        'preconditioners': list(preconditioners.keys()),
        'warm_starts': list(warm_starts.keys()),
        'solver_params': solver_params
    }


def main(
    config: Path = typer.Option(Path(__file__).parent / "config-ffnn.toml", help="Path to TOML config"),
    matrix: Path | None = typer.Option(None, help="Override matrix path"),
    rhs: Path | None = typer.Option(None, help="Override RHS path"),
    checkpoint: Path | None = typer.Option(None, help="Override checkpoint path"),
    output_dir: Path | None = typer.Option(None, help="Override output directory"),
    noise_strategy: str = typer.Option("none", help="Noise strategy"),
    noise_level: float = typer.Option(0.05, help="Noise level parameter"),
    noise_seed: int | None = typer.Option(None, help="Random seed for noise"),
):
    """Compare preconditioner methods."""
    print(f"Loading configuration from: {config}")

    try:
        results = compare_preconditioners(
            config_path=config,
            matrix_path=matrix,
            rhs_path=rhs,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            noise_strategy=noise_strategy,
            noise_level=noise_level,
            noise_seed=noise_seed,
            save_plots=True
        )

        print(f"Available preconditioners: {results['preconditioners']}")
        print(f"Available warm starts: {results['warm_starts']}")

        # Print results summary
        print("\n" + "="*60)
        print(results['summary'])
        print("="*60)

        # Show plot paths
        if results['plot_paths']:
            print(f"\nGenerated plots:")
            for plot_type, path in results['plot_paths'].items():
                print(f"  {plot_type}: {path}")

        print(f"\n✅ Analysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(130)
