"""Compare static and neural preconditioning metrics across experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer

from src.diagnostics import compute_condition_numbers, plot_condition_bars
from src.file_operations import ensure_dir, sanitize_identifier
from src.preconditioner_factory import create_all_preconditioners, create_preconditioner
from src.system_loading import _load_matrix_file, get_latest_checkpoint
from src.configuration.loader import load_experiments
from src.constants import DEFAULT_PROJECT_ROOT, DEFAULT_EXPERIMENTS_CONFIG

app = typer.Typer(add_completion=False, help=__doc__)


@dataclass
class ConditionRecord:
    dataset: str
    model: str
    config_name: str
    primary_preconditioner: str
    cond_original: float
    cond_preconditioned: float
    output_dir: Path
    all_conditions: dict[str, float]


def load_matrix(matrix_path: Path) -> np.ndarray:
    return np.asarray(_load_matrix_file(matrix_path), dtype=np.float64, copy=False)


def build_preconditioners(
    matrix: np.ndarray,
    *,
    checkpoint: Path | None,
    model_config_path: Path,
    data_config_path: Path,
    use_jacobi: bool,
    use_ilu: bool,
) -> dict[str, Any]:
    preconditioners, _ = create_all_preconditioners(
        matrix,
        checkpoint_path=checkpoint,
        config_path=model_config_path,
        data_config_path=data_config_path,
    )
    if not use_jacobi and "jacobi" in preconditioners:
        preconditioners.pop("jacobi")
    if not use_ilu and "ilu" in preconditioners:
        preconditioners.pop("ilu")
    if "none" not in preconditioners:
        preconditioners["none"] = create_preconditioner("none", matrix)
    return preconditioners


def select_primary(
    preconditioners: dict[str, Any], checkpoint: Path | None, fallback: str
) -> str:
    fallback_name = fallback.lower()
    if checkpoint is not None and "neural" in preconditioners:
        return "neural"
    if fallback_name in preconditioners:
        return fallback_name
    return "none"


def summarize_condition_numbers(
    matrix: np.ndarray,
    preconditioners: dict[str, Any],
    primary: str,
) -> tuple[float, float, dict[str, float]]:
    conds = compute_condition_numbers(matrix, preconditioners)
    cond_original = conds.get("none", float(np.linalg.cond(matrix)))
    cond_preconditioned = conds.get(primary, cond_original)
    return cond_original, cond_preconditioned, conds


def record_payload(record: ConditionRecord) -> dict[str, Any]:
    return {
        "dataset": record.dataset,
        "model": record.model,
        "config_name": record.config_name,
        "primary_preconditioner": record.primary_preconditioner,
        "cond_original": record.cond_original,
        "cond_preconditioned": record.cond_preconditioned,
        "output_dir": str(record.output_dir),
        "all_conditions": record.all_conditions,
    }


def save_record(
    record: ConditionRecord,
) -> None:
    diagnostics_dir = record.output_dir / "diagnostics"
    ensure_dir(diagnostics_dir)
    metrics_path = diagnostics_dir / "condition_numbers.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(record_payload(record), handle, indent=2)


def plot_summary(
    records: list[ConditionRecord],
    figures_root: Path,
    log_scale: bool = True,
) -> Path:
    labels = [
        f"{sanitize_identifier(r.dataset)} / {sanitize_identifier(r.config_name)}"
        for r in records
    ]
    cond_maps = [r.all_conditions for r in records]
    default_order = ["none", "jacobi", "ilu", "neural"]
    preconditioners = [
        name for name in default_order if any(name in m for m in cond_maps)
    ]
    if not preconditioners:
        preconditioners = sorted({k for m in cond_maps for k in m})
    figure_path = figures_root / "preconditioner_condition_numbers.png"
    return plot_condition_bars(
        labels, cond_maps, preconditioners, save_path=figure_path, log_scale=log_scale
    )


@app.command()
def main(
    experiments: Path = typer.Option(
        None, help="Path to experiments.toml (defaults to configs/experiments.toml)"
    ),
    fallback_preconditioner: str = typer.Option(
        "jacobi",
        help="Fallback static preconditioner when no neural checkpoint is available",
        case_sensitive=False,
    ),
    use_jacobi: bool = typer.Option(
        True,
        "--use-jacobi/--no-use-jacobi",
        help="Include Jacobi preconditioner in diagnostics",
    ),
    use_ilu: bool = typer.Option(
        True,
        "--use-ilu/--no-use-ilu",
        help="Include ILU preconditioner in diagnostics",
    ),
    log_scale: bool = typer.Option(
        True,
        "--log-scale/--no-log-scale",
        help="Plot condition numbers on a log scale",
    ),
    figures_dir: Path | None = typer.Option(
        None,
        help="Override figures output directory",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List resolved experiments without computing metrics",
    ),
) -> None:
    # Use loader to get all active experiments
    root_dir = DEFAULT_PROJECT_ROOT
    if experiments is None:
        experiments_path = root_dir / DEFAULT_EXPERIMENTS_CONFIG
    else:
        experiments_path = experiments
    
    experiments_path = Path(experiments_path)

    try:
        experiments_list = load_experiments(experiments_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading experiments: {exc}")
        raise typer.Exit(code=1)

    records: list[ConditionRecord] = []
    
    # We need a common output root for the summary plot if figures_dir isn't provided
    # We'll assume the first experiment's context provides a reasonable default if needed
    common_figures_root = figures_dir
    
    for exp_name, settings, context, model_path, data_path, solver_path in experiments_list:
        if common_figures_root is None:
             # Default to project-level figures dir from first experiment context
             common_figures_root = context.flow.roots.figures_root

        dataset_id = context.data.dataset_id
        model_name = context.run_id
        
        # Resolve paths via Context
        output_dir = context.training.base_dir
        
        # For matrix path, we need to look into the resolved settings or context
        # Context usually has the path to 'normalized.npz' in processed dir
        matrix_path = context.data.base_dir / "normalized.npz"
        
        # Checkpoint
        checkpoint = get_latest_checkpoint(context.training.checkpoint_dir)
        
        label = f"{sanitize_identifier(str(dataset_id))} / {sanitize_identifier(model_name)}"

        if dry_run:
            print(
                f"{label}: matrix={matrix_path} checkpoint={checkpoint or 'none'} -> {output_dir}"
            )
            continue

        if not matrix_path.exists():
             print(f"Skipping {label}: Matrix file not found at {matrix_path}")
             continue

        matrix = load_matrix(matrix_path)
        preconditioners = build_preconditioners(
            matrix,
            checkpoint=checkpoint,
            model_config_path=model_path,
            data_config_path=data_path,
            use_jacobi=use_jacobi,
            use_ilu=use_ilu,
        )
        primary = select_primary(preconditioners, checkpoint, fallback_preconditioner)
        cond_original, cond_precond, conds = summarize_condition_numbers(
            matrix, preconditioners, primary
        )

        record = ConditionRecord(
            dataset=str(dataset_id),
            model=model_name,
            config_name=exp_name,
            primary_preconditioner=primary,
            cond_original=cond_original,
            cond_preconditioned=cond_precond,
            output_dir=output_dir,
            all_conditions=conds,
        )
        ensure_dir(output_dir)
        save_record(record)
        records.append(record)
        print(
            f"{label}: cond={cond_original:.3e}, preconditioned={cond_precond:.3e} (primary={primary})"
        )

    if dry_run or not records:
        return

    if common_figures_root:
        ensure_dir(common_figures_root)
        figure_path = plot_summary(records, common_figures_root, log_scale=log_scale)
        print(f"Saved comparative figure to {figure_path}")


if __name__ == "__main__":
    app()