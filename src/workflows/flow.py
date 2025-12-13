"""Main workflow definition."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.futures import PrefectFuture
from prefect.task_runners import ConcurrentTaskRunner

from dlkit.interfaces.servers.mlflow_adapter import MLflowServerContext
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

from src.constants import DEFAULT_PROJECT_ROOT
from src.configuration.loader import load_experiments
from src.workflows.tasks.experiment import run_experiment_task
from src.workflows.utils.hashing import compute_directory_hash
from src.workflows.utils.paths import resolve_output_root

# Ensure necessary environment variables
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DASK_DISTRIBUTED__LOGGING__DISTRIBUTED", "error")


@flow(name="run_experiment_matrix", task_runner=ConcurrentTaskRunner(max_workers=1))
def run_experiment_matrix_flow(
    experiments_config_path: str | Path | None = None,
    force: bool = False,
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all experiments defined in the configuration system."""
    if project_root is None:
        project_root = DEFAULT_PROJECT_ROOT

    if experiments_config_path is None:
        master_config_path = project_root / "configs" / "experiments.toml"
    else:
        master_config_path = Path(experiments_config_path)

    experiments = load_experiments(master_config_path)

    print(f"\nFound {len(experiments)} experiments to run:")
    for idx, (exp_name, _, _, _, _, _) in enumerate(experiments, 1):
        print(f"  {idx}. {exp_name}")
    print()

    output_root = resolve_output_root(None)

    src_dir = project_root / "src"
    if not src_dir.exists():
        # Fallback
        src_dir = Path(__file__).resolve().parent.parent

    src_hash = compute_directory_hash(src_dir)
    print(f"Source code hash: {src_hash[:12]}...")

    mlruns_dir = project_root / "data" / "mlruns"
    mlartifacts_dir = project_root / "data" / "mlartifacts"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlartifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"MLflow tracking: {mlruns_dir}")
    print(f"MLflow artifacts: {mlartifacts_dir}")

    mlruns_db_path = mlruns_dir.resolve() / "mlflow.db"
    mlflow_server_config = MLflowServerSettings(
        host="127.0.0.1",
        port=5000,
        backend_store_uri=f"sqlite:///{mlruns_db_path.as_posix()}",
        artifacts_destination=str(mlartifacts_dir.resolve()),
    )

    with MLflowServerContext(server_config=mlflow_server_config) as server_info:
        print(f"\nMLflow server started at: {server_info.url}")
        print("Submitting experiments sequentially...")
        print(f"Output root: {output_root}")
        if force:
            print("Force mode: Ignoring filesystem checks, re-running all tasks")

        futures: dict[str, PrefectFuture] = {}
        for exp_name, _, _, model_path, data_gen_path, solver_path in experiments:
            future = run_experiment_task.submit(
                experiment_name=exp_name,
                model_config_path=model_path,
                data_gen_config_path=data_gen_path,
                solver_config_path=solver_path,
                output_root=output_root,
                src_hash=src_hash,
                force=force,
            )
            futures[exp_name] = future
            print(f"  ✓ Submitted: {exp_name}")

        print(f"\nWaiting for {len(futures)} experiments to complete...")
        print("(Running sequentially to prevent memory exhaustion)\n")

        results: dict[str, dict[str, Any]] = {}
        failed_experiments: list[tuple[str, Exception]] = []

        for experiment_id, future in futures.items():
            try:
                experiment_results = future.result()
                results[experiment_id] = experiment_results
                print(f"  ✓ Completed: {experiment_id}")
            except Exception as e:
                print(f"  ✗ Failed: {experiment_id} - {e}")
                failed_experiments.append((experiment_id, e))

        print(f"\n{'=' * 80}")
        print(
            f"Experiment matrix completed! "
            f"{len(results)}/{len(experiments)} succeeded, "
            f"{len(failed_experiments)} failed"
        )
        print(f"{ '=' * 80}")

        if failed_experiments:
            print("\nFailed experiments:")
            for experiment_id, error in failed_experiments:
                print(f"  ✗ {experiment_id}: {error}")

        # Create summary artifact
        experiment_data = []
        for exp_name, _, _, model_path, data_gen_path, solver_path in experiments:
            status = "✓ Success" if exp_name in results else "✗ Failed"
            details = "OK"
            if exp_name not in results:
                error_msg = next(
                    (str(err) for failed_id, err in failed_experiments if failed_id == exp_name),
                    "Unknown error",
                )
                details = error_msg

            experiment_data.append(
                {
                    "Experiment": exp_name,
                    "Status": status,
                    "Model": model_path.name,
                    "Data Gen": data_gen_path.name,
                    "Solver": solver_path.name,
                    "Details": details,
                }
            )

        create_table_artifact(
            key="experiment-results-summary",
            table=experiment_data,
            description=f"Experiment matrix results: {len(results)}/{len(experiments)} succeeded",
        )

        return results
