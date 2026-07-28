"""End-to-end integration test: a `run.type = "fit"` (POD-2G) assignment run for real.

Exercises the real `run_assignment_matrix()` path (no mocked `run_multirun_spec`)
against a self-contained synthetic case config: a small SPD matrix, a "random"
generation strategy (needs no pre-existing solution files), and a
`pod2g`-style `FitJobConfig` job binding `PODCoarseningFittable`. Verifies the
assignment succeeds, an MLflow run tagged with its assignment_id exists, and
that run carries a `checkpoints/` artifact — the concrete claim
`docs/plan.md`'s Phase B verification section asks for, not just "the code
imports without error."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tomli_w
from dlkit.interfaces.api.functions.model_registry import has_checkpoint_artifact
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.training_batch import run_assignment_matrix
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.config.settings import NeurallsSettings


@pytest.fixture
def pod_fit_case_config(tmp_path: Path) -> tuple[Path, str]:
    """Write a self-contained case config with one `run.type = "fit"` assignment.

    Returns the case config path and the MLflow tracking URI it declares.
    """
    n = 6
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((n, n))
    matrix = raw @ raw.T + n * np.eye(n)

    matrix_path = tmp_path / "matrix.txt"
    np.savetxt(matrix_path, matrix)

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    dataset_config_path = datasets_dir / "pod-snapshots.toml"
    dataset_config_path.write_text(
        "\n".join(
            [
                'id = "pod-snapshots"',
                "",
                "[source]",
                f'matrix_path = "{matrix_path.as_posix()}"',
                "",
                "[generation]",
                'normalize = "none"',
                "shuffle = false",
                "seed = 42",
                "",
                "[[generation.strategy]]",
                'name = "random"',
                "samples = 12",
                "",
                "[output]",
                f'data_dir = "{(tmp_path / "processed").as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    job_config_path = jobs_dir / "pod2g-test.toml"
    job_config_path.write_text(
        '[run]\ntype = "fit"\nseed = 42\n\n[model]\nname = "PODCoarseningFittable"\nmodule_path = "neuralls.composition.preconditioners.pod_fittable"\nrank = 2\n\n[data]\nname = "FlexibleDataset"\nbatch_size = 12\nnum_workers = 0\npin_memory = false\nshuffle = false\n\n[data.module]\nname = "ArrayDataModule"',
        encoding="utf-8",
    )

    tracking_uri = build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")
    case_config_path = tmp_path / "case.toml"
    with case_config_path.open("wb") as fh:
        tomli_w.dump(
            {
                "mlflow": {"tracking_uri": tracking_uri},
                "names": {"training": "PodFitIntegration-Training", "comparison": "unused"},
                "datasets": [{"id": "pod-snapshots", "path": "datasets/pod-snapshots.toml"}],
                "jobs": [{"id": "pod2g-test", "path": "jobs/pod2g-test.toml"}],
                "assignments": [
                    {"id": "pod2g_test", "dataset": "pod-snapshots", "job": "pod2g-test"}
                ],
            },
            fh,
        )
    return case_config_path, tracking_uri


def test_run_assignment_matrix_fits_pod_job_and_uploads_checkpoint(
    pod_fit_case_config: tuple[Path, str],
    tmp_path: Path,
) -> None:
    """A real (unmocked) `run_assignment_matrix()` call succeeds for a fit-kind job.

    Confirms the assignment resolves to `AssignmentResult(status="Success")`,
    that an MLflow run tagged `assignment_id=pod2g_test` was created for it,
    and that run has a non-empty `checkpoints/` artifact directory.
    """
    case_config_path, tracking_uri = pod_fit_case_config
    settings = NeurallsSettings(
        _env_file=[],
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )

    results = run_assignment_matrix(case_config_path, settings=settings, force=True)

    assert len(results) == 1
    assert results[0].status == "Success", results[0].error
    assert results[0].assignment_id == "pod2g_test"

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("PodFitIntegration-Training")
    assert experiment is not None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.assignment_id = 'pod2g_test'",
    )
    assert len(runs) == 1
    run_id = runs[0].info.run_id
    assert has_checkpoint_artifact(run_id, tracking_uri=tracking_uri)
