# MLflow Artifact Logging Plan

## Goals
- Shift artifact ownership to MLflow while keeping existing filesystem saves intact.
- Use functional helpers, minimal abstractions, and delegate lifecycle to MLflow.
- Keep only project-root as the single absolute anchor; everything else relative.

## Design
- Add `src/mlflow_utils.py` with small, focused functions:
  - `resolve_mlflow_paths(tracking_uri, project_root, workspace)` resolves relative URIs against the output layout; leaves absolutes untouched.
  - `make_run_labels(dataset_id, model_name, session_name)` builds names/tags.
  - `artifact_groups(workspace)` returns existing paths to upload (checkpoints, figures, predictions, comparison reports, metrics).
  - `start_run_if_needed(exp_name, run_name, tags)` reuses `mlflow.active_run()` or starts one (nested allowed via MLflow); returns handle and started-flag.
  - `log_artifacts(run, artifacts)` logs grouped dirs/files via `mlflow.log_artifacts`.
  - `log_metrics(run, metrics, step)` thin pass-through.
- No custom managers; let MLflow handle nesting and run lifecycle.
- Keep an allowlist of artifact dirs to avoid uploading raw data by default.

## Integration (opt-in flag)
- Training CLI:
  - After loading settings/workspace, resolve MLflow URIs; bail out if MLflow disabled/missing.
  - Start/reuse run, execute training, log artifact groups, log final metrics if available, end run only if we started it (status FINISHED/FAILED).
- Prediction/comparison scripts:
  - Same pattern: start/reuse, log predictions/reports/figures, end if we started.
- Preserve current saves on disk; MLflow uploads are additive.

## Config handling
- Normalize relative `MLFLOW.server.backend_store_uri` and `artifacts_destination` against the resolved output root/workspace.
- Leave absolute URIs untouched.

## Testing/Docs
- Unit tests with mocked `mlflow`: active-run reuse, start/end behavior, artifact path mapping, relative URI resolution.
- Short README/usage note: enabling MLflow logging, what gets uploaded, local saves remain.

## Rollout
1) Implement helpers + train integration behind a flag.
2) Extend to predict/compare.
3) Refine artifact allowlist if needed.
