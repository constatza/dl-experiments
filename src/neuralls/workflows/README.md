# Workflows Module

The workflows package is the orchestration layer above generation, training,
prediction, and comparison. Users usually reach it through the CLI commands
first.

## Command To Workflow Map

| Command | Workflow layer |
| --- | --- |
| `process-data` | dataset processing helpers in `data.py` plus generation config processing |
| `train-model` | single-run training flow in `training.py` |
| `generate-all` | batch dataset generation in `multi_generation.py` |
| `run-experiments` | dataset + training matrix in `runner.py` |
| `train-all` | batch training summary in `multi_training.py` |
| `compare-all` | batch comparison orchestration in `comparison.py` |
| `predict` | inference flow in `prediction.py` and `inference/` |

## Basic To Advanced Progression

### Basic: one dataset, one model

Use:

- `process-data`
- `train-model`

These are the fastest commands for proving a new dataset or model config works.

### Intermediate: one registry

Use:

- `generate-all`
- `run-experiments`
- `train-all`

These commands treat the registry as the single orchestration input.

### Advanced: solver comparison

Use:

- `compare-all`

This command resolves comparison profiles, prepares tracking, constructs the
requested preconditioners, and executes the solver benchmarks.

## Package Map

- `runner.py`: combined dataset-generation and training matrix
- `training.py`: single-run training orchestration
- `multi_training.py`: batch training plus aggregate metrics
- `comparison.py`: registry-driven comparison batch execution
- `compare.py`: one comparison profile end-to-end
- `cg_runner.py`: low-level multi-preconditioner CG execution
- `prediction.py` and `inference/`: inference loading, execution, and outputs

## Why This Package Is Not A Split Candidate Yet

Unlike the solver package, workflows are still tightly coupled to:

- configuration models and registry resolution
- MLflow topology and run naming
- artifact paths and dataset storage
- plotting and diagnostics

That is why this package does not get a repo-split TODO for now.
