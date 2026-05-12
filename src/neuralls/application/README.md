# Application Module

The application package holds use-case logic only.

## Package Map

- `comparison/execution.py`: thin redirect to `domain/solver/comparison.py`
- `inference/prediction.py`: batch prediction execution over prepared inference data through a domain-owned inference predictor port
- `inference/models.py`: `InferenceConfig` and `InferenceData` DTOs
- `models.py`: `ExperimentResult` and `TaskResult` reporting DTOs
- `training/`: reserved package boundary for training use-case services

## Semantic Difference

Application code owns decisions about what a workflow does, not how runtime
systems are wired or where artifacts live. If a module starts resolving config
paths, creating MLflow clients, or touching filesystem layout, it belongs in
`composition` or `platform` instead.

## Boundary

Application modules do not load configs, resolve workspaces, create MLflow
runs, or touch filesystem layout concerns directly. Cross-layer DTOs that are
shared across multiple layers live in `domain/` or `shared/`, not here.
