# Application Module

The application package holds use-case logic only.

## Package Map

- `comparison/execution.py`: CG execution and recommendation logic
- `inference/prediction.py`: predictor execution over prepared inference data
- `training/`: reserved package boundary for training use-case services
- `datasets/`: reserved package boundary for dataset use-case services

## Semantic Difference

Application code owns decisions about what a workflow does, not how runtime
systems are wired or where artifacts live. If a module starts resolving config
paths, creating MLflow clients, or touching filesystem layout, it belongs in
`composition` or `platform` instead.

## Boundary

Application modules do not load configs, resolve workspaces, create MLflow
runs, or touch filesystem layout concerns directly. Cross-layer DTOs no longer
live here; they live under `neuralls.shared`.
